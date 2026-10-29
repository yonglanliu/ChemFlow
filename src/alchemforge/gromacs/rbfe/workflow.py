"""Staging and execution for ligand RBFE with GROMACS."""

from __future__ import annotations

import argparse
import csv
import shlex
import shutil
import subprocess
from pathlib import Path

from alchemforge.gromacs.membrane import apply_membrane_pressure

from .charge import build_charge_plan
from .config import LigandRBFEConfig
from .hybrid import stage_hybrid_systems
from .mapping import NetworkPlan, existing_network_plan, plan_network
from .mdp import render_rbfe_stage_mdp


def _directive(value: str, label: str) -> str:
    """Validate a user-provided value before placing it in an SBATCH line."""
    if "\n" in value or "\r" in value:
        raise ValueError(f"{label} cannot contain a newline")
    return value


def render_slurm_array_script(
    config: LigandRBFEConfig,
    *,
    expected_tasks: int,
    equilibration_only: bool = False,
) -> str:
    """Render an independent staged protocol for each FEP window."""
    partition = (
        f"#SBATCH --partition={_directive(config.slurm_partition, 'Partition')}\n"
        if config.slurm_partition.strip()
        else ""
    )
    gres = (
        f"#SBATCH --gres=gpu:{config.slurm_gpus}\n"
        if config.slurm_gpus > 0
        else ""
    )
    activation = ""
    if config.slurm_conda_init.strip():
        activation += f"source {shlex.quote(config.slurm_conda_init)}\n"
    if config.slurm_conda_environment.strip():
        activation += (
            "conda activate "
            f"{shlex.quote(config.slurm_conda_environment)}\n"
        )
    if equilibration_only and config.hrex_module.strip():
        module_name = _directive(config.hrex_module.strip(), "HREX module")
        activation += (
            'type module >/dev/null 2>&1 || { echo "ERROR: module command is unavailable" >&2; exit 127; }\n'
            f"module load {shlex.quote(module_name)}\n"
        )
    mdrun_options = " ".join(
        shlex.quote(option) for option in config.mdrun_options
    )
    gmx_command = " ".join(shlex.quote(part) for part in shlex.split(config.gmx))
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={_directive(config.job_name, 'Job name')}"
        + ("_equil\n" if equilibration_only else "\n")
        + partition
        + gres
        + f"#SBATCH --nodes={config.slurm_nodes}\n"
        + f"#SBATCH --ntasks={1 if equilibration_only else config.slurm_tasks}\n"
        + f"#SBATCH --cpus-per-task={config.slurm_cpus_per_task}\n"
        + f"#SBATCH --mem={_directive(config.slurm_memory, 'Memory')}\n"
        + f"#SBATCH --time={_directive(config.slurm_walltime, 'Walltime')}\n"
        + f"#SBATCH --array=0-{expected_tasks - 1}\n\n"
        + "set -euo pipefail\n\n"
        + 'TASK_FILE="${1:?Provide production_tasks.txt}"\n'
        + 'TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Not running as a SLURM array}"\n\n'
        + activation
        + f'export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{config.slurm_cpus_per_task}}}"\n\n'
        + "# sed uses one-based line numbers; SLURM array indices are zero-based.\n"
        + 'TASK_DIR="$(sed -n "$((TASK_INDEX + 1))p" "$TASK_FILE")"\n\n'
        + 'if [[ -z "$TASK_DIR" || ! -d "$TASK_DIR" ]]; then\n'
        + '    echo "ERROR: No accessible FEP task directory for array index $TASK_INDEX" >&2\n'
        + "    exit 1\n"
        + "fi\n\n"
        + 'LEG_DIR="$(dirname "$(dirname "$(dirname "$TASK_DIR")")")"\n'
        + 'BASE_GRO="$LEG_DIR/setup/system.gro"\n'
        + 'TOPOLOGY="$LEG_DIR/setup/topol.top"\n'
        + 'for required in "$BASE_GRO" "$TOPOLOGY" "$TASK_DIR/em.mdp" '
        + '"$TASK_DIR/nvt.mdp" "$TASK_DIR/npt.mdp" '
        + '"$TASK_DIR/transition.mdp" "$TASK_DIR/equil.mdp"'
        + ('; do\n' if equilibration_only else ' "$TASK_DIR/prod.mdp"; do\n')
        + '    [[ -s "$required" ]] || { echo "ERROR: Missing $required" >&2; exit 1; }\n'
        + "done\n\n"
        + 'cd "$TASK_DIR"\n\n'
        + 'echo "Job:             $SLURM_JOB_ID"\n'
        + 'echo "Array task:      $SLURM_ARRAY_TASK_ID"\n'
        + 'echo "Node:            $(hostname)"\n'
        + 'echo "Task directory:  $TASK_DIR"\n'
        + 'echo "Visible GPU(s):  ${CUDA_VISIBLE_DEVICES:-not set}"\n'
        + "nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader\n\n"
        + f"GMX_COMMAND=( {gmx_command} )\n"
        + 'if ! command -v "${GMX_COMMAND[0]}" >/dev/null 2>&1; then\n'
        + '    echo "ERROR: GROMACS executable not found: ${GMX_COMMAND[0]}" >&2\n'
        + '    echo "Set the GROMACS command in Job Setting to an installed executable (usually gmx)." >&2\n'
        + "    exit 127\n"
        + "fi\n"
        + f"MDRUN_COMMON=( -ntomp \"$OMP_NUM_THREADS\" {mdrun_options} )\n\n"
        + "MDRUN_EM=( -ntmpi 1 -ntomp 1 -nb cpu -pme cpu )\n\n"
        + "validate_minimization() {\n"
        + '    local log_file="$1"\n'
        + '    if [[ ! -s "$log_file" ]] || grep -Eqi '
        + "'force on at least one atom is not finite|Maximum force[[:space:]]*=[[:space:]]*(inf|nan)|Potential Energy[[:space:]]*=[[:space:]]*(inf|nan)' \"$log_file\"; then\n"
        + '        echo "ERROR: Invalid energy minimization in $log_file; check for overlapping atoms." >&2\n'
        + "        exit 1\n"
        + "    fi\n"
        + "}\n\n"
        + "run_stage() {\n"
        + '    local stage="$1"\n'
        + '    local coordinates="$2"\n'
        + '    local checkpoint="${3:-}"\n'
        + '    local restraint_reference="$BASE_GRO"\n'
        + '    if [[ "$stage" != "em" && -s "em.gro" ]]; then\n'
        + '        restraint_reference="em.gro"\n'
        + "    fi\n"
        + '    if [[ -s "${stage}.gro" ]]; then\n'
        + '        [[ "$stage" == "em" ]] && validate_minimization "${stage}.log"\n'
        + '        echo "Skipping completed ${stage}: ${stage}.gro exists"\n'
        + "        return\n"
        + "    fi\n"
        + '    if [[ ! -s "${stage}.tpr" ]]; then\n'
        + '        grompp_args=( grompp -f "${stage}.mdp" -c "$coordinates" '
        + '-r "$restraint_reference" -p "$TOPOLOGY" -o "${stage}.tpr" )\n'
        + '        if [[ -n "$checkpoint" && -s "$checkpoint" ]]; then\n'
        + '            grompp_args+=( -t "$checkpoint" )\n'
        + "        fi\n"
        + '        if ! "${GMX_COMMAND[@]}" "${grompp_args[@]}"; then\n'
        + '            echo "ERROR: grompp failed for ${stage}; ${stage}.tpr was not created" >&2\n'
        + "            exit 1\n"
        + "        fi\n"
        + "    fi\n"
        + '    [[ -s "${stage}.tpr" ]] || { echo "ERROR: Missing ${stage}.tpr after grompp" >&2; exit 1; }\n'
        + '    if [[ "$stage" == "em" ]]; then\n'
        + '        mdrun_args=( mdrun -s "${stage}.tpr" -deffnm "$stage" "${MDRUN_EM[@]}" )\n'
        + "    else\n"
        + '        mdrun_args=( mdrun -s "${stage}.tpr" -deffnm "$stage" "${MDRUN_COMMON[@]}" )\n'
        + "    fi\n"
        + '    if [[ -s "${stage}.cpt" ]]; then\n'
        + '        mdrun_args+=( -cpi "${stage}.cpt" -append )\n'
        + "    fi\n"
        + '    if [[ "$stage" == "em" ]]; then\n'
        + '        OMP_NUM_THREADS=1 "${GMX_COMMAND[@]}" "${mdrun_args[@]}"\n'
        + "    else\n"
        + '        "${GMX_COMMAND[@]}" "${mdrun_args[@]}"\n'
        + "    fi\n"
        + '    [[ -s "${stage}.gro" ]] || { '
        + 'echo "ERROR: ${stage} did not produce ${stage}.gro" >&2; exit 1; }\n'
        + '    [[ "$stage" == "em" ]] && validate_minimization "${stage}.log"\n'
        + "}\n\n"
        + 'run_stage em "$BASE_GRO"\n'
        + 'run_stage nvt em.gro\n'
        + 'run_stage npt nvt.gro nvt.cpt\n'
        + 'run_stage transition npt.gro npt.cpt\n'
        + 'run_stage equil transition.gro transition.cpt\n'
        + ("" if equilibration_only else 'run_stage prod equil.gro equil.cpt\n')
    )


def _without_thread_mpi_options(options: tuple[str, ...]) -> tuple[str, ...]:
    """Remove rank/thread counts that HREX must derive from its replica count."""
    result: list[str] = []
    skip_value = False
    for option in options:
        if skip_value:
            skip_value = False
            continue
        if option in {"-ntmpi", "-ntomp"}:
            skip_value = True
            continue
        result.append(option)
    return tuple(result)


def render_slurm_hrex_script(
    config: LigandRBFEConfig, *, expected_tasks: int
) -> str:
    """Render one coordinated lambda-HREX job per edge/leg/replicate."""
    partition = (
        f"#SBATCH --partition={_directive(config.slurm_partition, 'Partition')}\n"
        if config.slurm_partition.strip()
        else ""
    )
    gres = (
        f"#SBATCH --gres=gpu:{config.slurm_gpus}\n"
        if config.slurm_gpus > 0
        else ""
    )
    activation = ""
    if config.slurm_conda_init.strip():
        activation += f"source {shlex.quote(config.slurm_conda_init)}\n"
    if config.slurm_conda_environment.strip():
        activation += f"conda activate {shlex.quote(config.slurm_conda_environment)}\n"
    options = " ".join(
        shlex.quote(option)
        for option in _without_thread_mpi_options(config.mdrun_options)
    )
    gmx_command = " ".join(
        shlex.quote(part) for part in shlex.split(config.hrex_gmx_command)
    )
    prep_command = " ".join(
        shlex.quote(part) for part in shlex.split(config.gmx)
    )
    module_load = ""
    if config.hrex_module.strip():
        module_name = _directive(config.hrex_module.strip(), "HREX module")
        module_load = (
            'if ! type module >/dev/null 2>&1; then\n'
            '    echo "ERROR: The environment-modules command is unavailable; '
            f'cannot load {shlex.quote(module_name)}." >&2\n'
            "    exit 127\n"
            "fi\n"
            f"module load {shlex.quote(module_name)}\n"
        )
    if config.hrex_mpi_launcher == "mpirun":
        mpi_launcher = 'MPI_LAUNCHER=( mpirun -np "$REPLICA_COUNT" )\n'
    else:
        mpi_launcher = 'MPI_LAUNCHER=( srun --ntasks="$REPLICA_COUNT" )\n'
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={_directive(config.job_name, 'Job name')}_hrex\n"
        + partition
        + gres
        + f"#SBATCH --nodes={config.slurm_nodes}\n"
        + f"#SBATCH --ntasks={config.lambda_windows}\n"
        + "#SBATCH --cpus-per-task=1\n"
        + "#SBATCH --ntasks-per-core=1\n"
        + f"#SBATCH --mem={_directive(config.slurm_memory, 'Memory')}\n"
        + f"#SBATCH --time={_directive(config.slurm_walltime, 'Walltime')}\n"
        + f"#SBATCH --array=0-{expected_tasks - 1}\n\n"
        + "set -euo pipefail\n\n"
        + 'TASK_FILE="${1:?Provide production_tasks.txt}"\n'
        + 'TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Not running as a SLURM array}"\n'
        + f"EXPECTED_REPLICAS={config.lambda_windows}\n"
        + f"EXCHANGE_INTERVAL={config.exchange_interval_steps}\n\n"
        + activation
        + module_load
        + 'FEP_DIR="$(sed -n "$((TASK_INDEX + 1))p" "$TASK_FILE")"\n'
        + '[[ -d "$FEP_DIR" ]] || { echo "ERROR: Missing HREX directory: $FEP_DIR" >&2; exit 1; }\n'
        + 'mapfile -t REPLICA_DIRS < <(find "$FEP_DIR" -maxdepth 1 -type d '
        + "-name 'lambda_*' -print | sort)\n"
        + 'REPLICA_COUNT="${#REPLICA_DIRS[@]}"\n'
        + 'if (( REPLICA_COUNT != EXPECTED_REPLICAS )); then\n'
        + '    echo "ERROR: Found $REPLICA_COUNT lambda replicas; expected $EXPECTED_REPLICAS" >&2\n'
        + "    exit 1\n"
        + "fi\n"
        + mpi_launcher
        + 'OMP_PER_REPLICA="${SLURM_CPUS_PER_TASK:-1}"\n'
        + 'export OMP_NUM_THREADS="$OMP_PER_REPLICA"\n'
        + 'LEG_DIR="$(dirname "$(dirname "$FEP_DIR")")"\n'
        + 'BASE_GRO="$LEG_DIR/setup/system.gro"\n'
        + 'TOPOLOGY="$LEG_DIR/setup/topol.top"\n'
        + '[[ -s "$BASE_GRO" && -s "$TOPOLOGY" ]] || { echo "ERROR: Missing leg setup" >&2; exit 1; }\n\n'
        + f"GMX_PREP_COMMAND=( {prep_command} )\n"
        + f"GMX_COMMAND=( {gmx_command} )\n"
        + 'if ! command -v "${GMX_PREP_COMMAND[0]}" >/dev/null 2>&1; then\n'
        + '    echo "ERROR: GROMACS preparation executable not found: ${GMX_PREP_COMMAND[0]}" >&2\n'
        + '    echo "Set the GROMACS command in Job Setting to an installed executable (usually gmx)." >&2\n'
        + "    exit 127\n"
        + "fi\n"
        + 'if ! command -v "${GMX_COMMAND[0]}" >/dev/null 2>&1; then\n'
        + '    echo "ERROR: HREX executable not found: ${GMX_COMMAND[0]}" >&2\n'
        + '    echo "HREX requires an external-MPI GROMACS build, normally gmx_mpi." >&2\n'
        + "    exit 127\n"
        + "fi\n"
        + 'if ! command -v "${MPI_LAUNCHER[0]}" >/dev/null 2>&1; then\n'
        + '    echo "ERROR: HREX MPI launcher not found: ${MPI_LAUNCHER[0]}" >&2\n'
        + "    exit 127\n"
        + "fi\n"
        + 'if ! GMX_VERSION="$("${GMX_COMMAND[@]}" --version 2>&1)"; then\n'
        + '    echo "ERROR: Could not run ${GMX_COMMAND[*]} --version" >&2\n'
        + '    echo "$GMX_VERSION" >&2\n'
        + "    exit 1\n"
        + "fi\n"
        + 'if grep -Eqi "MPI version:.*built in" <<<"$GMX_VERSION"; then\n'
        + '    echo "ERROR: Lambda-HREX requires an external-MPI GROMACS build (normally gmx_mpi); thread-MPI is not supported by -multidir." >&2\n'
        + "    exit 1\n"
        + "fi\n"
        + f"MDRUN_COMMON=( {options} -ntomp \"$OMP_PER_REPLICA\" )\n\n"
        + "prepare_stage() {\n"
        + '    local stage="$1" previous="$2" checkpoint="${3:-}"\n'
        + '    for replica in "${REPLICA_DIRS[@]}"; do\n'
        + '        [[ -s "$replica/${stage}.mdp" ]] || { echo "ERROR: Missing $replica/${stage}.mdp" >&2; exit 1; }\n'
        + '        [[ -s "$replica/${stage}.tpr" ]] && continue\n'
        + '        local coordinates="$BASE_GRO"\n'
        + '        [[ "$previous" != "BASE" ]] && coordinates="$replica/${previous}.gro"\n'
        + '        local restraint_reference="$BASE_GRO"\n'
        + '        [[ -s "$replica/em.gro" ]] && restraint_reference="$replica/em.gro"\n'
        + '        args=( grompp -f "$replica/${stage}.mdp" -c "$coordinates" -r "$restraint_reference" -p "$TOPOLOGY" -o "$replica/${stage}.tpr" )\n'
        + '        if [[ -n "$checkpoint" && -s "$replica/${checkpoint}.cpt" ]]; then args+=( -t "$replica/${checkpoint}.cpt" ); fi\n'
        + '        if ! "${GMX_PREP_COMMAND[@]}" "${args[@]}"; then\n'
        + '            echo "ERROR: grompp failed for $replica/${stage}" >&2\n'
        + "            exit 1\n"
        + "        fi\n"
        + '        [[ -s "$replica/${stage}.tpr" ]] || { echo "ERROR: Missing $replica/${stage}.tpr after grompp" >&2; exit 1; }\n'
        + "    done\n"
        + "}\n\n"
        + "run_hrex_production() {\n"
        + '    local stage="prod" complete=1\n'
        + '    for replica in "${REPLICA_DIRS[@]}"; do [[ -s "$replica/${stage}.gro" ]] || complete=0; done\n'
        + '    if (( complete == 1 )); then\n'
        + '        echo "Skipping completed HREX production"\n'
        + "        return\n"
        + "    fi\n"
        + '    args=( mdrun -multidir "${REPLICA_DIRS[@]}" -deffnm "$stage" "${MDRUN_COMMON[@]}" )\n'
        + '    args+=( -replex "$EXCHANGE_INTERVAL" )\n'
        + '    if [[ -s "${REPLICA_DIRS[0]}/${stage}.cpt" ]]; then args+=( -cpi "${stage}.cpt" -append ); fi\n'
        + '    "${MPI_LAUNCHER[@]}" "${GMX_COMMAND[@]}" "${args[@]}"\n'
        + "}\n\n"
        + 'echo "HREX production group: $FEP_DIR; replicas: $REPLICA_COUNT; exchange every $EXCHANGE_INTERVAL steps"\n'
        + 'for replica in "${REPLICA_DIRS[@]}"; do\n'
        + '    [[ -s "$replica/equil.gro" ]] || { echo "ERROR: Independent equilibration is incomplete: $replica/equil.gro" >&2; exit 1; }\n'
        + 'done\n'
        + 'prepare_stage prod equil equil\nrun_hrex_production\n'
    )


def stage_fep_task_directories(
    config: LigandRBFEConfig, network: NetworkPlan
) -> list[Path]:
    """Create one sequential six-stage task per FEP window."""
    with network.edges_csv.open(newline="", encoding="utf-8") as handle:
        edge_ids = [row["edge_id"].strip() for row in csv.DictReader(handle)]
    tasks: list[Path] = []
    for edge_id in edge_ids:
        for leg in ("complex", "solvent"):
            membrane = leg == "complex" and config.membrane.enabled
            for replicate in range(
                config.start_replicate,
                config.start_replicate + config.replicates,
            ):
                for window in range(config.lambda_windows):
                    task = (
                        config.project_dir
                        / "edges"
                        / edge_id
                        / leg
                        / f"rep_{replicate}"
                        / "fep"
                        / f"lambda_{window:03d}"
                    )
                    task.mkdir(parents=True, exist_ok=True)
                    for filename, stage_name in (
                        ("em.mdp", "mini"),
                        ("nvt.mdp", "nvt"),
                        ("npt.mdp", "npt"),
                        ("transition.mdp", "transition"),
                        ("equil.mdp", "final"),
                        ("prod.mdp", "prod"),
                    ):
                        (task / filename).write_text(
                            render_rbfe_stage_mdp(
                                config,
                                window,
                                stage_name,
                                membrane=membrane,
                                position_restraints=leg == "complex",
                            ),
                            encoding="utf-8",
                        )
                    tasks.append(task)
    return tasks


def write_slurm_array_files(
    config: LigandRBFEConfig, *, edge_count: int, tasks: list[Path] | None = None
) -> tuple[Path, Path, Path]:
    """Write staged equilibration/production manifests and Slurm runners."""
    project = config.project_dir
    project.mkdir(parents=True, exist_ok=True)
    expected_tasks = edge_count * config.tasks_per_edge
    window_tasks = list(tasks or [])
    group_tasks = list(dict.fromkeys(task.parent for task in window_tasks))
    equilibration_tasks = (
        edge_count * 2 * config.replicates * config.lambda_windows
    )
    summary = (
        f"edges={edge_count}\n"
        "legs=2\n"
        f"replicates={config.replicates}\n"
        f"lambda_windows={config.lambda_windows}\n"
        f"sampling_method={config.sampling_method}\n"
        f"exchange_interval_steps={config.exchange_interval_steps}\n"
        f"tasks_per_edge={config.tasks_per_edge}\n"
        f"total_tasks={expected_tasks}\n"
        f"array=0-{expected_tasks - 1}\n"
    )
    if config.sampling_method == "hrex":
        summary += (
            f"equilibration_tasks={equilibration_tasks}\n"
            f"equilibration_array=0-{equilibration_tasks - 1}\n"
        )
    (project / "slurm_array_summary.txt").write_text(summary, encoding="utf-8")
    task_file = project / "production_tasks.txt"
    manifest_tasks = group_tasks if config.sampling_method == "hrex" else window_tasks
    task_file.write_text(
        "".join(f"{task.resolve()}\n" for task in manifest_tasks), encoding="utf-8"
    )

    array_script = project / "gromacs_production_array.sh"
    array_script.write_text(
        (
            render_slurm_hrex_script(config, expected_tasks=expected_tasks)
            if config.sampling_method == "hrex"
            else render_slurm_array_script(config, expected_tasks=expected_tasks)
        ),
        encoding="utf-8",
    )
    array_script.chmod(array_script.stat().st_mode | 0o100)

    submit_script = project / "submit_production_array.sh"
    if config.sampling_method == "hrex":
        equil_task_file = project / "equilibration_tasks.txt"
        equil_task_file.write_text(
            "".join(f"{task.resolve()}\n" for task in window_tasks),
            encoding="utf-8",
        )
        equil_script = project / "gromacs_equilibration_array.sh"
        equil_script.write_text(
            render_slurm_array_script(
                config,
                expected_tasks=equilibration_tasks,
                equilibration_only=True,
            ),
            encoding="utf-8",
        )
        equil_script.chmod(equil_script.stat().st_mode | 0o100)
        submit_text = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            f"PROJECT_DIR={shlex.quote(str(project))}\n"
            f"EQUIL_TASK_FILE={shlex.quote(str(equil_task_file))}\n"
            f"EQUIL_SCRIPT={shlex.quote(str(equil_script))}\n"
            f"PROD_TASK_FILE={shlex.quote(str(task_file))}\n"
            f"PROD_SCRIPT={shlex.quote(str(array_script))}\n"
            f"EXPECTED_EQUIL_TASKS={equilibration_tasks}\n"
            f"EXPECTED_PROD_TASKS={expected_tasks}\n"
            f"WINDOWS_PER_GROUP={config.lambda_windows}\n\n"
            'N_EQUIL="$(awk \'NF {count++} END {print count+0}\' "$EQUIL_TASK_FILE")"\n'
            'N_PROD="$(awk \'NF {count++} END {print count+0}\' "$PROD_TASK_FILE")"\n'
            '[[ "$N_EQUIL" -eq "$EXPECTED_EQUIL_TASKS" ]] || { echo "ERROR: Found $N_EQUIL equilibration tasks; expected $EXPECTED_EQUIL_TASKS" >&2; exit 1; }\n'
            '[[ "$N_PROD" -eq "$EXPECTED_PROD_TASKS" ]] || { echo "ERROR: Found $N_PROD HREX production groups; expected $EXPECTED_PROD_TASKS" >&2; exit 1; }\n'
            '[[ "$N_EQUIL" -eq "$((N_PROD * WINDOWS_PER_GROUP))" ]] || { echo "ERROR: Equilibration tasks are not grouped into $WINDOWS_PER_GROUP windows per HREX system" >&2; exit 1; }\n'
            'cd "$PROJECT_DIR"\n'
            'echo "Submitting $N_PROD independent system groups; each HREX production starts after only its own $WINDOWS_PER_GROUP lambda equilibrations"\n'
            'ALL_JOB_IDS=()\n'
            'for ((GROUP_INDEX=0; GROUP_INDEX<N_PROD; GROUP_INDEX++)); do\n'
            '    EQUIL_START=$((GROUP_INDEX * WINDOWS_PER_GROUP))\n'
            '    EQUIL_END=$((EQUIL_START + WINDOWS_PER_GROUP - 1))\n'
            '    EQUIL_ARRAY="${EQUIL_START}-${EQUIL_END}"\n'
            '    PROD_ARRAY="${GROUP_INDEX}-${GROUP_INDEX}"\n'
            '    EQUIL_SUBMISSION="$(sbatch --parsable --array="$EQUIL_ARRAY" "$EQUIL_SCRIPT" "$EQUIL_TASK_FILE")"\n'
            '    EQUIL_JOB_ID="${EQUIL_SUBMISSION%%;*}"\n'
            '    PROD_SUBMISSION="$(sbatch --parsable --dependency="afterok:${EQUIL_JOB_ID}" --array="$PROD_ARRAY" "$PROD_SCRIPT" "$PROD_TASK_FILE")"\n'
            '    PROD_JOB_ID="${PROD_SUBMISSION%%;*}"\n'
            '    ALL_JOB_IDS+=("$EQUIL_JOB_ID" "$PROD_JOB_ID")\n'
            '    echo "Group $((GROUP_INDEX + 1))/$N_PROD: equilibration $EQUIL_JOB_ID [$EQUIL_ARRAY] -> HREX $PROD_JOB_ID [$PROD_ARRAY]"\n'
            'done\n'
            'JOB_ID_LIST="$(IFS=,; echo "${ALL_JOB_IDS[*]}")"\n'
            'echo "Inspect all: sacct -X -j $JOB_ID_LIST --format=JobID,State,ExitCode,Reason,Elapsed,NodeList"\n'
        )
    else:
        submit_text = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            f"PROJECT_DIR={shlex.quote(str(project))}\n"
            f"TASK_FILE={shlex.quote(str(task_file))}\n"
            f"ARRAY_SCRIPT={shlex.quote(str(array_script))}\n"
            f"EXPECTED_TASKS={expected_tasks}\n\n"
            "# Task formula: edges x 2 legs x replicates x lambda windows\n"
            'N_TASKS="$(awk \'NF {count++} END {print count+0}\' "$TASK_FILE")"\n'
            '[[ "$N_TASKS" -eq "$EXPECTED_TASKS" ]] || { echo "ERROR: Found $N_TASKS FEP tasks; expected $EXPECTED_TASKS" >&2; exit 1; }\n'
            'cd "$PROJECT_DIR"\n'
            'ARRAY_SPEC="0-$((N_TASKS - 1))"\n'
            'echo "Submitting $N_TASKS independent six-stage FEP windows as array $ARRAY_SPEC"\n'
            'SUBMISSION="$(sbatch --parsable --array="$ARRAY_SPEC" "$ARRAY_SCRIPT" "$TASK_FILE")"\n'
            'JOB_ID="${SUBMISSION%%;*}"\n'
            'echo "Submitted job array: $JOB_ID ($ARRAY_SPEC)"\n'
            'echo "Combined output: $PROJECT_DIR/slurm-${JOB_ID}_<array-index>.out"\n'
            'echo "Inspect: sacct -X -j $JOB_ID --format=JobID,State,ExitCode,Reason,Elapsed,NodeList"\n'
        )
    submit_script.write_text(submit_text, encoding="utf-8")
    submit_script.chmod(submit_script.stat().st_mode | 0o100)
    return task_file, array_script, submit_script


def stage(
    config: LigandRBFEConfig, *, reuse_network: bool = False
) -> tuple[Path, NetworkPlan]:
    project = config.project_dir
    inputs = project / "input"
    internal = project / ".alchemforge"
    inputs.mkdir(parents=True, exist_ok=True)
    internal.mkdir(parents=True, exist_ok=True)
    if config.existing_network_dir is not None:
        destination = project / "network"
        if config.existing_network_dir.resolve() != destination.resolve():
            shutil.copytree(
                config.existing_network_dir, destination, dirs_exist_ok=True
            )
        reuse_network = True

    mdp_dir = project / "mdp"
    mdp_dir.mkdir(parents=True, exist_ok=True)
    for leg, membrane, position_restraints in (
        ("complex", config.membrane.enabled, True),
        ("solvent", False, False),
    ):
        leg_dir = mdp_dir / leg
        leg_dir.mkdir(parents=True, exist_ok=True)
        for filename, stage_name in (
            ("em.mdp", "mini"),
            ("nvt.mdp", "nvt"),
            ("npt.mdp", "npt"),
            ("transition.mdp", "transition"),
            ("equil.mdp", "final"),
            ("prod.mdp", "prod"),
        ):
            (leg_dir / filename).write_text(
                render_rbfe_stage_mdp(
                    config,
                    0,
                    stage_name,
                    membrane=membrane,
                    position_restraints=position_restraints,
                ),
                encoding="utf-8",
            )

    # Keep the established top-level paths as complex-leg templates so older
    # worker scripts continue to function. New workers should select the MDP
    # directory by leg via RBFE_COMPLEX_MDP_DIR/RBFE_SOLVENT_MDP_DIR.
    for filename in (
        "em.mdp",
        "nvt.mdp",
        "npt.mdp",
        "transition.mdp",
        "equil.mdp",
        "prod.mdp",
    ):
        shutil.copy2(
            mdp_dir / "complex" / filename,
            mdp_dir / filename,
        )

    sources = [
        config.protein,
        config.ligand_library,
        config.prepared_complex_gro,
        config.prepared_complex_topology,
        config.ligand_charge_manifest,
        config.charge_correction_file,
    ]
    for source in sources:
        if source is None:
            continue
        target = inputs / source.name
        if source.resolve() != target.resolve() and not target.exists():
            shutil.copy2(source, target)

    if config.worker_script is not None:
        worker = project / "LigandRBFE_GROMACS.sh"
        if config.worker_script.resolve() != worker.resolve():
            shutil.copy2(config.worker_script, worker)
        worker.chmod(worker.stat().st_mode | 0o100)

    generated = internal / "ligand_rbfe_config.sh"
    if config.membrane.enabled:
        (internal / "membrane_pressure.mdp").write_text(
            apply_membrane_pressure("", config.membrane), encoding="utf-8"
        )
    network = existing_network_plan(config) if reuse_network else plan_network(config)
    build_charge_plan(
        edges_csv=network.edges_csv,
        ligand_library=config.ligand_library,
        output_directory=project / "charge_correction",
        method=config.charge_change_method,
        charge_manifest=config.ligand_charge_manifest,
        correction_file=config.charge_correction_file,
        force_field=config.force_field,
    )
    stage_hybrid_systems(config, network)
    tasks = stage_fep_task_directories(config, network)
    write_slurm_array_files(
        config, edge_count=network.edge_count, tasks=tasks
    )
    generated.write_text(config.render_shell_config(), encoding="utf-8")
    return generated, network


def _load(args: argparse.Namespace) -> LigandRBFEConfig:
    return LigandRBFEConfig.load(args.config)


def _validate(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs(require_runtime=args.for_execution)
    print(f"Valid ligand-RBFE configuration: {config.source}")
    print(f"Protein: {config.protein or config.prepared_complex_gro}")
    print(f"System type: {config.membrane.system_type}")
    print(f"Ligand library: {config.ligand_library}")
    print(f"Bound + solvent tasks per edge: {config.tasks_per_edge}")
    return 0


def _stage(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs()
    generated, network = stage(config)
    print(f"Staged ligand-RBFE project: {config.project_dir}")
    print(f"Generated shell configuration: {generated}")
    print(
        f"Planned network: {network.ligand_count} ligands, {network.edge_count} edges"
    )
    print(f"Edge table: {network.edges_csv}")
    hybrid_builder = config.project_dir / "build_hybrid_systems.sh"
    if hybrid_builder.is_file():
        print(f"Hybrid complex + solvent builder: {hybrid_builder}")
    return 0


def _run(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs(require_runtime=True)
    generated, _ = stage(config)
    worker = config.project_dir / "LigandRBFE_GROMACS.sh"
    return subprocess.run(
        ["bash", str(worker), str(generated)], check=False, cwd=config.project_dir
    ).returncode


def _render_mdp(args: argparse.Namespace) -> int:
    config = _load(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    membrane = config.membrane.enabled and args.leg == "complex"
    args.output.write_text(
        render_rbfe_stage_mdp(
            config,
            args.window,
            args.stage,
            membrane=membrane,
            position_restraints=args.leg == "complex",
        ),
        encoding="utf-8",
    )
    print(
        f"Rendered {args.leg} {args.stage} lambda window "
        f"{args.window}: {args.output}"
    )
    return 0


def add_ligand_rbfe_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "ligand-rbfe", help="Ligand relative binding free energy with GROMACS"
    )
    commands = parser.add_subparsers(dest="ligand_rbfe_command", required=True)

    validate = commands.add_parser("validate", help="Validate configuration")
    validate.add_argument("config", type=Path)
    validate.add_argument("--for-execution", action="store_true")
    validate.set_defaults(func=_validate)

    stage_parser = commands.add_parser("stage", help="Stage inputs and manifest")
    stage_parser.add_argument("config", type=Path)
    stage_parser.set_defaults(func=_stage)

    run = commands.add_parser("run", help="Run the configured GROMACS worker")
    run.add_argument("config", type=Path)
    run.set_defaults(func=_run)

    render = commands.add_parser(
        "render-mdp",
        help="Render one task-local MDP for a selected lambda window",
    )
    render.add_argument("config", type=Path)
    render.add_argument("--window", type=int, required=True)
    render.add_argument(
        "--stage", choices=("mini", "nvt", "npt", "prod"), default="prod"
    )
    render.add_argument("--leg", choices=("complex", "solvent"), default="complex")
    render.add_argument("--output", type=Path, required=True)
    render.set_defaults(func=_render_mdp)
    return parser
