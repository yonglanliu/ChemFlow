"""Staging and execution for ligand RBFE with GROMACS."""

from __future__ import annotations

import argparse
import shutil
import shlex
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
    config: LigandRBFEConfig, *, expected_tasks: int
) -> str:
    """Render one-GPU-per-TPR SLURM array execution for production windows."""
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
    mdrun_options = "\n".join(
        f"    {shlex.quote(option)}" for option in config.mdrun_options
    )
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={_directive(config.job_name, 'Job name')}\n"
        + partition
        + gres
        + f"#SBATCH --nodes={config.slurm_nodes}\n"
        + f"#SBATCH --ntasks={config.slurm_tasks}\n"
        + f"#SBATCH --cpus-per-task={config.slurm_cpus_per_task}\n"
        + f"#SBATCH --mem={_directive(config.slurm_memory, 'Memory')}\n"
        + f"#SBATCH --time={_directive(config.slurm_walltime, 'Walltime')}\n"
        + f"#SBATCH --output={_directive(config.slurm_log_output, 'Log output')}\n"
        + f"#SBATCH --error={_directive(config.slurm_log_error, 'Log error')}\n"
        + f"#SBATCH --array=0-{expected_tasks - 1}%{config.max_parallel_jobs}\n\n"
        + "set -euo pipefail\n\n"
        + 'TASK_FILE="${1:?Provide production_tasks.txt}"\n'
        + 'TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Not running as a SLURM array}"\n\n'
        + activation
        + f'export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{config.slurm_cpus_per_task}}}"\n\n'
        + "# sed uses one-based line numbers; SLURM array indices are zero-based.\n"
        + 'TPR_FILE="$(sed -n "$((TASK_INDEX + 1))p" "$TASK_FILE")"\n\n'
        + 'if [[ -z "$TPR_FILE" || ! -f "$TPR_FILE" ]]; then\n'
        + '    echo "ERROR: No accessible TPR for array index $TASK_INDEX" >&2\n'
        + "    exit 1\n"
        + "fi\n\n"
        + 'RUN_DIR="$(dirname "$TPR_FILE")"\n'
        + 'RUN_NAME="$(basename "$TPR_FILE" .tpr)"\n\n'
        + 'cd "$RUN_DIR"\n\n'
        + 'echo "Job:             $SLURM_JOB_ID"\n'
        + 'echo "Array task:      $SLURM_ARRAY_TASK_ID"\n'
        + 'echo "Node:            $(hostname)"\n'
        + 'echo "TPR:             $TPR_FILE"\n'
        + 'echo "Visible GPU(s):  ${CUDA_VISIBLE_DEVICES:-not set}"\n'
        + "nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader\n\n"
        + "MDRUN_ARGS=(\n"
        + '    -s "$TPR_FILE"\n'
        + '    -deffnm "$RUN_NAME"\n'
        + '    -ntomp "$OMP_NUM_THREADS"\n'
        + mdrun_options
        + "\n)\n\n"
        + "# Continue an interrupted window when a checkpoint exists.\n"
        + 'if [[ -f "${RUN_NAME}.cpt" ]]; then\n'
        + "    MDRUN_ARGS+=(\n"
        + '        -cpi "${RUN_NAME}.cpt"\n'
        + "        -append\n"
        + "    )\n"
        + "fi\n\n"
        + f"{shlex.quote(config.gmx)} mdrun \"${{MDRUN_ARGS[@]}}\"\n"
    )


def write_slurm_array_files(
    config: LigandRBFEConfig, *, edge_count: int
) -> tuple[Path, Path, Path]:
    """Write the production task manifest, array runner, and submit helper."""
    project = config.project_dir
    project.mkdir(parents=True, exist_ok=True)
    (project / "logs").mkdir(parents=True, exist_ok=True)
    expected_tasks = edge_count * config.tasks_per_edge
    (project / "slurm_array_summary.txt").write_text(
        f"edges={edge_count}\n"
        "legs=2\n"
        f"replicates={config.replicates}\n"
        f"lambda_windows={config.lambda_windows}\n"
        f"tasks_per_edge={config.tasks_per_edge}\n"
        f"total_tasks={expected_tasks}\n"
        f"maximum_simultaneous_tasks={config.max_parallel_jobs}\n"
        f"array=0-{expected_tasks - 1}%{config.max_parallel_jobs}\n",
        encoding="utf-8",
    )
    task_file = project / "production_tasks.txt"
    tasks = sorted(project.rglob("prod.tpr"))
    task_file.write_text(
        "".join(f"{task.resolve()}\n" for task in tasks), encoding="utf-8"
    )

    array_script = project / "gromacs_production_array.sh"
    array_script.write_text(
        render_slurm_array_script(config, expected_tasks=expected_tasks),
        encoding="utf-8",
    )
    array_script.chmod(array_script.stat().st_mode | 0o100)

    submit_script = project / "submit_production_array.sh"
    submit_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"PROJECT_DIR={shlex.quote(str(project))}\n"
        f"TASK_FILE={shlex.quote(str(task_file))}\n"
        f"ARRAY_SCRIPT={shlex.quote(str(array_script))}\n"
        f"MAX_PARALLEL={config.max_parallel_jobs}\n\n"
        f"EXPECTED_TASKS={expected_tasks}\n\n"
        'mkdir -p "$PROJECT_DIR/logs"\n'
        'find "$PROJECT_DIR" -type f -name prod.tpr -print '
        '| LC_ALL=C sort > "$TASK_FILE"\n'
        'N_TASKS="$(awk \'NF {count++} END {print count+0}\' "$TASK_FILE")"\n'
        'if (( N_TASKS == 0 )); then\n'
        '    echo "ERROR: No prod.tpr files exist under $PROJECT_DIR." >&2\n'
        '    echo "Build the EM/NVT/NPT/production TPR files before submission." >&2\n'
        "    exit 1\n"
        "fi\n"
        'if (( N_TASKS != EXPECTED_TASKS )); then\n'
        '    echo "ERROR: Found $N_TASKS prod.tpr files; expected '
        '$EXPECTED_TASKS (= edges x 2 legs x replicates x lambda windows)." >&2\n'
        "    exit 1\n"
        "fi\n"
        'cd "$PROJECT_DIR"\n'
        'ARRAY_SPEC="0-$((N_TASKS - 1))%${MAX_PARALLEL}"\n'
        'echo "Submitting $N_TASKS production tasks as array $ARRAY_SPEC"\n'
        'sbatch --array="$ARRAY_SPEC" "$ARRAY_SCRIPT" "$TASK_FILE"\n',
        encoding="utf-8",
    )
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
    for leg, membrane in (
        ("complex", config.membrane.enabled),
        ("solvent", False),
    ):
        leg_dir = mdp_dir / leg
        leg_dir.mkdir(parents=True, exist_ok=True)
        for stage_name in ("mini", "nvt", "npt", "prod"):
            (leg_dir / f"{stage_name}.mdp").write_text(
                render_rbfe_stage_mdp(
                    config, 0, stage_name, membrane=membrane
                ),
                encoding="utf-8",
            )
        shutil.copy2(leg_dir / "mini.mdp", leg_dir / "em.mdp")

    # Keep the established top-level paths as complex-leg templates so older
    # worker scripts continue to function. New workers should select the MDP
    # directory by leg via RBFE_COMPLEX_MDP_DIR/RBFE_SOLVENT_MDP_DIR.
    for stage_name in ("mini", "nvt", "npt", "prod"):
        shutil.copy2(
            mdp_dir / "complex" / f"{stage_name}.mdp",
            mdp_dir / f"{stage_name}.mdp",
        )
    shutil.copy2(mdp_dir / "mini.mdp", mdp_dir / "em.mdp")

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
    write_slurm_array_files(config, edge_count=network.edge_count)
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
            config, args.window, args.stage, membrane=membrane
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
