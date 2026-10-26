"""Plan, prepare, and execute GROMACS SepTop ligand-RBFE calculations."""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from alchemforge.gromacs.membrane import MembraneSettings, apply_membrane_pressure
from alchemforge.gromacs.rbfe.mapping import (
    NetworkPlan,
    existing_network_plan,
    plan_network,
)

from .config import SepTopGromacsConfig
from .prepare import prepare_edge

PHASES = {
    "complex_step1": (
        "septop_complex/MDP_step1",
        "complex/complex.gro",
        "complex/complex_1_scale.top",
        "complex/complex_1_scale.top",
    ),
    "complex_step2": (
        "septop_complex/MDP_step2",
        "complex/complex.gro",
        "complex/complex_2_scale.top",
        "complex/complex_2_scale.top",
    ),
    "solvent_step1": (
        "septop_solvent/MDP_step1",
        "solvent/solvent.gro",
        "solvent/step1_eq.top",
        "solvent/step1.top",
    ),
    "solvent_step2": (
        "septop_solvent/MDP_step2",
        "solvent/solvent.gro",
        "solvent/step2_eq.top",
        "solvent/step2.top",
    ),
}


@dataclass(frozen=True)
class StageResult:
    network: NetworkPlan
    tasks: Path
    task_count: int
    slurm_script: Path


def _replace_mdp_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^\s*{re.escape(key)}\s*=.*$")
    replacement = f"{key:<24} = {value}"
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text + f"\n{replacement}\n"


def _copy_mdp(
    config: SepTopGromacsConfig,
    template_dir: Path,
    state: int,
    destination: Path,
    phase: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for stage_name in ("em", "nvt", "prod"):
        source = template_dir / "complete_files" / f"{stage_name}.{state}.mdp"
        if not source.is_file():
            raise ValueError(f"Missing reference SepTop MDP: {source}")
        text = source.read_text(encoding="utf-8")
        text = _replace_mdp_value(text, "dt", str(config.timestep_ps))
        if stage_name == "nvt":
            text = _replace_mdp_value(text, "nsteps", str(config.equilibration_steps))
            text = _replace_mdp_value(text, "gen_seed", "-1")
        elif stage_name == "prod":
            text = _replace_mdp_value(text, "nsteps", str(config.production_steps))
            if config.system_type == "membrane" and phase.startswith("complex_"):
                text = apply_membrane_pressure(text, _membrane_settings(config))
        (destination / f"{stage_name}.mdp").write_text(text, encoding="utf-8")
    if (
        config.system_type == "membrane"
        and phase.startswith("complex_")
        and config.pressure_equilibration_steps > 0
    ):
        text = (destination / "prod.mdp").read_text(encoding="utf-8")
        text = _replace_mdp_value(
            text, "nsteps", str(config.pressure_equilibration_steps)
        )
        text = _replace_mdp_value(text, "continuation", "yes")
        text = _replace_mdp_value(text, "gen-vel", "no")
        (destination / "npt.mdp").write_text(text, encoding="utf-8")


def _membrane_settings(config: SepTopGromacsConfig) -> MembraneSettings:
    return MembraneSettings(
        enabled=config.system_type == "membrane",
        normal=config.membrane_normal,
        lipid_resnames=config.lipid_resnames,
        minimum_lipids=config.minimum_lipids,
        barostat=config.membrane_barostat,
        reference_pressure_bar=config.membrane_reference_pressure_bar,
        compressibility_bar=config.membrane_compressibility_bar,
        refcoord_scaling=config.membrane_refcoord_scaling,
    )


def _read_edges(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_edge_charges(
    config: SepTopGromacsConfig, edges: list[dict[str, str]]
) -> None:
    if config.allow_charge_change:
        return
    from rdkit import Chem

    charges: dict[str, int] = {}
    for molecule in Chem.SDMolSupplier(str(config.ligand_library), removeHs=False):
        if molecule is not None and molecule.HasProp("_Name"):
            charges[molecule.GetProp("_Name").strip()] = Chem.GetFormalCharge(molecule)
    changed = [
        f"{row['ligand_a']}({charges[row['ligand_a']]}) -> "
        f"{row['ligand_b']}({charges[row['ligand_b']]})"
        for row in edges
        if charges[row["ligand_a"]] != charges[row["ligand_b"]]
    ]
    if changed:
        raise ValueError(
            "SepTop network contains charge-changing edge(s): "
            + ", ".join(changed)
            + ". Split the network by formal charge or explicitly set "
            "[protocol].allow_charge_change=true and apply an appropriate correction."
        )


def stage(config: SepTopGromacsConfig, *, reuse_network: bool = False) -> StageResult:
    config.project_dir.mkdir(parents=True, exist_ok=True)
    if config.existing_network_dir is not None:
        destination = config.project_dir / "network"
        if config.existing_network_dir.resolve() != destination.resolve():
            shutil.copytree(
                config.existing_network_dir, destination, dirs_exist_ok=True
            )
        reuse_network = True
    network = (
        existing_network_plan(config) if reuse_network else plan_network(config)
    )  # SepTop uses the graph, not the atom maps.
    edges = _read_edges(network.edges_csv)
    _validate_edge_charges(config, edges)
    ligand_names = {row[key] for row in edges for key in ("ligand_a", "ligand_b")}
    config.validate_prepared_ligands(ligand_names)

    mdp_root = config.septop_source_dir / "MDP_files"
    rows: list[dict[str, str | int]] = []
    task_id = 0
    for edge_row in edges:
        edge_id = edge_row["edge_id"]
        edge_dir = config.project_dir / "edges" / edge_id
        for phase, (
            template,
            coordinate,
            equil_topology,
            prod_topology,
        ) in PHASES.items():
            template_dir = mdp_root / template
            states = sorted(
                int(path.stem.split(".")[-1])
                for path in (template_dir / "complete_files").glob("prod.*.mdp")
            )
            if not states:
                raise ValueError(f"No lambda-window MDP files found in {template_dir}")
            for replicate in range(
                config.start_replicate, config.start_replicate + config.replicates
            ):
                for state in states:
                    task_dir = (
                        edge_dir
                        / "windows"
                        / phase
                        / f"rep{replicate}"
                        / f"lambda_{state:03d}"
                    )
                    _copy_mdp(config, template_dir, state, task_dir, phase)
                    rows.append(
                        {
                            "task_id": task_id,
                            "edge_id": edge_id,
                            "ligand_a": edge_row["ligand_a"],
                            "ligand_b": edge_row["ligand_b"],
                            "phase": phase,
                            "replicate": replicate,
                            "lambda_state": state,
                            "task_dir": str(task_dir),
                            "coordinate": str(edge_dir / coordinate),
                            "equil_topology": str(edge_dir / equil_topology),
                            "production_topology": str(edge_dir / prod_topology),
                        }
                    )
                    task_id += 1

    internal = config.project_dir / ".alchemforge"
    internal.mkdir(exist_ok=True)
    tasks = internal / "septop_tasks.csv"
    with tasks.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    slurm_script = internal / "submit_septop.sh"
    partition = (
        f"#SBATCH --partition={config.slurm_partition}\n"
        if config.slurm_partition
        else ""
    )
    gpu = f"#SBATCH --gpus={config.slurm_gpus}\n" if config.slurm_gpus else ""
    logs = config.project_dir / "logs"
    logs.mkdir(exist_ok=True)
    slurm_script.write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH --job-name={config.job_name}_septop\n"
        f"#SBATCH --array=0-{len(rows) - 1}%{config.max_parallel_jobs}\n"
        f"#SBATCH --time={config.slurm_time}\n"
        f"#SBATCH --nodes={config.slurm_nodes}\n"
        f"#SBATCH --ntasks={config.slurm_tasks}\n"
        f"#SBATCH --cpus-per-task={config.slurm_cpus}\n"
        f"#SBATCH --mem={config.slurm_memory}\n"
        f"#SBATCH --output={logs}/septop_%A_%a.out\n"
        f"#SBATCH --error={logs}/septop_%A_%a.err\n"
        + partition
        + gpu
        + "set -euo pipefail\n"
        + f'python -m alchemforge septop-gromacs run-task {shlex.quote(str(config.source))} --task-id "${{SLURM_ARRAY_TASK_ID}}"\n',
        encoding="utf-8",
    )
    slurm_script.chmod(slurm_script.stat().st_mode | 0o100)
    return StageResult(network, tasks, len(rows), slurm_script)


def _task(config: SepTopGromacsConfig, task_id: int) -> dict[str, str]:
    tasks = config.project_dir / ".alchemforge" / "septop_tasks.csv"
    with tasks.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["task_id"]) == task_id:
                return row
    raise ValueError(f"SepTop task ID does not exist: {task_id}")


def run_task(config: SepTopGromacsConfig, task_id: int) -> int:
    row = _task(config, task_id)
    task_dir = Path(row["task_dir"])
    coordinate = Path(row["coordinate"])
    equil_topology = Path(row["equil_topology"])
    production_topology = Path(row["production_topology"])
    for required in (coordinate, equil_topology, production_topology):
        if not required.is_file():
            raise ValueError(f"Run `septop-gromacs prepare` first; missing {required}")
    stages = [
        ("em", coordinate, equil_topology, None),
        ("nvt", task_dir / "em.gro", equil_topology, task_dir / "em.cpt"),
    ]
    if (task_dir / "npt.mdp").is_file():
        stages.append(
            ("npt", task_dir / "nvt.gro", production_topology, task_dir / "nvt.cpt")
        )
        production_coordinates = task_dir / "npt.gro"
        production_checkpoint = task_dir / "npt.cpt"
    else:
        production_coordinates = task_dir / "nvt.gro"
        production_checkpoint = task_dir / "nvt.cpt"
    stages.append(
        ("prod", production_coordinates, production_topology, production_checkpoint)
    )
    for name, coordinates, topology, checkpoint in stages:
        grompp = [
            config.gmx,
            "grompp",
            "-f",
            f"{name}.mdp",
            "-c",
            str(coordinates),
            "-r",
            str(coordinates),
            "-p",
            str(topology),
            "-o",
            f"{name}.tpr",
        ]
        if checkpoint is not None:
            grompp.extend(["-t", str(checkpoint)])
        subprocess.run(grompp, cwd=task_dir, check=True)
        subprocess.run(
            [config.gmx, "mdrun", "-deffnm", name, *config.mdrun_options],
            cwd=task_dir,
            check=True,
        )
    return 0


def _load(args: argparse.Namespace) -> SepTopGromacsConfig:
    return SepTopGromacsConfig.load(args.config)


def _validate(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs(runtime=args.for_preparation)
    print(f"Valid GROMACS SepTop configuration: {config.source}")
    print("Input model: multi-ligand SDF plus prepared complex/solvent systems")
    print(f"System type: {config.system_type}")
    return 0


def _stage(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs()
    result = stage(config)
    print(
        f"Staged {result.network.edge_count} SepTop edges and {result.task_count} window tasks"
    )
    print(f"Task table: {result.tasks}")
    print(f"Slurm script: {result.slurm_script}")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs(runtime=True)
    edges_path = config.project_dir / "network" / "edges.csv"
    if not edges_path.is_file():
        stage(config)
    for row in _read_edges(edges_path):
        if args.edge and row["edge_id"] != args.edge:
            continue
        print(f"Preparing {row['edge_id']}: {row['ligand_a']} -> {row['ligand_b']}")
        prepare_edge(config, row["edge_id"], row["ligand_a"], row["ligand_b"])
    return 0


def _run_task(args: argparse.Namespace) -> int:
    return run_task(_load(args), args.task_id)


def _submit(args: argparse.Namespace) -> int:
    config = _load(args)
    script = config.project_dir / ".alchemforge" / "submit_septop.sh"
    if not script.is_file():
        script = stage(config).slurm_script
    return subprocess.run(["sbatch", str(script)], check=False).returncode


def add_septop_gromacs_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "septop-gromacs",
        aliases=["septop_gromacs"],
        help="Separated-topologies ligand RBFE with GROMACS",
    )
    commands = parser.add_subparsers(dest="septop_command", required=True)
    validate = commands.add_parser("validate", help="Validate SepTop inputs")
    validate.add_argument("config", type=Path)
    validate.add_argument("--for-preparation", action="store_true")
    validate.set_defaults(func=_validate)
    staging = commands.add_parser(
        "stage", help="Plan the network, windows, and Slurm array"
    )
    staging.add_argument("config", type=Path)
    staging.set_defaults(func=_stage)
    prepare = commands.add_parser(
        "prepare", help="Generate separated topologies and restraints"
    )
    prepare.add_argument("config", type=Path)
    prepare.add_argument("--edge")
    prepare.set_defaults(func=_prepare)
    task = commands.add_parser("run-task", help=argparse.SUPPRESS)
    task.add_argument("config", type=Path)
    task.add_argument("--task-id", type=int, required=True)
    task.set_defaults(func=_run_task)
    submit = commands.add_parser("submit", help="Submit the generated Slurm array")
    submit.add_argument("config", type=Path)
    submit.set_defaults(func=_submit)
    return parser
