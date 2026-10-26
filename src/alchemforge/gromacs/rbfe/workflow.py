"""Staging and execution for ligand RBFE with GROMACS."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from alchemforge.gromacs.membrane import apply_membrane_pressure

from .charge import build_charge_plan
from .config import LigandRBFEConfig
from .hybrid import stage_hybrid_systems
from .mapping import NetworkPlan, existing_network_plan, plan_network
from .mdp import render_rbfe_stage_mdp


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
