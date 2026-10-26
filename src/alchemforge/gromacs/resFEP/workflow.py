"""Python orchestration around the established residue-FEP shell backend."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from alchemforge.gromacs.membrane import apply_membrane_pressure

from .config import ResFEPConfig

PACKAGE_DIR = Path(__file__).resolve().parent


def stage(config: ResFEPConfig) -> Path:
    """Create reproducible project inputs and return the generated shell config."""
    project_dir = config.project_dir
    mdp_dir = project_dir / "mdp"
    internal_dir = project_dir / ".alchemforge"
    mdp_dir.mkdir(parents=True, exist_ok=True)
    internal_dir.mkdir(parents=True, exist_ok=True)

    for name in ("ions.mdp", "fep_base.mdp"):
        target = mdp_dir / name
        if not target.exists():
            shutil.copy2(PACKAGE_DIR / "mdp" / name, target)

    if config.membrane.enabled:
        membrane_dir = project_dir / "input" / "membrane"
        membrane_dir.mkdir(parents=True, exist_ok=True)
        for source in (
            config.prepared_complex_gro,
            config.prepared_complex_topology,
        ):
            assert source is not None
            target = membrane_dir / source.name
            if source.resolve() != target.resolve() and not target.exists():
                shutil.copy2(source, target)
        membrane_mdp = mdp_dir / "membrane_pressure.mdp"
        membrane_mdp.write_text(
            apply_membrane_pressure("", config.membrane), encoding="utf-8"
        )

    if any((config.ligand_itp, config.ligand_gro, config.ligand_top)):
        config.staged_ligand_dir.mkdir(parents=True, exist_ok=True)
        ligand_files = (
            (config.ligand_itp, f"{config.ligand_resname}_GMX.itp"),
            (config.ligand_gro, f"{config.ligand_resname}_GMX.gro"),
            (config.ligand_top, f"{config.ligand_resname}_GMX.top"),
        )
        for source, name in ligand_files:
            target = config.staged_ligand_dir / name
            if (
                source is not None
                and source.resolve() != target.resolve()
                and not target.exists()
            ):
                shutil.copy2(source, target)

    if config.worker_script is not None:
        target = project_dir / "ResAlchemFEP.sh"
        shutil.copy2(config.worker_script, target)
        target.chmod(target.stat().st_mode | 0o100)

    shell_config = internal_dir / "resfep_config.sh"
    shell_config.write_text(config.render_shell_config(), encoding="utf-8")
    return shell_config


def run_backend(
    config: ResFEPConfig, script_name: str, *, require_worker: bool = False
) -> int:
    config.validate_inputs(require_worker=require_worker)
    shell_config = stage(config)
    environment = os.environ.copy()
    if config.environment_setup is not None:
        environment["ALCHEMFORGE_ENV_SETUP"] = str(config.environment_setup)
    command = ["bash", str(PACKAGE_DIR / script_name), str(shell_config)]
    return subprocess.run(command, check=False, env=environment).returncode


def _load(args: argparse.Namespace) -> ResFEPConfig:
    return ResFEPConfig.load(args.config)


def _validate(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs(require_worker=args.for_submission)
    print(f"Valid residue-FEP configuration: {config.source}")
    print(f"Project directory: {config.project_dir}")
    print(
        f"Replicates: {config.start_replicate}-{config.start_replicate + config.replicates - 1}"
    )
    print(f"Lambda windows: {config.lambda_windows}")
    print(f"System type: {config.membrane.system_type}")
    print(f"Total bound + apo FEP tasks: {config.total_fep_tasks}")
    return 0


def _stage(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs(require_worker=False)
    generated = stage(config)
    print(f"Staged residue-FEP project: {config.project_dir}")
    print(f"Generated shell configuration: {generated}")
    return 0


def _prepare_ligand(args: argparse.Namespace) -> int:
    return run_backend(_load(args), "para_ligand.sh")


def _prepare_system(args: argparse.Namespace) -> int:
    config = _load(args)
    if config.membrane.enabled:
        config.validate_inputs()
        stage(config)
        print(
            "Membrane system is already prepared; staged coordinates, topology, "
            "and membrane pressure MDP without running the soluble setup script."
        )
        return 0
    return run_backend(config, "setup.sh")


def _submit(args: argparse.Namespace) -> int:
    return run_backend(_load(args), "run_pipeline.sh", require_worker=True)


def add_resfep_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "resfep", help="Residue-based alchemical FEP with GROMACS"
    )
    commands = parser.add_subparsers(dest="resfep_command", required=True)

    validate = commands.add_parser(
        "validate", help="Validate TOML configuration and input paths"
    )
    validate.add_argument("config", type=Path)
    validate.add_argument(
        "--for-submission",
        action="store_true",
        help="also require the FEP worker script",
    )
    validate.set_defaults(func=_validate)

    stage_parser = commands.add_parser(
        "stage", help="Create the project manifest and packaged MDP files"
    )
    stage_parser.add_argument("config", type=Path)
    stage_parser.set_defaults(func=_stage)

    ligand = commands.add_parser(
        "prepare-ligand", help="Parameterize and validate the ligand"
    )
    ligand.add_argument("config", type=Path)
    ligand.set_defaults(func=_prepare_ligand)

    system = commands.add_parser("prepare-system", help="Prepare apo and bound systems")
    system.add_argument("config", type=Path)
    system.set_defaults(func=_prepare_system)

    submit = commands.add_parser("submit", help="Submit the Slurm FEP array")
    submit.add_argument("config", type=Path)
    submit.set_defaults(func=_submit)
    return parser
