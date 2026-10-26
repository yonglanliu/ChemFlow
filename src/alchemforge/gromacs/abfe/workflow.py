"""Staging and execution for GROMACS ABFE projects."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from alchemforge.gromacs.lambda_schedule import generate_lambda_windows

from .config import ABFEConfig

PACKAGE_DIR = Path(__file__).resolve().parent


def _lambda_vectors(
    config: ABFEConfig, leg: str
) -> tuple[list[float], list[float], list[float]]:
    """Build sequential restraint, electrostatic, and van der Waals schedules."""
    points = generate_lambda_windows(
        config.lambda_windows, config.lambda_function, config.lambda_power
    )
    if leg == "complex":
        bonded = [min(3.0 * point, 1.0) for point in points]
        coulomb = [min(max(3.0 * point - 1.0, 0.0), 1.0) for point in points]
        vdw = [min(max(3.0 * point - 2.0, 0.0), 1.0) for point in points]
    else:
        bonded = [0.0] * config.lambda_windows
        coulomb = [min(2.0 * point, 1.0) for point in points]
        vdw = [min(max(2.0 * point - 1.0, 0.0), 1.0) for point in points]
    if not config.decouple_coulomb_first:
        coulomb, vdw = vdw, coulomb
    return bonded, coulomb, vdw


def _format_lambdas(values: list[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def _mdp_text(config: ABFEConfig, leg: str, window: int, stage_name: str) -> str:
    bonded, coulomb, vdw = _lambda_vectors(config, leg)
    minimization = stage_name == "em"
    pressure = stage_name in {"npt", "prod"}
    duration_ns = (
        config.production_ns if stage_name == "prod" else config.equilibration_ns / 2
    )
    nsteps = 50000 if minimization else round(duration_ns * 1000 / config.timestep_ps)
    lines = [
        f"integrator = {'steep' if minimization else 'md'}",
        f"nsteps = {nsteps}",
        f"dt = {config.timestep_ps}",
        "cutoff-scheme = Verlet",
        "nstlist = 20",
        "rlist = 1.0",
        "coulombtype = PME",
        "rcoulomb = 1.0",
        "vdwtype = Cut-off",
        "rvdw = 1.0",
        "DispCorr = EnerPres",
        "constraints = h-bonds",
        "constraint-algorithm = lincs",
        "free-energy = yes",
        f"init-lambda-state = {window}",
        f"bonded-lambdas = {_format_lambdas(bonded)}",
        f"coul-lambdas = {_format_lambdas(coulomb)}",
        f"vdw-lambdas = {_format_lambdas(vdw)}",
        f"couple-moltype = {config.ligand_resname}",
        "couple-lambda0 = vdw-q",
        "couple-lambda1 = none",
        "couple-intramol = no",
        f"sc-alpha = {config.softcore_alpha}",
        "sc-power = 1",
        "sc-sigma = 0.3",
        "nstdhdl = 100",
        "dhdl-print-energy = total",
        "calc-lambda-neighbors = 1",
        "nstenergy = 1000",
        "nstlog = 1000",
        "nstxout-compressed = 5000",
    ]
    if minimization:
        lines.extend(["emtol = 1000", "emstep = 0.01"])
    else:
        lines.extend(
            [
                "continuation = yes" if stage_name != "nvt" else "continuation = no",
                "tcoupl = V-rescale",
                "tc-grps = System",
                "tau-t = 1.0",
                f"ref-t = {config.temperature_k}",
                "gen-vel = yes" if stage_name == "nvt" else "gen-vel = no",
                f"gen-temp = {config.temperature_k}",
                "gen-seed = __GEN_SEED__",
            ]
        )
        if pressure:
            if config.membrane.enabled and leg == "complex":
                lines.extend(
                    [
                        "pcoupl = C-rescale"
                        if stage_name == "npt"
                        else f"pcoupl = {config.membrane.barostat}",
                        "pcoupltype = semiisotropic",
                        "tau-p = 5.0",
                        "ref-p = "
                        + " ".join(map(str, config.membrane.reference_pressure_bar)),
                        "compressibility = "
                        + " ".join(map(str, config.membrane.compressibility_bar)),
                        f"refcoord-scaling = {config.membrane.refcoord_scaling}",
                    ]
                )
            else:
                lines.extend(
                    [
                        "pcoupl = C-rescale"
                        if stage_name == "npt"
                        else "pcoupl = Parrinello-Rahman",
                        "pcoupltype = isotropic",
                        "tau-p = 5.0",
                        f"ref-p = {config.pressure_bar}",
                        "compressibility = 4.5e-5",
                    ]
                )
        else:
            lines.append("pcoupl = no")
    return "\n".join(lines) + "\n"


def _write_mdp_files(config: ABFEConfig, project: Path) -> None:
    for leg in ("complex", "solvent"):
        for window in range(config.lambda_windows):
            directory = project / "mdp" / leg / f"lambda_{window:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            for stage_name in ("em", "nvt", "npt", "prod"):
                (directory / f"{stage_name}.mdp").write_text(
                    _mdp_text(config, leg, window, stage_name), encoding="utf-8"
                )


def stage(config: ABFEConfig) -> Path:
    project = config.project_dir
    inputs = project / "input"
    internal = project / ".alchemforge"
    inputs.mkdir(parents=True, exist_ok=True)
    internal.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PACKAGE_DIR / "pmx_setup.py", internal / "pmx_setup.py")
    for source in (config.protein, config.ligand):
        if source is None:
            continue
        target = inputs / source.name
        if source.resolve() != target.resolve() and not target.exists():
            shutil.copy2(source, target)
    for leg in ("complex", "solvent"):
        for replicate in range(1, config.replicas + 1):
            (project / leg / f"rep_{replicate}").mkdir(parents=True, exist_ok=True)
    (project / "analysis").mkdir(exist_ok=True)
    (project / "logs").mkdir(exist_ok=True)
    _write_mdp_files(config, project)
    if config.worker_script is not None:
        worker = project / "ABFE_GROMACS.sh"
        if config.worker_script.resolve() != worker.resolve():
            shutil.copy2(config.worker_script, worker)
        worker.chmod(worker.stat().st_mode | 0o100)
    else:
        worker = project / "ABFE_GROMACS.sh"
        shutil.copy2(PACKAGE_DIR / "run_window.sh", worker)
        worker.chmod(worker.stat().st_mode | 0o100)
    setup = project / "setup_abfe.sh"
    shutil.copy2(PACKAGE_DIR / "setup_abfe.sh", setup)
    setup.chmod(setup.stat().st_mode | 0o100)
    launcher = project / "run_abfe.sh"
    shutil.copy2(PACKAGE_DIR / "launcher.sh", launcher)
    launcher.chmod(launcher.stat().st_mode | 0o100)
    generated = internal / "abfe_config.sh"
    generated.write_text(config.render_shell_config(), encoding="utf-8")
    return generated


def _load(args: argparse.Namespace) -> ABFEConfig:
    return ABFEConfig.load(args.config)


def _validate(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs(require_worker=args.for_execution)
    print(f"Valid GROMACS ABFE configuration: {config.source}")
    print(f"Execution mode: {config.execution_mode}")
    print(f"System type: {config.membrane.system_type}")
    print(f"Complex + solvent tasks: {config.total_tasks}")
    return 0


def _stage(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs()
    generated = stage(config)
    print(f"Staged GROMACS ABFE project: {config.project_dir}")
    print(f"Generated shell configuration: {generated}")
    return 0


def _run(args: argparse.Namespace) -> int:
    config = _load(args)
    config.validate_inputs()
    generated = stage(config)
    return subprocess.run(
        ["bash", str(config.project_dir / "run_abfe.sh"), str(generated)], check=False
    ).returncode


def _analyze(args: argparse.Namespace) -> int:
    from .analysis import analyze

    config = _load(args)
    summary = analyze(config)
    print(f"GROMACS ABFE analysis: {summary}")
    return 0


def add_abfe_gromacs_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "abfe-gromacs",
        aliases=["abfe_gromacs", "abfe"],
        help="Absolute binding free energy with GROMACS",
    )
    commands = parser.add_subparsers(dest="abfe_command", required=True)
    validate = commands.add_parser("validate", help="Validate ABFE inputs")
    validate.add_argument("config", type=Path)
    validate.add_argument("--for-execution", action="store_true")
    validate.set_defaults(func=_validate)
    stage_parser = commands.add_parser("stage", help="Stage ABFE project")
    stage_parser.add_argument("config", type=Path)
    stage_parser.set_defaults(func=_stage)
    run = commands.add_parser("run", help="Run locally or submit through Slurm")
    run.add_argument("config", type=Path)
    run.set_defaults(func=_run)
    analyze_parser = commands.add_parser(
        "analyze", help="Analyze completed ABFE windows"
    )
    analyze_parser.add_argument("config", type=Path)
    analyze_parser.set_defaults(func=_analyze)
    return parser


# Compatibility for code that imported the initial generic parser name.
add_abfe_parser = add_abfe_gromacs_parser
