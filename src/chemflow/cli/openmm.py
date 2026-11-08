"""`chemflow openmm` CLI — OpenMM-based system modeling and MD.

OpenMM is the engine for building and equilibrating systems (soluble or
membrane) that later feed OpenFE / GROMACS free-energy calculations:

    chemflow openmm modeling generate {soluble,membrane,relax,md} ...
    chemflow openmm modeling build-soluble   {validate,stage,run} CONFIG
    chemflow openmm modeling build-membrane  {validate,stage,run} CONFIG
    chemflow openmm modeling relax-membrane  {validate,stage,run} CONFIG
    chemflow openmm modeling md              {validate,stage,run} CONFIG

The modeling steps live in ``chemflow.fe.openmm_modeling.*``. The light config
layer (validate/stage/generate) needs no OpenMM; only ``run`` imports it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_cmd(command: list[str], *, dry_run: bool) -> int:
    if dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, check=False).returncode


# ---------------------------------------------------------------------------
# Membrane relaxation (CHARMM-GUI/Desmond-style staged pre-equilibration)
# ---------------------------------------------------------------------------


def _relax_cmd(conf_path: Path) -> list[str]:
    return [sys.executable, "-m", "chemflow.fe.openmm_modeling.membrane_relax",
            "--config", str(conf_path)]


def _relax_validate(args: argparse.Namespace) -> int:
    from chemflow.fe.openmm_modeling.membrane_relax_config import (
        load_config, validate_config,
    )
    conf_path = Path(args.config).expanduser().resolve()
    validate_config(load_config(conf_path))
    print(f"Valid membrane-relax configuration: {args.config}")
    return 0


def _relax_stage(args: argparse.Namespace) -> int:
    from chemflow.fe.openmm_modeling.membrane_relax_config import (
        load_config, stage as stage_relax, validate_config,
    )
    conf_path = Path(args.config).expanduser().resolve()
    conf = load_config(conf_path)
    validate_config(conf)
    runner, _ = stage_relax(conf_path, Path(getattr(conf, "workdir", conf_path.parent)))
    print(f"Wrote membrane-relax runner: {runner}")
    print(f"Run it with: bash {runner} {conf_path}")
    return 0


def _relax_run(args: argparse.Namespace) -> int:
    conf_path = Path(args.config).expanduser().resolve()
    from chemflow.fe.openmm_modeling.membrane_relax_config import (
        load_config, validate_config,
    )
    validate_config(load_config(conf_path))
    command = _relax_cmd(conf_path)
    if args.debug_first_step:
        command.append("--debug-first-step")
    return _run_cmd(command, dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# Membrane builder (tile + trim a lipid patch, or solvate an existing membrane)
# ---------------------------------------------------------------------------


def _build_cmd(conf_path: Path) -> list[str]:
    return [sys.executable, "-m", "chemflow.fe.openmm_modeling.membrane_build",
            "--config", str(conf_path)]


def _build_validate(args: argparse.Namespace) -> int:
    from chemflow.fe.openmm_modeling.membrane_build_config import (
        load_config, validate_config,
    )
    conf_path = Path(args.config).expanduser().resolve()
    validate_config(load_config(conf_path))
    print(f"Valid membrane-build configuration: {args.config}")
    return 0


def _build_stage(args: argparse.Namespace) -> int:
    from chemflow.fe.openmm_modeling.membrane_build_config import (
        load_config, stage as stage_build, validate_config,
    )
    conf_path = Path(args.config).expanduser().resolve()
    conf = load_config(conf_path)
    validate_config(conf)
    runner, _ = stage_build(conf_path, Path(getattr(conf, "workdir", conf_path.parent)))
    print(f"Wrote membrane-build runner: {runner}")
    print(f"Run it with: bash {runner} {conf_path}")
    return 0


def _build_run(args: argparse.Namespace) -> int:
    conf_path = Path(args.config).expanduser().resolve()
    from chemflow.fe.openmm_modeling.membrane_build_config import (
        load_config, validate_config,
    )
    validate_config(load_config(conf_path))
    return _run_cmd(_build_cmd(conf_path), dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# Soluble-system builder
# ---------------------------------------------------------------------------


def _soluble_cmd(conf_path: Path) -> list[str]:
    return [sys.executable, "-m", "chemflow.fe.openmm_modeling.soluble_build",
            "--config", str(conf_path)]


def _soluble_validate(args: argparse.Namespace) -> int:
    from chemflow.fe.openmm_modeling.soluble_build_config import (
        load_config, validate_config,
    )
    conf_path = Path(args.config).expanduser().resolve()
    validate_config(load_config(conf_path))
    print(f"Valid soluble-build configuration: {args.config}")
    return 0


def _soluble_stage(args: argparse.Namespace) -> int:
    from chemflow.fe.openmm_modeling.soluble_build_config import (
        load_config, stage as stage_soluble, validate_config,
    )
    conf_path = Path(args.config).expanduser().resolve()
    conf = load_config(conf_path)
    validate_config(conf)
    runner, _ = stage_soluble(conf_path, Path(getattr(conf, "workdir", conf_path.parent)))
    print(f"Wrote soluble-build runner: {runner}")
    print(f"Run it with: bash {runner} {conf_path}")
    return 0


def _soluble_run(args: argparse.Namespace) -> int:
    conf_path = Path(args.config).expanduser().resolve()
    from chemflow.fe.openmm_modeling.soluble_build_config import (
        load_config, validate_config,
    )
    validate_config(load_config(conf_path))
    return _run_cmd(_soluble_cmd(conf_path), dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# MD runner (unrestrained, soluble or membrane)
# ---------------------------------------------------------------------------


def _md_cmd(conf_path: Path) -> list[str]:
    return [sys.executable, "-m", "chemflow.fe.openmm_modeling.md_run",
            "--config", str(conf_path)]


def _md_validate(args: argparse.Namespace) -> int:
    from chemflow.fe.openmm_modeling.md_config import load_config, validate_config
    conf_path = Path(args.config).expanduser().resolve()
    validate_config(load_config(conf_path))
    print(f"Valid MD configuration: {args.config}")
    return 0


def _md_stage(args: argparse.Namespace) -> int:
    from chemflow.fe.openmm_modeling.md_config import (
        load_config, stage as stage_md, validate_config,
    )
    conf_path = Path(args.config).expanduser().resolve()
    conf = load_config(conf_path)
    validate_config(conf)
    runner, _ = stage_md(conf_path, Path(getattr(conf, "workdir", conf_path.parent)))
    print(f"Wrote MD runner: {runner}")
    print(f"Run it with: bash {runner} {conf_path}")
    return 0


def _md_run_handler(args: argparse.Namespace) -> int:
    conf_path = Path(args.config).expanduser().resolve()
    from chemflow.fe.openmm_modeling.md_config import load_config, validate_config
    validate_config(load_config(conf_path))
    return _run_cmd(_md_cmd(conf_path), dry_run=args.dry_run)


def _add_vsr_commands(
    commands: argparse._SubParsersAction,
    *,
    label: str,
    validate_func,
    stage_func,
    run_func,
) -> None:
    """Register a standard validate/stage/run trio sharing one config argument."""
    validate = commands.add_parser("validate", help=f"Validate the {label} configuration")
    validate.add_argument("config", type=Path, help=f"{label} YAML config")
    validate.set_defaults(func=validate_func)

    stage_cmd = commands.add_parser("stage", help=f"Write a runner script for {label}")
    stage_cmd.add_argument("config", type=Path, help=f"{label} YAML config")
    stage_cmd.set_defaults(func=stage_func)

    run = commands.add_parser("run", help=f"Run {label} now (requires OpenMM)")
    run.add_argument("config", type=Path, help=f"{label} YAML config")
    run.add_argument("--dry-run", action="store_true", help="Print the command only")
    run.set_defaults(func=run_func)


def _add_relax_commands(commands: argparse._SubParsersAction) -> None:
    """Membrane relaxation shares the trio but adds --debug-first-step to run."""
    validate = commands.add_parser("validate", help="Validate the membrane-relax configuration")
    validate.add_argument("config", type=Path, help="Membrane-relax YAML config")
    validate.set_defaults(func=_relax_validate)

    stage_cmd = commands.add_parser("stage", help="Write a runner script for the relaxation")
    stage_cmd.add_argument("config", type=Path, help="Membrane-relax YAML config")
    stage_cmd.set_defaults(func=_relax_stage)

    run = commands.add_parser("run", help="Run the relaxation now (requires OpenMM)")
    run.add_argument("config", type=Path, help="Membrane-relax YAML config")
    run.add_argument("--dry-run", action="store_true", help="Print the command only")
    run.add_argument("--debug-first-step", action="store_true",
                     help="Run one diagnostic step before Stage 1")
    run.set_defaults(func=_relax_run)


# ---------------------------------------------------------------------------
# Config generator (writes a fully-commented example YAML for a modeling step)
# ---------------------------------------------------------------------------

# kind -> (config module name, default output filename). Each module exposes a
# ``default_yaml_text(...)`` returning a commented, ready-to-edit config.
_MODELING_KINDS = {
    "soluble": ("soluble_build_config", "soluble_build.yaml"),
    "membrane": ("membrane_build_config", "membrane_build.yaml"),
    "relax": ("membrane_relax_config", "membrane_relax.yaml"),
    "md": ("md_config", "md.yaml"),
}

_NEXT_STEP = {
    "soluble": "build-soluble",
    "membrane": "build-membrane",
    "relax": "relax-membrane",
    "md": "md",
}


def _modeling_generate(args: argparse.Namespace) -> int:
    import importlib

    module = importlib.import_module(
        f"chemflow.fe.openmm_modeling.{_MODELING_KINDS[args.kind][0]}"
    )

    # Pass only the keyword arguments each generator actually accepts.
    kwargs: dict = {}
    if args.input_pdb:
        kwargs["input_pdb"] = args.input_pdb
    if args.ligand:
        kwargs["ligand_sdf"] = args.ligand
    if args.kind == "membrane":
        kwargs["lipid_type"] = args.lipid
    if args.kind == "md":
        kwargs["system_type"] = args.system_type

    text = module.default_yaml_text(**kwargs)

    output = Path(args.output) if args.output else Path(_MODELING_KINDS[args.kind][1])
    output = output.expanduser()
    if output.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {output} (use --force)",
              file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote commented {args.kind} config: {output}")
    print(f"Edit it, then: chemflow openmm modeling {_NEXT_STEP[args.kind]} run {output}")
    return 0


def add_openmm_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``chemflow openmm`` command group."""
    parser = subparsers.add_parser(
        "openmm",
        help="OpenMM system modeling and MD (soluble + membrane builders, relax, MD)",
        description=(
            "OpenMM-based system modeling: solvate soluble systems, build and trim "
            "membrane systems, pre-equilibrate membranes, and run MD. Output feeds "
            "OpenFE / GROMACS free-energy calculations."
        ),
    )
    groups = parser.add_subparsers(dest="openmm_group", required=True)

    modeling = groups.add_parser(
        "modeling",
        help="Build, equilibrate, and simulate soluble or membrane systems",
    )
    steps = modeling.add_subparsers(dest="openmm_modeling", required=True)

    generate = steps.add_parser(
        "generate",
        help="Write a fully-commented example config YAML for a modeling step",
        description=(
            "Generate a ready-to-edit, commented configuration file for a modeling "
            "step (soluble/membrane builders, membrane relaxation, or MD). Every "
            "field carries an inline explanation."
        ),
    )
    generate.add_argument("kind", choices=sorted(_MODELING_KINDS),
                          help="Which modeling step to generate a config for")
    generate.add_argument("-o", "--output", type=Path,
                          help="Output YAML path (default: <kind>.yaml in the cwd)")
    generate.add_argument("--input-pdb", default=None,
                          help="Pre-fill the input_pdb field with this path")
    generate.add_argument("--ligand", default=None,
                          help="Include a ligand block pre-filled with this SDF path")
    generate.add_argument("--lipid", default="POPC",
                          help="Lipid type for the membrane builder (default: POPC)")
    generate.add_argument("--system-type", choices=["soluble", "membrane"],
                          default="soluble",
                          help="system_type for the MD config (default: soluble)")
    generate.add_argument("--force", action="store_true",
                          help="Overwrite an existing file")
    generate.set_defaults(func=_modeling_generate)

    build_soluble = steps.add_parser(
        "build-soluble",
        help="Solvate + ionize a soluble protein(+ligand) complex in a water box",
    )
    _add_vsr_commands(
        build_soluble.add_subparsers(dest="openmm_command", required=True),
        label="soluble-build",
        validate_func=_soluble_validate,
        stage_func=_soluble_stage,
        run_func=_soluble_run,
    )

    build = steps.add_parser(
        "build-membrane",
        help="Tile + trim a membrane around the protein (or solvate an existing one)",
    )
    _add_build_commands = _add_vsr_commands  # same trio shape
    _add_build_commands(
        build.add_subparsers(dest="openmm_command", required=True),
        label="membrane-build",
        validate_func=_build_validate,
        stage_func=_build_stage,
        run_func=_build_run,
    )

    relax = steps.add_parser(
        "relax-membrane",
        help="CHARMM-GUI/Desmond-style membrane pre-equilibration",
    )
    _add_relax_commands(relax.add_subparsers(dest="openmm_command", required=True))

    md = steps.add_parser(
        "md",
        help="Unrestrained MD (min/NVT/NPT/production) for soluble or membrane systems",
    )
    _add_vsr_commands(
        md.add_subparsers(dest="openmm_command", required=True),
        label="MD",
        validate_func=_md_validate,
        stage_func=_md_stage,
        run_func=_md_run_handler,
    )
    return parser
