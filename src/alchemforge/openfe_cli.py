"""`alchemforge openfe` CLI — drive the OpenFE workflows from their native YAML.

OpenFE rbfe/abfe/septop are configured by a single flat YAML consumed directly
by ``alchemforge.openfe.<workflow>.create_json`` (the same file the desktop
dialogs generate). This CLI is a thin, faithful wrapper around that path:

    alchemforge openfe septop validate CONFIG.yaml   # check inputs + lambda lengths
    alchemforge openfe septop generate CONFIG.yaml   # create_json -> transformation JSONs
    alchemforge openfe septop run      CONFIG.yaml   # create_json, then execute them

`rbfe` and `abfe` work the same way. `run` writes an execution config derived
from the YAML's `json_dir`/`result_dir` and calls ``alchemforge.openfe.execute``.
Use `--dry-run` to print the commands without running them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

OPENFE_WORKFLOWS = ("rbfe", "abfe", "septop")


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"OpenFE config must be a YAML mapping: {path}")
    return data


def _check_lambda_lengths(conf: dict) -> None:
    """All lambda lists within a leg must share one length (= n_replicas)."""
    for leg in ("complex", "solvent"):
        lengths = {
            key: len(value)
            for key, value in conf.items()
            if key.startswith(f"{leg}_lambda_") and isinstance(value, list)
        }
        if lengths and len(set(lengths.values())) > 1:
            raise ValueError(
                f"{leg} lambda lists have inconsistent lengths: {lengths}"
            )


def _check_rbfe(conf: dict) -> None:
    """RBFE uses per-leg window counts and a network/transformation selection."""
    for key in ("n_lambda_windows_complex_leg", "n_lambda_windows_solvent_leg"):
        if key in conf:
            value = conf[key]
            if not isinstance(value, int) or value < 2:
                raise ValueError(f"{key} must be an integer >= 2 (got {value!r})")
    transformation = conf.get("transformation", "both")
    if transformation not in ("both", "complex", "solvent"):
        raise ValueError(
            f"transformation must be both|complex|solvent (got {transformation!r})"
        )
    network = conf.get("alchemical_network", "all")
    if network != "all" and not isinstance(network, list):
        raise ValueError(
            "alchemical_network must be 'all' or a list of ligand names"
        )


def _check_abfe(conf: dict) -> None:
    """ABFE decouples one ligand: needs a name, exactly one ligand, and per-leg
    lambda lists (elec/vdw/restraints) of equal length within each leg."""
    _check_lambda_lengths(conf)
    if not conf.get("name"):
        raise ValueError("abfe requires a 'name' (transformation name)")
    sdf = Path(conf["ligand_sdf"]).read_text(encoding="utf-8", errors="replace")
    count = sdf.count("$$$$") or (1 if sdf.strip() else 0)
    if count != 1:
        raise ValueError(f"OpenFE ABFE requires exactly one ligand; found {count}")


def _validate_conf(workflow: str, conf_path: Path) -> dict:
    conf = _load_yaml(conf_path)
    for key in ("protein_pdb", "ligand_sdf"):
        value = conf.get(key)
        if not value:
            raise ValueError(f"[{key}] is required in {conf_path}")
        if not Path(value).is_file():
            raise ValueError(f"{key} not found: {value}")
    cofactor = conf.get("cofactor_sdf")
    if cofactor and not Path(cofactor).is_file():
        raise ValueError(f"cofactor_sdf not found: {cofactor}")
    if workflow == "septop":
        _check_lambda_lengths(conf)
    elif workflow == "rbfe":
        _check_rbfe(conf)
    elif workflow == "abfe":
        _check_abfe(conf)
    return conf


def _create_json_cmd(workflow: str, conf_path: Path) -> list[str]:
    module = f"alchemforge.openfe.{workflow}.create_json"
    return [sys.executable, "-m", module, "--conf_file", str(conf_path)]


def _execution_config(conf: dict, conf_path: Path) -> Path:
    import yaml

    json_dir = conf.get("json_dir")
    result_dir = conf.get("result_dir") or conf.get("output_directory")
    if not json_dir or not result_dir:
        raise ValueError(
            "run requires 'json_dir' and 'result_dir' in the config"
        )
    exec_path = conf_path.parent / f"{conf_path.stem}_execution.yaml"
    if "restart" in conf:
        resume = bool(conf["restart"])
    else:
        resume = bool(conf.get("resume", True))
    exec_path.write_text(
        yaml.safe_dump(
            {
                "json_directory": str(json_dir),
                "output_directory": str(result_dir),
                "max_parallel": int(conf.get("max_parallel", 3)),
                "openfe_command": str(conf.get("openfe_command", "openfe")),
                "resume": resume,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return exec_path


def _execute_cmd(exec_path: Path) -> list[str]:
    return [sys.executable, "-m", "alchemforge.openfe.execute", "--config", str(exec_path)]


def _run_cmd(command: list[str], *, dry_run: bool) -> int:
    if dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, check=False).returncode


def _validate(args: argparse.Namespace) -> int:
    conf_path = Path(args.config).expanduser().resolve()
    _validate_conf(args.openfe_workflow, conf_path)
    print(f"Valid OpenFE {args.openfe_workflow} configuration: {args.config}")
    return 0


def _generate(args: argparse.Namespace) -> int:
    conf_path = Path(args.config).expanduser().resolve()
    _validate_conf(args.openfe_workflow, conf_path)
    return _run_cmd(_create_json_cmd(args.openfe_workflow, conf_path), dry_run=args.dry_run)


def _run(args: argparse.Namespace) -> int:
    conf_path = Path(args.config).expanduser().resolve()
    conf = _validate_conf(args.openfe_workflow, conf_path)
    exec_path = _execution_config(conf, conf_path)
    if args.dry_run:
        print(" ".join(_create_json_cmd(args.openfe_workflow, conf_path)))
        print(" ".join(_execute_cmd(exec_path)))
        return 0
    result = _run_cmd(_create_json_cmd(args.openfe_workflow, conf_path), dry_run=False)
    if result != 0:
        return result
    return _run_cmd(_execute_cmd(exec_path), dry_run=False)


def add_openfe_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``alchemforge openfe`` command group."""
    parser = subparsers.add_parser(
        "openfe",
        help="OpenFE alchemical workflows (rbfe, abfe, septop) from a YAML config",
        description=(
            "Run OpenFE relative/absolute/separated-topology workflows from their "
            "native YAML configuration (the file create_json consumes)."
        ),
    )
    workflows = parser.add_subparsers(dest="openfe_workflow", required=True)
    for key in OPENFE_WORKFLOWS:
        workflow = workflows.add_parser(key, help=f"OpenFE {key} workflow")
        commands = workflow.add_subparsers(dest="openfe_command", required=True)

        validate = commands.add_parser(
            "validate", help="Validate inputs and lambda-schedule lengths"
        )
        validate.add_argument("config", type=Path, help="OpenFE YAML config")
        validate.set_defaults(func=_validate)

        generate = commands.add_parser(
            "generate", help="Run create_json to write transformation JSONs"
        )
        generate.add_argument("config", type=Path, help="OpenFE YAML config")
        generate.add_argument("--dry-run", action="store_true", help="Print the command only")
        generate.set_defaults(func=_generate)

        run = commands.add_parser(
            "run", help="Generate transformations then execute them"
        )
        run.add_argument("config", type=Path, help="OpenFE YAML config")
        run.add_argument("--dry-run", action="store_true", help="Print the commands only")
        run.set_defaults(func=_run)

    return parser
