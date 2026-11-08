"""``chemflow tools`` — manage the external-tools config and check the links.

    chemflow tools init [-o chemflow_tools.yaml]   # write a template to fill in
    chemflow tools show                            # resolve every tool + check it

``show`` is a doctor: it applies the config, then reports, per tool, what resolved
and whether it actually works (can the FE python import openmm/openfe? the Rosetta
python import pyrosetta? are gmx/boltz on PATH?).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _write_template(args: argparse.Namespace) -> int:
    from chemflow.tools_config import TEMPLATE

    out = Path(args.output).expanduser()
    if out.exists() and not args.force:
        print(f"Refusing to overwrite existing {out} (use --force).")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE, encoding="utf-8")
    print(f"Wrote tools-config template: {out}")
    print("Edit it, then `chemflow tools show` to verify the links.")
    return 0


def _can_import(python: str, module: str) -> bool:
    try:
        return subprocess.run(
            [python, "-c", f"import {module}"],
            capture_output=True, timeout=120).returncode == 0
    except Exception:
        return False


def _show(args: argparse.Namespace) -> int:
    from chemflow.tools_config import apply_env, config_path, hpc_defaults, load_config
    from chemflow.desktop.env_locate import find_env_python

    path = config_path()
    print(f"Config file: {path or '(none found — using auto-detection / env vars)'}")
    config = load_config()
    applied = apply_env(config)
    if applied:
        print("Applied env vars: " + ", ".join(f"{k}={v}" for k, v in applied.items()))
    print()

    # Python-imported tools
    fe_py = find_env_python(("openfe", "alchemforge", "fep"),
                            ("CHEMFLOW_FE_PYTHON", "ALCHEMFORGE_PYTHON"))
    print(f"[FE python]      {fe_py}")
    print(f"   import openfe  -> {'OK' if _can_import(fe_py, 'openfe') else 'MISSING'}")
    print(f"   import openmm  -> {'OK' if _can_import(fe_py, 'openmm') else 'MISSING'}")
    print(f"   import chemflow-> {'OK' if _can_import(fe_py, 'chemflow') else 'MISSING'}")

    ros_py = find_env_python(("pyrosetta", "rosetta"),
                             ("CHEMFLOW_ROSETTA_PYTHON", "PYROSETTA_PYTHON"))
    print(f"[Rosetta python] {ros_py}")
    print(f"   import pyrosetta -> {'OK' if _can_import(ros_py, 'pyrosetta') else 'MISSING'}")
    print(f"   import chemflow  -> {'OK' if _can_import(ros_py, 'chemflow') else 'MISSING'}")

    # Executables
    def _exe(name: str, env_var: str | None = None) -> str:
        cand = os.environ.get(env_var) if env_var else None
        resolved = shutil.which(cand or name)
        return resolved or f"NOT FOUND ({cand or name})"

    print(f"[gromacs]        {_exe('gmx', 'CHEMFLOW_GROMACS_COMMAND')}")
    print(f"[boltz]          {_exe('boltz', 'CHEMFLOW_BOLTZ_COMMAND')}")
    print(f"[reinvent]       {_exe('reinvent', 'REINVENT_COMMAND')}")
    rosetta3 = os.environ.get("ROSETTA3") or os.environ.get("ROSETTA") or "(unset)"
    print(f"[rosetta apps]   $ROSETTA3 = {rosetta3}")

    hpc = hpc_defaults(config)
    if any(hpc.values()):
        print()
        print("HPC defaults for SLURM: "
              + ", ".join(f"{k}={v}" for k, v in hpc.items() if v))
    return 0


def add_tools_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "tools", help="Manage the external-tools config (chemflow_tools.yaml) and check links")
    commands = parser.add_subparsers(dest="tools_command", required=True)

    init = commands.add_parser("init", help="Write a tools-config template to fill in")
    init.add_argument("-o", "--output", default="chemflow_tools.yaml",
                      help="where to write the template (default: ./chemflow_tools.yaml)")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(func=_write_template)

    show = commands.add_parser("show", help="Resolve and check every external-tool link")
    show.set_defaults(func=_show)
    return parser
