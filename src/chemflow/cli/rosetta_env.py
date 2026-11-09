"""``chemflow rosetta-env`` — create and link the Rosetta environment.

Installs BOTH the classic Rosetta C++ app suite (rosetta_scripts, docking_protocol,
molfile_to_params …) and PyRosetta into one dedicated conda env (default name
``rosetta``), from the RosettaCommons conda channel — credential-free, no license
tarball — and links the chemflow package into it so the PROTAC commands (ternary
dock, KIC fix-structure, params) run through its interpreter:

    chemflow rosetta-env status
    chemflow rosetta-env install                  # classic Rosetta apps + PyRosetta (default)
    chemflow rosetta-env install --no-apps        # PyRosetta only (pyrosetta-installer)
    chemflow rosetta-env install --rosetta-version 2024.xx   # pin the conda rosetta build

After installing, the desktop auto-detects the env (or set $CHEMFLOW_ROSETTA_PYTHON).
Rosetta/PyRosetta are free for academic / non-commercial use.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROSETTA_CHANNEL = "https://conda.rosettacommons.org"


def _conda_tool(prefer_mamba: bool = True) -> str | None:
    for tool in (("mamba", "conda") if prefer_mamba else ("conda", "mamba")):
        if shutil.which(tool):
            return tool
    return None


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() or (parent / "setup.py").is_file():
            return parent
    return here.parents[3]


def install_commands(env: str, repo: str, *, tool: str, method: str = "pip-installer",
                     channel: str = "conda-forge", python_version: str = "3.11",
                     rosetta_channel: str = _ROSETTA_CHANNEL,
                     with_apps: bool = True, rosetta_version: str | None = None) -> list[list[str]]:
    """Commands to build + link a Rosetta/PyRosetta env (pure, for testing).

    ``with_apps`` (default): also install the classic Rosetta C++ app suite
    (``rosetta_scripts``, ``docking_protocol``, ``molfile_to_params`` …) from the
    RosettaCommons conda channel — credential-free, no license tarball — together
    with PyRosetta, in one conda env (like ``fe-env install`` for alchemforge).
    ``rosetta_version`` optionally pins the conda ``rosetta`` build (date/release
    based, e.g. ``2024.xx`` — not the ``3.15`` tarball naming).

    Without ``with_apps``: ``method='pip-installer'`` (PyRosetta via the
    ``pyrosetta-installer`` wheel helper) or ``method='conda'`` (PyRosetta from the
    RosettaCommons channel).
    """
    link = [tool, "run", "-n", env, "pip", "install", "-e", repo]
    if with_apps:
        # Classic Rosetta apps are conda-only; co-install with PyRosetta.
        rosetta_spec = f"rosetta={rosetta_version}" if rosetta_version else "rosetta"
        create = [tool, "create", "-y", "-n", env, "-c", rosetta_channel, "-c", channel,
                  rosetta_spec, "pyrosetta", f"python={python_version}"]
        return [create, link]
    if method == "conda":
        create = [tool, "create", "-y", "-n", env, "-c", rosetta_channel, "-c", channel,
                  "pyrosetta", f"python={python_version}"]
        return [create, link]
    # pip-installer path
    create = [tool, "create", "-y", "-n", env, "-c", channel, f"python={python_version}"]
    get_installer = [tool, "run", "-n", env, "pip", "install", "pyrosetta-installer"]
    run_installer = [tool, "run", "-n", env, "python", "-c",
                     "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"]
    return [create, get_installer, run_installer, link]


def _install(args) -> int:
    tool = args.tool or _conda_tool()
    if not tool:
        print("error: neither mamba nor conda found on PATH. On HPC, load conda first "
              "(e.g. `module load miniconda` or `source <conda-prefix>/etc/profile.d/conda.sh`), "
              "then retry. Otherwise install Miniforge.", file=sys.stderr)
        return 1
    repo = str(Path(args.repo).expanduser().resolve()) if args.repo else str(_repo_root())
    commands = install_commands(args.env, repo, tool=tool, method=args.method,
                                channel=args.channel, python_version=args.python_version,
                                with_apps=args.apps, rosetta_version=args.rosetta_version)
    for command in commands:
        print("$ " + " ".join(command))
        if args.dry_run:
            continue
        rc = subprocess.run(command, check=False, env=dict(os.environ)).returncode
        if rc != 0:
            print(f"error: command failed (exit {rc}): {' '.join(command)}", file=sys.stderr)
            return rc
    if not args.dry_run:
        what = ("classic Rosetta apps + PyRosetta" if args.apps else "PyRosetta")
        print(f"\nRosetta environment '{args.env}' is ready ({what}). "
              f"The desktop auto-detects it; or set CHEMFLOW_ROSETTA_PYTHON. "
              f"Run 'chemflow rosetta-env status' to verify.")
    return 0


def _status(args) -> int:
    from chemflow.desktop.rosetta_env import find_rosetta_python

    python = find_rosetta_python()
    print(f"Rosetta env interpreter: {python}")
    if python == sys.executable:
        print("  (no dedicated rosetta env detected — run 'chemflow rosetta-env install')")
        return 0
    check = subprocess.run([python, "-c", "import pyrosetta; print(pyrosetta.version())"],
                           capture_output=True, text=True)
    print("  pyrosetta importable ✓" if check.returncode == 0
          else "  pyrosetta NOT importable:\n   " + (check.stderr.strip() or "unknown error"))
    # Classic Rosetta apps: look for rosetta_scripts next to the env's python, then PATH.
    env_bin = Path(python).resolve().parent
    rs = env_bin / "rosetta_scripts"
    if not rs.exists():
        found = shutil.which("rosetta_scripts")
        rs = Path(found) if found else None
    print(f"  rosetta apps: rosetta_scripts -> {rs}" if rs
          else "  rosetta apps: rosetta_scripts NOT found "
               "(install with 'chemflow rosetta-env install --apps')")
    return 0


def add_rosetta_env_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "rosetta-env", help="Create/link the Rosetta conda env (classic apps + PyRosetta)")
    commands = parser.add_subparsers(dest="rosetta_env_command", required=True)

    install = commands.add_parser("install", help="Create the env and install the package into it")
    install.add_argument("--env", default="rosetta", help="conda env name (default rosetta)")
    install.add_argument("--apps", action=argparse.BooleanOptionalAction, default=True,
                         help="also install the classic Rosetta C++ apps (conda `rosetta` from "
                              "the RosettaCommons channel) alongside PyRosetta (default: on)")
    install.add_argument("--rosetta-version", default=None,
                         help="pin the classic rosetta conda build (default: latest; conda "
                              "versions are date/release-based, not the 3.15 tarball naming)")
    install.add_argument("--method", default="pip-installer", choices=["pip-installer", "conda"],
                         help="PyRosetta source when --no-apps: pip-installer (default) | conda")
    install.add_argument("--channel", default="conda-forge")
    install.add_argument("--python-version", default="3.11")
    install.add_argument("--repo", default=None, help="repo to pip install -e (default: this repo)")
    install.add_argument("--tool", default=None, help="mamba | conda (default: mamba if present)")
    install.add_argument("--dry-run", action="store_true", help="print the commands only")
    install.set_defaults(func=_install)

    status = commands.add_parser("status", help="Show the detected Rosetta/PyRosetta env")
    status.set_defaults(func=_status)
