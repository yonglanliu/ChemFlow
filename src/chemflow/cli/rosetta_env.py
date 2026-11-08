"""``chemflow rosetta-env`` — create and link the PyRosetta environment.

PyRosetta is CPU-only and best kept in its own conda env (it's large and pins
specific deps). This builds that env and installs the chemflow package into it so
the PROTAC PyRosetta commands (pyrosetta-dock, KIC fix-structure) run through its
interpreter:

    chemflow rosetta-env status
    chemflow rosetta-env install                 # pip 'pyrosetta-installer' path (default)
    chemflow rosetta-env install --method conda  # RosettaCommons conda channel (needs license)

After installing, the desktop auto-detects the env (or set $CHEMFLOW_ROSETTA_PYTHON).
PyRosetta requires a RosettaCommons license.
"""

from __future__ import annotations

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
                     rosetta_channel: str = _ROSETTA_CHANNEL) -> list[list[str]]:
    """Commands to build + link a PyRosetta env (pure, for testing).

    ``method='pip-installer'`` (default): a plain env + the ``pyrosetta-installer``
    helper fetches the wheel. ``method='conda'``: install ``pyrosetta`` from the
    RosettaCommons conda channel directly.
    """
    link = [tool, "run", "-n", env, "pip", "install", "-e", repo]
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
                                channel=args.channel, python_version=args.python_version)
    for command in commands:
        print("$ " + " ".join(command))
        if args.dry_run:
            continue
        rc = subprocess.run(command, check=False, env=dict(os.environ)).returncode
        if rc != 0:
            print(f"error: command failed (exit {rc}): {' '.join(command)}", file=sys.stderr)
            return rc
    if not args.dry_run:
        print(f"\nPyRosetta environment '{args.env}' is ready (needs a RosettaCommons license). "
              f"The desktop auto-detects it; or set CHEMFLOW_ROSETTA_PYTHON.")
    return 0


def _status(args) -> int:
    from chemflow.desktop.rosetta_env import find_rosetta_python

    python = find_rosetta_python()
    print(f"PyRosetta interpreter: {python}")
    if python == sys.executable:
        print("  (no dedicated pyrosetta env detected — run 'chemflow rosetta-env install')")
        return 0
    check = subprocess.run([python, "-c", "import pyrosetta; print(pyrosetta.version())"],
                           capture_output=True, text=True)
    print("  pyrosetta importable ✓" if check.returncode == 0
          else "  pyrosetta NOT importable:\n   " + (check.stderr.strip() or "unknown error"))
    return 0


def add_rosetta_env_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "rosetta-env", help="Create/link the PyRosetta conda environment")
    commands = parser.add_subparsers(dest="rosetta_env_command", required=True)

    install = commands.add_parser("install", help="Create the env and install the package into it")
    install.add_argument("--env", default="pyrosetta", help="conda env name (default pyrosetta)")
    install.add_argument("--method", default="pip-installer", choices=["pip-installer", "conda"],
                         help="pip-installer (default) | conda (RosettaCommons channel)")
    install.add_argument("--channel", default="conda-forge")
    install.add_argument("--python-version", default="3.11")
    install.add_argument("--repo", default=None, help="repo to pip install -e (default: this repo)")
    install.add_argument("--tool", default=None, help="mamba | conda (default: mamba if present)")
    install.add_argument("--dry-run", action="store_true", help="print the commands only")
    install.set_defaults(func=_install)

    status = commands.add_parser("status", help="Show the detected PyRosetta interpreter")
    status.set_defaults(func=_status)
