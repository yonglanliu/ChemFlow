"""``chemflow reinvent-env`` — create the REINVENT4 environment.

REINVENT4 pins torch and other heavy deps, so it must live in its own conda env
(co-installing breaks ChemFlow's ML stack). This builds a py3.11 env and installs
REINVENT4 from its source clone. The resulting ``reinvent`` executable is then
auto-detected by the linker designer (or set via the REINVENT4 executable field).

    chemflow reinvent-env status
    chemflow reinvent-env install --reinvent-src ~/REINVENT4
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _conda_tool(prefer_mamba: bool = True) -> str | None:
    for tool in (("mamba", "conda") if prefer_mamba else ("conda", "mamba")):
        if shutil.which(tool):
            return tool
    return None


def _default_reinvent_src() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "REINVENT4"
        if candidate.is_dir():
            return candidate
    return None


def install_commands(env: str, reinvent_src: str, *, tool: str,
                     channel: str = "conda-forge", python_version: str = "3.11",
                     no_build_isolation: bool = True) -> list[list[str]]:
    """Commands to build the REINVENT4 env (pure, for testing).

    1. create a py3.11 env; 2. pip install REINVENT4 from source (``--no-build-
    isolation`` works around the macOS distutils issue); 3. add scipy (upstream
    omits it).
    """
    create = [tool, "create", "-y", "-n", env, "-c", channel, f"python={python_version}"]
    install = [tool, "run", "-n", env, "pip", "install", "-e", reinvent_src]
    if no_build_isolation:
        install.append("--no-build-isolation")
    scipy = [tool, "run", "-n", env, "pip", "install", "scipy"]
    return [create, install, scipy]


def _install(args) -> int:
    tool = args.tool or _conda_tool()
    if not tool:
        print("error: neither mamba nor conda found on PATH. On HPC, load conda first "
              "(e.g. `module load miniconda` or `source <conda-prefix>/etc/profile.d/conda.sh`), "
              "then retry. Otherwise install Miniforge.", file=sys.stderr)
        return 1
    src = Path(args.reinvent_src).expanduser().resolve() if args.reinvent_src else _default_reinvent_src()
    if not src or not Path(src).is_dir():
        print("error: REINVENT4 source not found; pass --reinvent-src /path/to/REINVENT4",
              file=sys.stderr)
        return 1
    commands = install_commands(args.env, str(src), tool=tool, channel=args.channel,
                                python_version=args.python_version,
                                no_build_isolation=not args.build_isolation)
    run_env = dict(os.environ)
    # macOS/miniforge pip build of REINVENT4 needs the stdlib distutils shim.
    run_env.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
    for command in commands:
        print("$ " + " ".join(command))
        if args.dry_run:
            continue
        rc = subprocess.run(command, check=False, env=run_env).returncode
        if rc != 0:
            print(f"error: command failed (exit {rc}): {' '.join(command)}", file=sys.stderr)
            return rc
    if not args.dry_run:
        exe = Path.home()  # informational
        print(f"\nREINVENT4 environment '{args.env}' is ready. Its 'reinvent' executable is "
              f"auto-detected by the linker designer (envs/{args.env}/bin/reinvent).")
    return 0


def _status(args) -> int:
    from chemflow.generative.reinvent.locate import find_reinvent_command

    cmd = find_reinvent_command()
    print(f"REINVENT4 executable: {cmd}")
    if cmd == "reinvent":
        print("  (not found on PATH or in a reinvent env — run 'chemflow reinvent-env install')")
    return 0


def add_reinvent_env_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "reinvent-env", help="Create the REINVENT4 conda environment from source")
    commands = parser.add_subparsers(dest="reinvent_env_command", required=True)

    install = commands.add_parser("install", help="Create the env and install REINVENT4")
    install.add_argument("--env", default="reinvent", help="conda env name (default reinvent)")
    install.add_argument("--reinvent-src", default=None,
                         help="path to the REINVENT4 source clone (default: bundled REINVENT4/)")
    install.add_argument("--channel", default="conda-forge")
    install.add_argument("--python-version", default="3.11")
    install.add_argument("--build-isolation", action="store_true",
                         help="use pip build isolation (default off, for the macOS distutils fix)")
    install.add_argument("--tool", default=None, help="mamba | conda (default: mamba if present)")
    install.add_argument("--dry-run", action="store_true", help="print the commands only")
    install.set_defaults(func=_install)

    status = commands.add_parser("status", help="Show the detected REINVENT4 executable")
    status.set_defaults(func=_status)
