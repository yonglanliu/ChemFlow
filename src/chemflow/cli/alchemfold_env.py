"""``chemflow alchemfold-env`` — create the AlchemFold co-folding/docking env.

AlchemFold's structure-prediction engines (Boltz-2, Chai-1) run in their own conda
env (they pull a heavy Torch stack). This builds that env and links chemflow into
it so the desktop can drive ``alchemfold dock`` / co-folding through it:

    chemflow alchemfold-env status                 # env present? boltz/chai importable?
    chemflow alchemfold-env install                # create 'alchemfold' + boltz + chai + chemflow
    chemflow alchemfold-env install --cuda 12.4    # GPU Torch build (HPC/GPU)
    chemflow alchemfold-env install --dry-run      # print the commands only

Boltz-2 (`boltz`) and Chai-1 (`chai_lab`) are installed via pip; both bring Torch.
For GPU set --cuda to pull a CUDA Torch build.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_DEFAULT_ENV = "alchemfold"
_PACKAGES = ("boltz", "chai_lab")
# Boltz-2's optimized triangle kernels (cuequivariance/trifast) are intentionally
# NOT installed here: the current cuequivariance-ops wheels pull a very new PyTorch
# (2.11+/CUDA 13) that breaks chai_lab's torch<2.7 pin. AlchemFold defaults Boltz to
# `no_kernels: true` (pure-PyTorch). For max Boltz speed use a separate boltz-only
# env with `pip install "boltz[cuda]"` and set `no_kernels: false` there.
# Conda deps that must match env/environment-alchemfold.yaml so the CLI/desktop
# one-click build the SAME env as `conda env create -f env/environment-alchemfold.yaml`.
_CONDA_EXTRAS = ("pyyaml", "rdkit>=2024.03", "setuptools>=80", "wheel")
# GNU C++ runtime for enterprise-Linux clusters (Linux-only on conda-forge).
_LINUX_RUNTIME = ("libgcc-ng>=12", "libstdcxx-ng>=12")


def _conda_tool(prefer_mamba: bool = True) -> str | None:
    order = ("mamba", "conda") if prefer_mamba else ("conda", "mamba")
    for tool in order:
        if shutil.which(tool):
            return tool
    return None


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    fallback: Path | None = None
    for parent in here.parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            fallback = fallback or parent
            try:
                if 'name = "chemflow"' in pyproject.read_text(encoding="utf-8"):
                    return parent
            except OSError:
                pass
    return fallback or here.parents[3]


def install_commands(env: str, repo: str, *, tool: str, channel: str = "conda-forge",
                     python_version: str = "3.11", packages: tuple[str, ...] = _PACKAGES,
                     conda_extras: tuple[str, ...] = _CONDA_EXTRAS,
                     torch_index: str | None = None, exists: bool = False) -> list[list[str]]:
    """Commands that build + link the AlchemFold env (pure, for testing).

    1. create the base env (Python + conda deps that mirror
       env/environment-alchemfold.yaml) — or reuse an existing one (``exists``);
    2. pip-install Boltz-2 + Chai-1 (optionally from a CUDA Torch index URL);
    3. pip-install chemflow editable (``--no-deps --no-build-isolation``).
    """
    commands: list[list[str]] = []
    if not exists:
        commands.append([tool, "create", "-y", "-n", env, "-c", channel,
                         f"python={python_version}", *conda_extras])
    elif conda_extras:
        # updating an existing env: make sure the conda deps are present too
        commands.append([tool, "install", "-y", "-n", env, "-c", channel, *conda_extras])
    pip_pkgs = [tool, "run", "-n", env, "pip", "install", *packages]
    if torch_index:
        pip_pkgs += ["--extra-index-url", torch_index]
    commands.append(pip_pkgs)
    commands.append([tool, "run", "-n", env, "pip", "install", "-e", repo,
                     "--no-deps", "--no-build-isolation"])
    return commands


def _env_exists(tool: str, env: str) -> bool:
    return subprocess.run([tool, "run", "-n", env, "python", "-c", ""],
                          capture_output=True).returncode == 0


def _install(args) -> int:
    tool = args.tool or _conda_tool()
    if not tool:
        print("error: neither mamba nor conda found on PATH. On HPC, load conda first, "
              "then retry; otherwise install Miniforge.", file=sys.stderr)
        return 1
    repo = str(Path(args.repo).expanduser().resolve()) if args.repo else str(_repo_root())
    torch_index = None
    packages = list(_PACKAGES)
    if args.cuda:
        cu = "cu" + args.cuda.replace(".", "")
        torch_index = f"https://download.pytorch.org/whl/{cu}"

    preamble: list[list[str]] = []
    exists = _env_exists(tool, args.env)
    if args.recreate and exists:
        preamble.append([tool, "env", "remove", "-y", "-n", args.env])
        exists = False
    elif exists:
        print(f"Environment '{args.env}' already exists — installing/updating into it "
              f"(use --recreate for a clean rebuild).")

    commands = preamble + install_commands(
        args.env, repo, tool=tool, channel=args.channel, python_version=args.python_version,
        conda_extras=_CONDA_EXTRAS + (_LINUX_RUNTIME if sys.platform == "linux" else ()),
        packages=tuple(packages),
        torch_index=torch_index, exists=exists)
    for command in commands:
        print("$ " + " ".join(command))
        if args.dry_run:
            continue
        rc = subprocess.run(command, check=False).returncode
        if rc != 0:
            print(f"error: command failed (exit {rc}): {' '.join(command)}", file=sys.stderr)
            return rc
    if not args.dry_run:
        gpu = f" (CUDA {args.cuda})" if args.cuda else " (CPU)"
        print(f"\nAlchemFold environment '{args.env}'{gpu} is ready (boltz, chai_lab). "
              f"Set the boltz/chai commands via chemflow_tools.yaml or run through this env.")
    return 0


def _status(args) -> int:
    tool = args.tool or _conda_tool()
    print(f"AlchemFold environment: {args.env}")
    if not tool:
        print("  (no conda/mamba on PATH to inspect it)")
        return 0
    for mod, label in (("boltz", "Boltz-2"), ("chai_lab", "Chai-1")):
        probe = subprocess.run([tool, "run", "-n", args.env, "python", "-c", f"import {mod}"],
                               capture_output=True)
        print(f"  {label:8s}-> {'importable ✓' if probe.returncode == 0 else 'NOT importable'}")
    return 0


def add_alchemfold_env_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "alchemfold-env", help="Create/link the AlchemFold env (Boltz-2 + Chai-1)")
    commands = parser.add_subparsers(dest="alchemfold_env_command", required=True)

    install = commands.add_parser("install", help="Create the env and install Boltz-2 + Chai-1 + chemflow")
    install.add_argument("--env", default=_DEFAULT_ENV, help=f"conda env name (default {_DEFAULT_ENV})")
    install.add_argument("--channel", default="conda-forge")
    install.add_argument("--python-version", default="3.11")
    install.add_argument("--cuda", default=None,
                         help="CUDA version for a GPU Torch build, e.g. 12.4 (HPC/GPU). Omit for CPU.")
    install.add_argument("--recreate", action="store_true",
                         help="remove the env first and rebuild it cleanly")
    install.add_argument("--repo", default=None, help="repo to pip install -e (default: this repo)")
    install.add_argument("--tool", default=None, help="mamba | conda (default: mamba if present)")
    install.add_argument("--dry-run", action="store_true", help="print the commands only")
    install.set_defaults(func=_install)

    status = commands.add_parser("status", help="Show whether Boltz-2 / Chai-1 are importable")
    status.add_argument("--env", default=_DEFAULT_ENV, help=f"conda env name (default {_DEFAULT_ENV})")
    status.add_argument("--tool", default=None, help="mamba | conda")
    status.set_defaults(func=_status)
