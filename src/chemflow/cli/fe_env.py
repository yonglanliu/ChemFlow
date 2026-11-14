"""``chemflow fe-env`` — create and link the ONE compute environment.

ChemFlow's physics-based engines (OpenFE RBFE/ABFE/SepTop, OpenMM MD, and the
GROMACS/PMX RBFE pipeline) share a conda env kept separate from ChemFlow's ML
stack. This builds that single env — named ``alchemforge`` by default — with the
whole compute toolchain and links chemflow into it, so the desktop drives every
free-energy workflow through one interpreter:

    chemflow fe-env status                 # env present? openfe/gmx/obabel/acpype/pmx?
    chemflow fe-env install                # create 'alchemforge' + full stack + chemflow
    chemflow fe-env install --env myenv    # install into an existing env instead
    chemflow fe-env install --cuda 12.6    # GPU builds (HPC)
    chemflow fe-env install --staged       # smaller solves (low-memory login nodes; avoids exit -9)
    chemflow fe-env install --dry-run      # print the commands only

The package pins here mirror ``env/environment-alchemforge.yaml`` (a test keeps them in
sync), so this installer and ``mamba env create -f`` give the same environment.
Stack: OpenFE 1.12, openmm 8.4, rdkit, openbabel, acpype, ambertools, gromacs,
and PMX (installed from the bundled ``pmx/`` checkout; only needed for GROMACS
hybrid topologies). Afterwards the desktop auto-detects the env (name
alchemforge, then openfe/fep) or set $CHEMFLOW_FE_PYTHON.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_DEFAULT_ENV = "alchemforge"
_PMX_SPEC = "pmx"
# conda-forge GROMACS has OpenCL and CUDA GPU variants plus CPU builds. Plain
# `gromacs` can resolve to the OpenCL build, which cannot offload on NVIDIA nodes;
# a GPU install uses the CUDA build string instead (see _install).
_DEFAULT_GROMACS_SPEC = "gromacs"
_CUDA_GROMACS_SPEC = "gromacs=*=nompi_cuda*"
# Compute stack, pinned to the same versions as env/environment-alchemforge.yaml.
# NumPy 2.x: the validated env runs 2.4.6 against the MKL BLAS build. RDKit is
# still capped below 2026.03 for AmberTools.
# GUI-only libraries (Qt, VTK, ...) are intentionally left to the full yaml.
#
# openfe pulls gufe, openmmtools, openmmforcefields, lomap2, kartograf,
# konnektor, cinnabar, pymbar, plugcli and the openff-* stack transitively.
# Those were listed explicitly while openfe was vendored in-repo; the fork was
# reverted on 2026-11-13 (its membrane hook was unreachable, because
# ProteinMembraneComponent subclasses SolvatedPDBComponent and OpenFE therefore
# never solvates — nor enters that branch for — a membrane system).
FE_CONDA_SPECS = (
    "openfe=1.12",
    # Needed for `partial_charge_method: nagl`; an extra, not a hard dependency.
    "openff-nagl",
    "openff-nagl-models",
    "openfe-analysis",
    "openmm=8.4.0",
    "openff-toolkit",
    "openbabel",
    "acpype",
    "ambertools",
    "pdbfixer",
    "parmed",
    "mdtraj",
    "alchemlyb",
    "mpi4py",
    "pyyaml",
    "scipy",
    "numpy>=2.0,<3",
    "pandas>=2.1,<3",
    "rdkit>=2024.03,<2026.03",
    "pip",
    "setuptools>=80,<82",
    "wheel",
)
_CORE_SPECS = FE_CONDA_SPECS


def cuda_specs(cuda: str | None) -> list[str]:
    """Conda specs for a GPU build: the CUDA version pin plus the user-space toolkit
    (NVRTC is needed because OpenMM compiles CUDA kernels at runtime)."""
    return [f"cuda-version={cuda}", f"cuda-toolkit={cuda}"] if cuda else []


def _spec_name(spec: str) -> str:
    """Bare package name from a conda spec ('numpy>=1.26,<2' -> 'numpy')."""
    for sep in ("=", ">", "<", "!", "~", " "):
        spec = spec.split(sep, 1)[0]
    return spec


def staged_groups(specs: tuple[str, ...], gromacs_spec: str) -> list[list[str]]:
    """Split the compute stack into smaller sequential solves for low-memory nodes.

    A single solve of openfe+openmm+ambertools+gromacs can exhaust a login node's
    memory cgroup (mamba is OOM-killed, exit -9). Solving in stages keeps each
    resolver pass small. Stage 1 pins the scientific base (numpy/rdkit/pandas),
    stage 2 adds the alchemy core (openmm/openff/openfe), stage 3 the MD/FF tooling
    plus gromacs. The numpy/pandas/rdkit pins are carried into every later stage as
    constraints so a subsequent group can never bump them out of the OpenFE-
    compatible range. Derived from ``specs`` so it cannot drift from
    :data:`FE_CONDA_SPECS`.
    """
    base_names = {"rdkit", "numpy", "pandas", "scipy", "pip", "setuptools", "wheel", "pyyaml"}
    core_names = {"openmm", "openff-toolkit", "openfe"}
    pin_names = {"numpy", "pandas", "rdkit"}

    pins = [s for s in specs if _spec_name(s) in pin_names]
    base = [s for s in specs if _spec_name(s) in base_names]
    core = [s for s in specs if _spec_name(s) in core_names]
    rest = [s for s in specs if _spec_name(s) not in base_names | core_names]

    groups = [base, [*pins, *core], [*pins, *rest, gromacs_spec]]
    return [g for g in groups if g]


def _attach_extra_specs(groups: list[list[str]], extra_specs: tuple[str, ...]) -> list[list[str]]:
    """Add extra specs (e.g. CUDA toolkit) to the stage that pulls openmm, so GPU
    builds are selected while that group is solved; fall back to the last stage."""
    if not extra_specs:
        return groups
    for group in groups:
        if any(_spec_name(s) == "openmm" for s in group):
            group.extend(extra_specs)
            return groups
    groups[-1].extend(extra_specs)
    return groups


def _conda_tool(prefer_mamba: bool = True) -> str | None:
    order = ("mamba", "conda") if prefer_mamba else ("conda", "mamba")
    for tool in order:
        if shutil.which(tool):
            return tool
    return None


def _repo_root() -> Path:
    """Repo root whose pyproject is the chemflow distribution."""
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
                     python_version: str = "3.12", gromacs_spec: str = "gromacs",
                     extra_specs: tuple[str, ...] = (), exists: bool = False,
                     staged: bool = False) -> list[list[str]]:
    """Core commands that build + link the compute env (pure, for testing).

    Provisions the pinned stack (:data:`FE_CONDA_SPECS` + gromacs) and installs
    chemflow editable (``--no-deps --no-build-isolation``, using the env's pinned
    setuptools). PMX is separate (:func:`pmx_command`). ``exists`` → install into
    an existing env instead of creating it (idempotent retry). ``staged`` → split
    the provision into several smaller solves (:func:`staged_groups`) to avoid the
    OOM-kill (exit -9) seen when one big solve runs on a low-memory login node.
    """
    link = [tool, "run", "-n", env, "pip", "install", "-e", repo,
            "--no-deps", "--no-build-isolation"]

    if not staged:
        specs = [*FE_CONDA_SPECS, gromacs_spec]
        if exists:
            provision = [[tool, "install", "-y", "-n", env, "-c", channel, *specs, *extra_specs]]
        else:
            provision = [[tool, "create", "-y", "-n", env, "-c", channel,
                          f"python={python_version}", *specs, *extra_specs]]
        return [*provision, link]

    groups = _attach_extra_specs(staged_groups(FE_CONDA_SPECS, gromacs_spec), extra_specs)
    commands: list[list[str]] = []
    for i, group in enumerate(groups):
        if i == 0 and not exists:
            commands.append([tool, "create", "-y", "-n", env, "-c", channel,
                             f"python={python_version}", *group])
        else:
            commands.append([tool, "install", "-y", "-n", env, "-c", channel, *group])
    commands.append(link)
    return commands


def pmx_command(env: str, *, tool: str, channel: str = "conda-forge",
                spec: str = _PMX_SPEC, repo: str | None = None) -> list[list[str]]:
    """PMX install for the GROMACS hybrid-topology step.

    Prefers the bundled ``<repo>/pmx`` checkout (pip, no build isolation — PMX's
    isolated build requirements are obsolete); falls back to a conda spec."""
    if repo:
        bundled = Path(repo) / "pmx"
        if (bundled / "setup.py").is_file() or (bundled / "pyproject.toml").is_file():
            return [[tool, "run", "-n", env, "pip", "install", str(bundled),
                     "--no-deps", "--no-build-isolation"]]
    return [[tool, "install", "-y", "-n", env, "-c", channel, spec]]


def _env_exists(tool: str, env: str) -> bool:
    return subprocess.run([tool, "run", "-n", env, "python", "-c", ""],
                          capture_output=True).returncode == 0


def _install(args) -> int:
    tool = args.tool or _conda_tool()
    if not tool:
        print("error: neither mamba nor conda found on PATH. On HPC, load conda first "
              "(e.g. `module load miniconda` or `source <conda-prefix>/etc/profile.d/conda.sh`), "
              "then retry. Otherwise install Miniforge.", file=sys.stderr)
        return 1
    repo = str(Path(args.repo).expanduser().resolve()) if args.repo else str(_repo_root())

    extra_specs = list(args.with_specs or [])
    run_env = dict(os.environ)
    gromacs_spec = args.gromacs_spec
    if args.cuda:
        extra_specs.extend(cuda_specs(args.cuda))
        run_env["CONDA_OVERRIDE_CUDA"] = args.cuda
        # A GPU install must get the CUDA GROMACS variant: conda-forge otherwise
        # resolves plain `gromacs` to its OpenCL build, which fails on NVIDIA nodes
        # with "Cannot run short-ranged nonbonded interactions on a GPU". Only
        # override when the user left --gromacs-spec at its default.
        if gromacs_spec == _DEFAULT_GROMACS_SPEC:
            gromacs_spec = _CUDA_GROMACS_SPEC
            print(f"GPU build requested (--cuda {args.cuda}); selecting CUDA GROMACS "
                  f"'{gromacs_spec}'. Override with --gromacs-spec if needed.")

    preamble: list[list[str]] = []
    exists = _env_exists(tool, args.env)
    if args.recreate and exists:
        preamble.append([tool, "env", "remove", "-y", "-n", args.env])
        exists = False
    elif exists:
        print(f"Environment '{args.env}' already exists — installing/updating the "
              f"stack into it (use --recreate for a clean rebuild).")

    commands = preamble + install_commands(
        args.env, repo, tool=tool, channel=args.channel,
        python_version=args.python_version, gromacs_spec=gromacs_spec,
        extra_specs=tuple(extra_specs), exists=exists, staged=args.staged)
    for command in commands:
        print("$ " + shlex.join(command))
        if args.dry_run:
            continue
        rc = subprocess.run(command, check=False, env=run_env).returncode
        if rc != 0:
            print(f"error: command failed (exit {rc}): {shlex.join(command)}", file=sys.stderr)
            return rc

    pmx_ok = args.no_pmx
    if not args.no_pmx:
        for command in pmx_command(args.env, tool=tool, channel=args.channel,
                                   spec=getattr(args, "pmx_spec", _PMX_SPEC), repo=repo):
            print("$ " + shlex.join(command))
            if args.dry_run:
                pmx_ok = True
                continue
            pmx_ok = subprocess.run(command, check=False, env=run_env).returncode == 0

    if not args.dry_run:
        gpu = f" (CUDA {args.cuda})" if args.cuda else " (CPU/OpenCL)"
        print(f"\nCompute environment '{args.env}'{gpu} is ready "
              f"(openfe, openmm, rdkit, openbabel, acpype, ambertools, gromacs). "
              f"The desktop auto-detects it; or set CHEMFLOW_FE_PYTHON to its python.")
        if not pmx_ok:
            print("\nNOTE: PMX could not be installed. It is only needed for the GROMACS "
                  "hybrid-topology build (not charging/mapping/OpenFE). Install it later with "
                  f"`conda run -n {args.env} pip install <repo>/pmx --no-deps --no-build-isolation`.",
                  file=sys.stderr)
    return 0



def _status(args) -> int:
    tool = args.tool or _conda_tool()
    from chemflow.desktop.fe_env import find_fe_python

    python = find_fe_python()
    print(f"Free-energy interpreter: {python}")
    if python != sys.executable:
        check = subprocess.run([python, "-c", "import openfe; print(openfe.__version__)"],
                               capture_output=True, text=True)
        print(f"  openfe  -> {check.stdout.strip() if check.returncode == 0 else 'NOT importable'}")
    if tool:
        for exe in ("gmx", "obabel", "acpype"):
            probe = subprocess.run([tool, "run", "-n", args.env, "which", exe],
                                   capture_output=True, text=True)
            path = probe.stdout.strip()
            print(f"  {exe:8s}-> {path if probe.returncode == 0 and path else 'NOT FOUND'}")
        pmx = subprocess.run([tool, "run", "-n", args.env, "python", "-c", "import pmx"],
                             capture_output=True, text=True)
        print(f"  pmx     -> {'importable ✓' if pmx.returncode == 0 else 'NOT importable'}")
    return 0


def add_fe_env_parser(subparsers, name: str = "fe-env", help: str | None = None) -> None:
    parser = subparsers.add_parser(
        name, help=help or "Create/link the compute env (OpenFE + GROMACS + OpenMM) for free energy")
    commands = parser.add_subparsers(dest="fe_env_command", required=True)

    install = commands.add_parser("install", help="Create the env and install the full stack + chemflow")
    install.add_argument("--env", default=_DEFAULT_ENV,
                         help=f"conda env name (default {_DEFAULT_ENV})")
    install.add_argument("--channel", default="conda-forge")
    install.add_argument("--gromacs-spec", default=_DEFAULT_GROMACS_SPEC,
                         help="conda spec for GROMACS, e.g. 'gromacs=2024.*'. With --cuda, "
                              "defaults to the CUDA build ('gromacs=*=nompi_cuda*') unless set here.")
    install.add_argument("--python-version", default="3.12",
                         help="Python version (must satisfy chemflow's >=3.11,<3.13)")
    install.add_argument("--cuda", default=None,
                         help="CUDA version for GPU builds, e.g. 12.6 (HPC). Omit for CPU/OpenCL.")
    install.add_argument("--no-pmx", action="store_true", help="skip installing PMX")
    install.add_argument("--pmx-spec", default=_PMX_SPEC,
                         help="conda spec for PMX, used only if the repo has no bundled pmx/")
    install.add_argument("--recreate", action="store_true",
                         help="remove the env first and rebuild it cleanly")
    install.add_argument("--staged", action="store_true",
                         help="split the conda solve into smaller sequential solves "
                              "(scientific base, openmm/openfe, md-tooling+gromacs) to "
                              "avoid OOM-kills (exit -9) on low-memory login nodes")
    install.add_argument("--with", dest="with_specs", action="append", default=[],
                         metavar="SPEC", help="extra conda spec to add (repeatable)")
    install.add_argument("--repo", default=None, help="repo to pip install -e (default: this repo)")
    install.add_argument("--tool", default=None, help="mamba | conda (default: mamba if present)")
    install.add_argument("--dry-run", action="store_true", help="print the commands only")
    install.set_defaults(func=_install)

    status = commands.add_parser("status", help="Show the detected interpreter + tool availability")
    status.add_argument("--env", default=_DEFAULT_ENV, help=f"conda env name (default {_DEFAULT_ENV})")
    status.add_argument("--tool", default=None, help="mamba | conda")
    status.set_defaults(func=_status)
