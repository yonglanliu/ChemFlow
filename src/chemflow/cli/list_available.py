"""``chemflow list-available`` — show the conda environments ChemFlow can build.

ChemFlow deliberately does not install everything into one environment: the
docking engines pin mutually incompatible Torch versions, and the free-energy
stack is kept free of the ML dependencies so it stays solvable on a login node.
The result is one environment per concern, each with its own installer
subcommand — and no single place that told you what those were. This is that
place.

    chemflow list-available            # table, marking which already exist
    chemflow list-available --json     # same data for scripts

The YAML column is the equivalent ``mamba env create -f env/<file>`` recipe;
the installer subcommands and the YAMLs are kept in sync by design, so either
route gives the same environment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


class Env:
    """One installable environment."""

    def __init__(self, env: str, command: str | None, yaml: str | None,
                 summary: str, note: str = ""):
        self.env = env
        self.command = command
        self.yaml = yaml
        self.summary = summary
        self.note = note

    def as_dict(self, installed: bool) -> dict:
        return {
            "env": self.env,
            "command": self.command,
            "yaml": self.yaml,
            "summary": self.summary,
            "note": self.note,
            "installed": installed,
        }


# Keep this in step with _SUBCOMMANDS in chemflow/cli/main.py. Ordered the way
# someone setting a machine up would work through them, not alphabetically.
ENVIRONMENTS: tuple[Env, ...] = (
    Env("chemflow", None, "environment.yaml",
        "Core cheminformatics + ML stack: RDKit, PyTorch, Chemprop, the desktop app",
        "No installer subcommand — this is the env chemflow itself runs in. "
        "Use env/environment-hpc.yaml on Linux/HPC."),
    Env("alchemforge", "chemflow fe-env install", "environment-alchemforge.yaml",
        "Free energy: OpenFE, OpenMM, GROMACS, PMX, AmberTools",
        "Alias: chemflow gmx-env install. Add --cuda 12.6 for GPU builds and "
        "--staged on low-memory login nodes. Use environment-alchemforge-hpc.yaml on HPC."),
    Env("boltz", "chemflow boltz-env install", "environment-boltz.yaml",
        "Boltz-2 co-folding, with the CUDA kernels"),
    Env("chai", "chemflow chai-env install", "environment-chai.yaml",
        "Chai-1 co-folding"),
    Env("diffdock", "chemflow diffdock-env install", "environment-diffdock.yaml",
        "DiffDock-L diffusion docking; also clones the DiffDock repo"),
    Env("rosetta", "chemflow rosetta-env install", "environment-rosetta.yaml",
        "Classic Rosetta C++ apps + PyRosetta (conda, no licence tarball)"),
    Env("reinvent", "chemflow reinvent-env install", "environment-reinvent.yaml",
        "REINVENT4 generative models",
        "Built from the bundled REINVENT4/ source clone."),
    Env("alchemfold", "chemflow alchemfold-env install", None,
        "DEPRECATED shared docking env",
        "Superseded by the per-engine boltz / chai / diffdock envs, which avoid "
        "their conflicting Torch pins. Kept only so old scripts keep running."),
)


def _conda_tool() -> str | None:
    for tool in ("mamba", "conda"):
        if shutil.which(tool):
            return tool
    return None


def existing_envs() -> set[str]:
    """Names of conda envs that already exist, or an empty set if unknown.

    Never raises: the listing is a convenience, and a machine without conda on
    PATH should still get the table.
    """
    tool = _conda_tool()
    if tool is None:
        return set()
    try:
        probe = subprocess.run([tool, "env", "list", "--json"],
                               capture_output=True, text=True, timeout=60)
        if probe.returncode != 0:
            return set()
        prefixes = json.loads(probe.stdout).get("envs", [])
    except (OSError, ValueError, subprocess.SubprocessError):
        return set()
    return {Path(p).name for p in prefixes}


def _repo_env_dir() -> Path | None:
    """The repo's env/ directory, when running from a source checkout."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "env"
        if (candidate / "environment.yaml").is_file():
            return candidate
    return None


def _list(args) -> int:
    present = set() if args.no_probe else existing_envs()
    env_dir = _repo_env_dir()

    if args.json:
        print(json.dumps(
            {"environments": [e.as_dict(e.env in present) for e in ENVIRONMENTS],
             "env_dir": str(env_dir) if env_dir else None},
            indent=2))
        return 0

    rows = []
    for e in ENVIRONMENTS:
        mark = "installed" if e.env in present else "-"
        rows.append((mark, e.env, e.command or "(env file only)", e.summary))

    header = ("", "ENV", "INSTALL WITH", "PROVIDES")
    widths = [max([len(header[i])] + [len(row[i]) for row in rows])
              for i in range(3)]

    print("ChemFlow uses one conda environment per concern — the docking engines "
          "pin\nincompatible Torch versions, and the free-energy stack is kept "
          "free of the ML\ndependencies so it stays solvable on a login node.\n")
    print(f"{header[0]:<{widths[0]}}  {header[1]:<{widths[1]}}  "
          f"{header[2]:<{widths[2]}}  {header[3]}")
    print("-" * (sum(widths) + 6 + max(len(row[3]) for row in rows)))
    for mark, env, command, summary in rows:
        print(f"{mark:<{widths[0]}}  {env:<{widths[1]}}  "
              f"{command:<{widths[2]}}  {summary}")

    notes = [e for e in ENVIRONMENTS if e.note]
    if notes:
        print()
        for e in notes:
            print(f"{e.env}: {e.note}")

    if env_dir:
        print(f"\nEquivalent env files in {env_dir}:")
        for e in ENVIRONMENTS:
            if e.yaml:
                exists = "" if (env_dir / e.yaml).is_file() else "   (missing)"
                print(f"  mamba env create -f env/{e.yaml}{exists}")

    if not present and not args.no_probe:
        print("\n(Could not read the conda env list, so nothing is marked "
              "installed — is conda or mamba on PATH?)")
    return 0


def add_list_available_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "list-available",
        help="List the conda environments ChemFlow can install, and which exist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the conda env lookup (faster; nothing marked installed)")
    parser.set_defaults(func=_list)
