"""Scaffold starter configuration + driver scripts for the FEP workflows.

    alchemforge generate rbfe      [name] [--engine gromacs|openfe]
    alchemforge generate abfe      [name] [--engine ...]
    alchemforge generate septop    [name] [--engine ...]
    alchemforge generate res_rbfe  [name] [--engine ...]

Each command writes a commented ``<name>.toml`` configuration and a matching
``<name>.sh`` driver that validates, stages, and points at the submit step for
that workflow. ``name`` defaults to the workflow key and is used as the project
name, the output file basename, and the Slurm job name.

The ``--engine`` flag selects the simulation engine (default ``gromacs``). The
``openfe`` engine is reserved for a future set of templates; requesting a
workflow it does not yet provide raises a clear error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# --------------------------------------------------------------------------- #
# TOML templates (sentinels __PROJECT__ substituted at write time)
# --------------------------------------------------------------------------- #
_RBFE_TOML = """\
# AlchemForge ligand-RBFE configuration
# Edit the paths/values, then:  alchemforge ligand-rbfe validate __PROJECT__.toml

[project]
name = "__PROJECT__"               # project dir = work_dir/name
work_dir = "work"

[system]
protein = "protein.pdb"            # receptor PDB (or use prepared_complex_* below)
force_field = "amber99sb-star-ildn-mut"
water_model = "tip3p"
gromacs_command = "gmx"
# prepared_complex_gro = "complex.gro"        # pre-built GROMACS system (skip pdb prep)
# prepared_complex_topology = "topol.top"
box_shape = "dodecahedron"
box_distance = 1.2
neutralize = true
ion_concentration = 0.15

[ligands]
library_sdf = "ligands.sdf"        # one SDF; every record needs a unique _Name
# charge_manifest = "charges.csv"            # optional per-ligand net charges

[mapping]
network_method = "minimal_spanning"   # lomap | minimal_spanning | minimal_redundant | radial
atom_mapper = "lomap"                  # lomap | kartograf
element_change = false
max3d_angstrom = 10.0
require_cycle_covering = true
# radial_hub = "ligand_name"            # required when network_method = "radial"

[topology]
backend = "pmx"
charge_method = "bcc"                   # bcc | user
charge_change_method = "reject"         # reject | correction
# charge_correction_file = "coions.csv" # required when charge_change_method = "correction"

[simulation]
replicates = 3
lambda_windows = 21
timestep_ps = 0.002
equilibration_ns = 1.0
production_ns = 5.0
sampling_method = "independent"         # independent | hrex

[execution]
mdrun_options = ["-ntmpi", "1"]
# hrex_module = "gromacs/2026.3-gcc13.2"      # only when sampling_method = "hrex"
# hrex_mpi_launcher = "mpirun"
# hrex_gromacs_command = "gmx_mpi"

[slurm]
partition = "gpu"
nodes = 1
cpus_per_task = 8
gpus = 1
memory = "16G"
walltime = "24:00:00"
# conda_init = "/path/to/conda/etc/profile.d/conda.sh"
# conda_environment = "alchemforge"
"""

_ABFE_TOML = """\
# AlchemForge ABFE (absolute binding free energy) configuration
# Edit, then:  alchemforge abfe-gromacs validate __PROJECT__.toml

[project]
name = "__PROJECT__"
work_dir = "work"

[system]
protein = "protein.pdb"
force_field = "amber99sb-ildn"
water_model = "tip3p"

[ligand]
sdf = "ligand.sdf"                 # a single ligand
resname = "LIG"
charge = 0

[topology]
backend = "pmx"
charge_method = "bcc"

[restraints]
method = "boresch"                 # Boresch orientational restraints
seed = 17

[simulation]
replicas = 2
lambda_windows = 7
lambda_function = "cosine"         # linear | cosine | power
lambda_power = 2.0
timestep_ps = 0.002
equilibration_ns = 1.0
production_ns = 5.0

[execution]
mode = "local"                     # local | slurm
max_parallel_jobs = 4
mdrun_options = ["-ntmpi", "1"]
# gpus = 1
# memory = "16G"
"""

_SEPTOP_TOML = """\
# AlchemForge SepTop (separated-topologies RBFE) configuration
# Edit, then:  alchemforge septop-gromacs validate __PROJECT__.toml

[project]
name = "__PROJECT__"
work_dir = "work"

[inputs]
ligand_library = "ligands.sdf"
prepared_root = "prepared"
septop_source_dir = "SeparatedTopologies"

# [system]
# type = "membrane"                # soluble (default) | membrane

[network]
method = "minimal_spanning"
atom_mapper = "lomap"
max3d_angstrom = 10.0
element_change = false
require_cycle_covering = true

[protocol]
replicates = 2
timestep_ps = 0.002
equilibration_ns = 1.0
production_ns = 5.0

[execution]
max_parallel_jobs = 8
mdrun_options = ["-ntmpi", "1"]

[slurm]
gpus = 1
# nodes = 1
# cpus_per_task = 4
# memory = "16G"
"""

_RESFEP_TOML = """\
# AlchemForge resFEP (residue-mutation FEP) configuration
# Edit, then:  alchemforge resfep validate __PROJECT__.toml

[project]
name = "__PROJECT__"
work_dir = "work"

[system]
complex = "complex.pdb"
force_field = "amber99sb"
water_model = "tip3p"

[mutation]
chain = "A"
residue = 42
target = "VAL"                     # three-letter target residue

[ligand]
mol2 = "ligand.mol2"
resname = "LIG"
charge = 0

[simulation]
replicates = 2
lambda_windows = 17
mode = "unified"                   # unified | split

[execution]
max_parallel_jobs = 8
mdrun_options = ["-ntmpi", "1"]
"""


# --------------------------------------------------------------------------- #
# Driver (.sh) template
# --------------------------------------------------------------------------- #
def _driver(engine: str, command: str, submit_hint: str) -> str:
    return f"""\
#!/usr/bin/env bash
# AlchemForge driver for __PROJECT__  (engine: {engine}, command: {command}).
# Generated by `alchemforge generate`. Edit the environment setup for your host.
set -euo pipefail

CONFIG="${{1:-__PROJECT__.toml}}"

# --- environment (uncomment/edit as needed) ---
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate alchemforge
# module load gromacs

echo "== validate =="
alchemforge {command} validate "$CONFIG"

echo "== stage =="
alchemforge {command} stage "$CONFIG"

echo
echo "Staging complete. To launch the simulations:"
echo "  {submit_hint}"
"""


# engine -> workflow key -> (toml text, cli command, submit hint)
_GROMACS: dict[str, tuple[str, str, str]] = {
    "rbfe": (
        _RBFE_TOML,
        "ligand-rbfe",
        "bash work/__PROJECT__/submit_production_array.sh",
    ),
    "abfe": (
        _ABFE_TOML,
        "abfe-gromacs",
        "alchemforge abfe-gromacs run __PROJECT__.toml",
    ),
    "septop": (
        _SEPTOP_TOML,
        "septop-gromacs",
        "alchemforge septop-gromacs submit __PROJECT__.toml",
    ),
    "res_rbfe": (
        _RESFEP_TOML,
        "resfep",
        "alchemforge resfep submit __PROJECT__.toml",
    ),
}

# OpenFE templates are not implemented yet; the key reserves the engine so the
# CLI surface and error messages are ready for them.
_OPENFE: dict[str, tuple[str, str, str]] = {}

_ENGINES: dict[str, dict[str, tuple[str, str, str]]] = {
    "gromacs": _GROMACS,
    "openfe": _OPENFE,
}
_ENGINE_CHOICES = tuple(_ENGINES)
_ALIASES = {"resfep": "res_rbfe", "res-fep": "res_rbfe", "abfe-gromacs": "abfe"}


def _generate(args: argparse.Namespace) -> int:
    engine = args.engine
    workflow = _ALIASES.get(args.generate_workflow, args.generate_workflow)
    table = _ENGINES.get(engine, {})
    if workflow not in table:
        available = ", ".join(sorted(table)) or "(none yet)"
        raise ValueError(
            f"The '{engine}' engine has no '{workflow}' template yet. "
            f"Available for '{engine}': {available}."
        )
    toml_text, command, submit_hint = table[workflow]
    project = args.name
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    toml_path = output_dir / f"{project}.toml"
    sh_path = output_dir / f"{project}.sh"

    for path in (toml_path, sh_path):
        if path.exists() and not args.force:
            raise ValueError(f"{path} already exists (use --force to overwrite)")

    header = f"# engine = {engine}\n"
    toml_path.write_text(
        header + toml_text.replace("__PROJECT__", project), encoding="utf-8"
    )
    sh_text = _driver(engine, command, submit_hint).replace("__PROJECT__", project)
    sh_path.write_text(sh_text, encoding="utf-8")
    sh_path.chmod(sh_path.stat().st_mode | 0o100)

    print(f"Wrote {toml_path}  (engine: {engine})")
    print(f"Wrote {sh_path}")
    print(f"Next: edit {toml_path.name}, then run  bash {sh_path.name}")
    return 0


def add_generate_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``alchemforge generate`` scaffolding command."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate a starter config.toml + driver .sh for a workflow",
        description=(
            "Scaffold a commented TOML configuration and a matching driver "
            "script for an rbfe, abfe, septop, or res_rbfe workflow. Choose the "
            "engine with --engine (default: gromacs; openfe reserved for later)."
        ),
    )
    workflows = parser.add_subparsers(dest="generate_workflow", required=True)
    helps = {
        "rbfe": "Relative binding free energy",
        "abfe": "Absolute binding free energy",
        "septop": "Separated-topologies RBFE",
        "res_rbfe": "Residue-mutation FEP",
    }
    for key in ("rbfe", "abfe", "septop", "res_rbfe"):
        step = workflows.add_parser(key, help=helps[key])
        step.add_argument(
            "name",
            nargs="?",
            default=key,
            help=f"Project name / output basename (default: {key})",
        )
        step.add_argument(
            "--engine",
            choices=_ENGINE_CHOICES,
            default="gromacs",
            help="Simulation engine for the generated config (default: gromacs)",
        )
        step.add_argument(
            "-o",
            "--output-dir",
            type=Path,
            default=Path("."),
            help="Directory to write the files into (default: current directory)",
        )
        step.add_argument(
            "--force", action="store_true", help="Overwrite existing files"
        )
        step.set_defaults(func=_generate)
    return parser
