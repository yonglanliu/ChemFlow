"""Configuration layer for the OpenMM membrane builder.

Like :mod:`alchemforge.openmm_modeling.membrane_relax_config`, this module has
no OpenMM import so the CLI / desktop / tests can load, validate, and stage a
membrane-build job without a compute stack. The heavy builder (MembraneModeller)
lives in :mod:`alchemforge.openmm_modeling.membrane_build`.

Two modes:
  * ``build``    - tile + trim a lipid patch around the protein (your minimal,
                   protein-XY+padding membrane), then solvate and add ions.
  * ``existing`` - the input already contains a membrane; only (re)solvate the
                   empty space and add ions.

Units: distances in nm, ionic_strength in molar.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alchemforge.openmm_modeling.membrane_relax_config import (
    DEFAULT_FORCEFIELDS,
    SUPPORTED_PLATFORMS,
    _deep_merge,
    _load_yaml,
    _resolve,
    _to_namespace,
)

# Lipids with a bundled OpenMM patch PDB that MembraneModeller.addMembrane reads.
SUPPORTED_LIPIDS = {"POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC"}
SUPPORTED_MODES = {"build", "existing"}
SUPPORTED_WATER_MODELS = {"tip3p", "spce", "tip4pew", "tip5p"}

DEFAULTS: dict[str, Any] = {
    "mode": "build",
    "platform": "CUDA",
    "forcefields": list(DEFAULT_FORCEFIELDS),
    "output_pdb": "membrane_complex.pdb",
    "write_intermediates": False,
    "workdir": ".",
    # Optional: "ligand_sdf" + "small_molecule_forcefield" to parameterize a
    # ligand carried inside the input protein PDB.
    "solvent_model": "tip3p",
    "positive_ion": "Na+",
    "negative_ion": "Cl-",
    "ionic_strength": 0.15,   # molar
    "neutralize": True,
    # --- build mode (tile + trim a patch around the protein) ---
    "build": {
        "lipid_type": "POPC",
        "scale_factor": 0.6,         # XY shrink of the solute while packing lipids
        "membrane_center_z": 0.0,    # nm
        "x_padding": 1.0,            # nm added each side beyond the protein in X
        "y_padding": 1.0,            # nm, Y
        "z_padding": 1.0,            # nm, water above/below the bilayer
    },
    # --- existing-membrane mode (solvate empty space only) ---
    "existing": {
        "padding": 1.0,              # nm, used for the Z box if not otherwise set
        "boxsize": None,             # [Lx, Ly, Lz] nm, or null
        "box_vectors": None,         # [[Lx,0,0],[0,Ly,0],[0,0,Lz]] nm, or null
        "pbc_x": None,               # Lx (nm) for XY box from the membrane, or null
        "pbc_y": None,               # Ly (nm)
    },
}


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a membrane-build YAML, merged over the defaults; resolve paths."""
    config_path = Path(path).expanduser().resolve()
    user = _load_yaml(config_path)
    conf = _to_namespace(_deep_merge(DEFAULTS, user))

    base_dir = config_path.parent
    conf.input_pdb = _resolve(base_dir, getattr(conf, "input_pdb", None))
    if getattr(conf, "ligand_sdf", None):
        conf.ligand_sdf = _resolve(base_dir, conf.ligand_sdf)
    workdir = getattr(conf, "workdir", ".")
    if not workdir or str(workdir) == ".":
        conf.workdir = str(base_dir)
    else:
        conf.workdir = _resolve(base_dir, workdir)
    return conf


def validate_config(conf: SimpleNamespace) -> None:
    """Validate a loaded membrane-build configuration (no OpenMM needed)."""
    input_pdb = getattr(conf, "input_pdb", None)
    if not input_pdb:
        raise ValueError("Membrane-build config requires 'input_pdb'")
    pdb_path = Path(input_pdb)
    if not pdb_path.is_file() or pdb_path.suffix.lower() != ".pdb":
        raise ValueError(f"input_pdb must be an existing .pdb file: {input_pdb}")

    mode = str(getattr(conf, "mode", "build"))
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"mode must be one of {sorted(SUPPORTED_MODES)} (got {mode!r})"
        )

    forcefields = getattr(conf, "forcefields", None)
    if not forcefields or any(not str(item).strip() for item in forcefields):
        raise ValueError("Membrane-build config requires at least one force field")

    ligand_sdf = getattr(conf, "ligand_sdf", None)
    if ligand_sdf:
        sdf_path = Path(ligand_sdf)
        if not sdf_path.is_file() or sdf_path.suffix.lower() != ".sdf":
            raise ValueError(f"ligand_sdf must be an existing .sdf file: {ligand_sdf}")

    platform = str(getattr(conf, "platform", "CUDA"))
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Unsupported OpenMM platform: {platform}; choose one of "
            f"{', '.join(sorted(SUPPORTED_PLATFORMS))}"
        )

    if str(getattr(conf, "solvent_model", "tip3p")) not in SUPPORTED_WATER_MODELS:
        raise ValueError(
            "Unsupported water model: "
            f"{conf.solvent_model}; choose one of "
            f"{', '.join(sorted(SUPPORTED_WATER_MODELS))}"
        )

    if float(getattr(conf, "ionic_strength", 0.0)) < 0.0:
        raise ValueError("ionic_strength must be >= 0 molar")

    if mode == "build":
        build = getattr(conf, "build", None)
        if build is None:
            raise ValueError("build mode requires a [build] section")
        lipid = str(getattr(build, "lipid_type", "")).upper()
        if lipid not in SUPPORTED_LIPIDS:
            raise ValueError(
                f"lipid_type must be one of {sorted(SUPPORTED_LIPIDS)} (got {lipid!r})"
            )
        scale = float(getattr(build, "scale_factor", 0.6))
        if not (0.0 < scale <= 1.0):
            raise ValueError("scale_factor must be in (0, 1]")
        for axis in ("x_padding", "y_padding", "z_padding"):
            if float(getattr(build, axis, 0.0)) <= 0.0:
                raise ValueError(f"build.{axis} must be > 0 nm")


def stage(config_path: str | Path, task_directory: str | Path) -> tuple[Path, Path]:
    """Write a runner script that builds the membrane for ``config_path``."""
    config_path = Path(config_path).expanduser().resolve()
    task = Path(task_directory).expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    runner = task / "run_membrane_build.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'trap \'echo "[FAILED] line $LINENO: $BASH_COMMAND (exit $?)" >&2\' ERR\n'
        'CONFIG="${1:?Usage: $0 membrane_build.yaml}"\n'
        f"exec {shlex.quote(sys.executable)} -m alchemforge.openmm_modeling.membrane_build "
        '--config "$CONFIG"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, config_path


def default_yaml_text(
    input_pdb: str = "protein.pdb",
    *,
    ligand_sdf: str | None = None,
    platform: str = "CUDA",
    lipid_type: str = "POPC",
) -> str:
    """Return a commented example YAML for the membrane builder (build mode)."""
    ligand_block = (
        f'ligand_sdf: "{ligand_sdf}"\n'
        'small_molecule_forcefield: "gaff-2.11"   # or an openff-x.y.z SMIRNOFF line\n'
        if ligand_sdf
        else (
            '# ligand_sdf: "ligand.sdf"            # if a ligand is in the protein PDB\n'
            '# small_molecule_forcefield: "gaff-2.11"\n'
        )
    )
    forcefields = "\n".join(f'  - "{item}"' for item in DEFAULT_FORCEFIELDS)
    return (
        "# ===================================================================\n"
        "# OpenMM membrane builder (MembraneModeller).\n"
        "# mode: build    -> tile a lipid patch around the protein and TRIM it to\n"
        "#                   protein-XY + padding (a minimal membrane), then\n"
        "#                   solvate + add ions. Output feeds relax-membrane.\n"
        "# mode: existing -> input already has a membrane; only solvate + ionize.\n"
        "# REQUIRED: align the protein to the OPM frame first (membrane normal = Z,\n"
        "#   bilayer center = z = 0). Download a pre-oriented structure from\n"
        "#   https://opm.phar.umich.edu/ or orient it with PPM.\n"
        "# Distances in nm, ionic_strength in molar.\n"
        "# ===================================================================\n"
        "\n"
        f'input_pdb: "{input_pdb}"      # OPM-oriented protein (optionally with a ligand)\n'
        f"{ligand_block}"
        'mode: "build"                   # build | existing\n'
        f'platform: "{platform}"           # CUDA | OpenCL | CPU\n'
        f'output_pdb: "membrane_complex.pdb"\n'
        "write_intermediates: false      # dump addMembrane_*.pdb checkpoints\n"
        "\n"
        "solvent_model: \"tip3p\"\n"
        'positive_ion: "Na+"\n'
        'negative_ion: "Cl-"\n'
        "ionic_strength: 0.15            # molar\n"
        "neutralize: true\n"
        "forcefields:\n"
        f"{forcefields}\n"
        "\n"
        "build:                          # used when mode == build\n"
        f'  lipid_type: "{lipid_type}"        # POPC POPE DLPC DLPE DMPC DOPC DPPC\n'
        "  scale_factor: 0.6             # XY shrink of solute while packing lipids\n"
        "  membrane_center_z: 0.0        # nm; 0 = OPM bilayer midplane (keep at 0 for OPM input)\n"
        "  x_padding: 1.0                # nm beyond protein in X (each side)\n"
        "  y_padding: 1.0                # nm, Y\n"
        "  z_padding: 1.0                # nm, water above/below the bilayer\n"
        "\n"
        "existing:                       # used when mode == existing\n"
        "  padding: 1.0                  # nm, Z box padding if not set otherwise\n"
        "  # boxsize: [8.0, 8.0, 10.0]   # nm, or leave unset\n"
        "  # pbc_x: 8.0                   # nm, XY from membrane extent\n"
        "  # pbc_y: 8.0\n"
    )
