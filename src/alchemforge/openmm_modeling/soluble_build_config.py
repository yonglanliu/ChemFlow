"""Configuration layer for the OpenMM soluble-system builder (no OpenMM import).

Solvates and ionizes a soluble protein(+ligand) complex in a water box using
``openmm.app.Modeller.addSolvent``. The heavy builder lives in
:mod:`alchemforge.openmm_modeling.soluble_build`.

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
    _deep_merge,
    _load_yaml,
    _resolve,
    _to_namespace,
)

SUPPORTED_WATER_MODELS = {"tip3p", "spce", "tip4pew", "tip5p"}
SUPPORTED_BOX_SHAPES = {"cube", "dodecahedron", "octahedron"}
SUPPORTED_BOX_MODES = {"padding", "boxsize", "vectors"}

DEFAULTS: dict[str, Any] = {
    "forcefields": list(DEFAULT_FORCEFIELDS),
    "output_pdb": "soluble_complex.pdb",
    "solvent_model": "tip3p",
    "positive_ion": "Na+",
    "negative_ion": "Cl-",
    "ionic_strength": 0.15,   # molar
    "neutralize": True,
    "workdir": ".",
    # Optional: "ligand_sdf" + "small_molecule_forcefield".
    "box": {
        "mode": "padding",        # padding | boxsize | vectors
        "padding": 1.0,           # nm (used when mode == padding)
        "box_shape": "cube",      # cube | dodecahedron | octahedron
        "boxsize": None,          # [Lx, Ly, Lz] nm (mode == boxsize)
        "box_vectors": None,      # [[..],[..],[..]] nm (mode == vectors)
    },
}


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a soluble-build YAML, merged over the defaults; resolve paths."""
    config_path = Path(path).expanduser().resolve()
    conf = _to_namespace(_deep_merge(DEFAULTS, _load_yaml(config_path)))

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
    """Validate a loaded soluble-build configuration (no OpenMM needed)."""
    input_pdb = getattr(conf, "input_pdb", None)
    if not input_pdb:
        raise ValueError("Soluble-build config requires 'input_pdb'")
    pdb_path = Path(input_pdb)
    if not pdb_path.is_file() or pdb_path.suffix.lower() != ".pdb":
        raise ValueError(f"input_pdb must be an existing .pdb file: {input_pdb}")

    forcefields = getattr(conf, "forcefields", None)
    if not forcefields or any(not str(item).strip() for item in forcefields):
        raise ValueError("Soluble-build config requires at least one force field")

    ligand_sdf = getattr(conf, "ligand_sdf", None)
    if ligand_sdf:
        sdf_path = Path(ligand_sdf)
        if not sdf_path.is_file() or sdf_path.suffix.lower() != ".sdf":
            raise ValueError(f"ligand_sdf must be an existing .sdf file: {ligand_sdf}")

    if str(getattr(conf, "solvent_model", "tip3p")) not in SUPPORTED_WATER_MODELS:
        raise ValueError(
            f"Unsupported water model: {conf.solvent_model}; choose one of "
            f"{', '.join(sorted(SUPPORTED_WATER_MODELS))}"
        )

    if float(getattr(conf, "ionic_strength", 0.0)) < 0.0:
        raise ValueError("ionic_strength must be >= 0 molar")

    box = getattr(conf, "box", None)
    if box is None:
        raise ValueError("Soluble-build config requires a [box] section")
    mode = str(getattr(box, "mode", "padding"))
    if mode not in SUPPORTED_BOX_MODES:
        raise ValueError(
            f"box.mode must be one of {sorted(SUPPORTED_BOX_MODES)} (got {mode!r})"
        )
    if mode == "padding":
        if float(getattr(box, "padding", 0.0)) <= 0.0:
            raise ValueError("box.padding must be > 0 nm")
        if str(getattr(box, "box_shape", "cube")) not in SUPPORTED_BOX_SHAPES:
            raise ValueError(
                f"box.box_shape must be one of {sorted(SUPPORTED_BOX_SHAPES)}"
            )
    elif mode == "boxsize":
        bs = getattr(box, "boxsize", None)
        if not bs or len(bs) != 3:
            raise ValueError("box.boxsize must be [Lx, Ly, Lz] nm when mode == boxsize")
    elif mode == "vectors":
        bv = getattr(box, "box_vectors", None)
        if not bv or len(bv) != 3 or any(len(row) != 3 for row in bv):
            raise ValueError(
                "box.box_vectors must be three 3-vectors when mode == vectors"
            )


def stage(config_path: str | Path, task_directory: str | Path) -> tuple[Path, Path]:
    """Write a runner script that solvates the soluble system for ``config_path``."""
    config_path = Path(config_path).expanduser().resolve()
    task = Path(task_directory).expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    runner = task / "run_soluble_build.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'trap \'echo "[FAILED] line $LINENO: $BASH_COMMAND (exit $?)" >&2\' ERR\n'
        'CONFIG="${1:?Usage: $0 soluble_build.yaml}"\n'
        f"exec {shlex.quote(sys.executable)} -m alchemforge.openmm_modeling.soluble_build "
        '--config "$CONFIG"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, config_path


def default_yaml_text(
    input_pdb: str = "complex.pdb",
    *,
    ligand_sdf: str | None = None,
) -> str:
    """Return a commented example YAML for the soluble builder."""
    ligand_block = (
        f'ligand_sdf: "{ligand_sdf}"\n'
        'small_molecule_forcefield: "gaff-2.11"   # or an openff-x.y.z SMIRNOFF line\n'
        if ligand_sdf
        else (
            '# ligand_sdf: "ligand.sdf"            # if a ligand is in the complex PDB\n'
            '# small_molecule_forcefield: "gaff-2.11"\n'
        )
    )
    forcefields = "\n".join(f'  - "{item}"' for item in DEFAULT_FORCEFIELDS)
    return (
        "# ===================================================================\n"
        "# OpenMM soluble-system builder (Modeller.addSolvent).\n"
        "# Solvate a soluble protein(+ligand) complex in a water box and add\n"
        "# ions. Output feeds the MD runner / OpenFE FEP.\n"
        "# Distances in nm, ionic_strength in molar.\n"
        "# ===================================================================\n"
        "\n"
        f'input_pdb: "{input_pdb}"      # protein (optionally with a bound ligand)\n'
        f"{ligand_block}"
        f'output_pdb: "soluble_complex.pdb"\n'
        "\n"
        'solvent_model: "tip3p"\n'
        'positive_ion: "Na+"\n'
        'negative_ion: "Cl-"\n'
        "ionic_strength: 0.15            # molar\n"
        "neutralize: true\n"
        "forcefields:\n"
        f"{forcefields}\n"
        "\n"
        "box:\n"
        '  mode: "padding"               # padding | boxsize | vectors\n'
        "  padding: 1.0                  # nm (buffer beyond the solute)\n"
        '  box_shape: "cube"             # cube | dodecahedron | octahedron\n'
        "  # boxsize: [8.0, 8.0, 8.0]    # nm, when mode == boxsize\n"
        "  # box_vectors: [[8,0,0],[0,8,0],[0,0,8]]  # nm, when mode == vectors\n"
    )
