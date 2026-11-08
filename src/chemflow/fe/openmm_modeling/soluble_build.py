"""Soluble-system builder for OpenMM.

Solvate + ionize a soluble protein(+ligand) complex with
``openmm.app.Modeller.addSolvent`` and write the solvated complex PDB (with a
periodic box). The output feeds the MD runner or an OpenFE FEP workflow.

Run headlessly with::

    python -m chemflow.fe.openmm_modeling.soluble_build --config soluble_build.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openmm import Vec3
from openmm.app import ForceField, Modeller, PDBFile
from openmm.unit import molar, nanometer

from chemflow.fe.openmm_modeling.soluble_build_config import load_config, validate_config


def build_from_config(config_path):
    """Load + validate a YAML config, solvate + ionize, and write the complex PDB."""
    conf = load_config(config_path)
    validate_config(conf)

    # Shared ligand template helper (heavy module).
    from chemflow.fe.openmm_modeling.membrane_relax import _register_small_molecule

    workdir = Path(conf.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    pdb = PDBFile(str(conf.input_pdb))
    modeller = Modeller(pdb.topology, pdb.positions)

    forcefield = ForceField(*list(conf.forcefields))
    if getattr(conf, "ligand_sdf", None):
        _register_small_molecule(forcefield, conf.ligand_sdf, conf)

    box = conf.box
    kwargs = {
        "model": str(conf.solvent_model),
        "positiveIon": str(conf.positive_ion),
        "negativeIon": str(conf.negative_ion),
        "ionicStrength": float(conf.ionic_strength) * molar,
        "neutralize": bool(conf.neutralize),
    }

    mode = str(getattr(box, "mode", "padding"))
    if mode == "padding":
        kwargs["padding"] = float(box.padding) * nanometer
        kwargs["boxShape"] = str(getattr(box, "box_shape", "cube"))
    elif mode == "boxsize":
        kwargs["boxSize"] = Vec3(*[float(v) for v in box.boxsize]) * nanometer
    else:  # vectors
        kwargs["boxVectors"] = (
            tuple(Vec3(*[float(v) for v in row]) for row in box.box_vectors) * nanometer
        )

    print(f"Solvating soluble system ({mode} box, {conf.solvent_model})...")
    modeller.addSolvent(forcefield, **kwargs)

    out_pdb = workdir / conf.output_pdb
    with open(out_pdb, "w") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle)
    print(f"Wrote soluble complex: {out_pdb}")
    print("Feed this into: chemflow openmm modeling md")
    return out_pdb


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Solvate + ionize a soluble complex for OpenMM.",
    )
    parser.add_argument(
        "--config", required=True, help="Soluble-build YAML configuration file"
    )
    args = parser.parse_args(argv)
    build_from_config(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
