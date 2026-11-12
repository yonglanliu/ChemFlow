#!/usr/bin/env python
"""Merge structure files into one system (no repair, no PDBFixer).

    python merge_structures.py OUT.cif fixed_protein.pdb ligands.pdb [more ...]

Accepts PDB / mmCIF / Maestro .mae for every input (via ChemFlow's reader) and
writes PDB or mmCIF based on OUT's extension. Nothing is repaired or renamed —
atoms, residue names and numbering are carried through as-is, which is what you
want when the inputs have already been prepared.

The periodic box is taken from the first input that has one, or from --box-nm.
Systems over 99,999 atoms must be written as .cif (PDB's serial column overflows
and OpenMM cannot read the result back).

Run inside the alchemforge environment:
    conda activate alchemforge
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="OUT.pdb or OUT.cif")
    ap.add_argument("inputs", nargs="+", help="structures to merge, in order")
    ap.add_argument("--box-nm", help="Lx,Ly,Lz in nm (overrides any box in the inputs)")
    args = ap.parse_args()

    from openmm import Vec3, unit
    from openmm.app import Modeller

    from chemflow.fe.openmm_modeling.membrane_relax import read_structure, write_pdb

    first = read_structure(args.inputs[0])
    modeller = Modeller(first.topology, first.positions)
    print(f"{args.inputs[0]}: {first.topology.getNumAtoms()} atoms")

    for path in args.inputs[1:]:
        part = read_structure(path)
        modeller.add(part.topology, part.positions)
        print(f"{path}: +{part.topology.getNumAtoms()} atoms")

    # Box: explicit flag wins, else the first input that carries one.
    box = None
    if args.box_nm:
        lengths = [float(v) for v in args.box_nm.replace(" ", ",").split(",") if v]
        if len(lengths) != 3:
            sys.exit("--box-nm needs three numbers, e.g. 12.2,12.4,13.9")
        box = (
            Vec3(lengths[0], 0, 0),
            Vec3(0, lengths[1], 0),
            Vec3(0, 0, lengths[2]),
        ) * unit.nanometer
    else:
        for path in args.inputs:
            vectors = read_structure(path).topology.getPeriodicBoxVectors()
            if vectors is not None:
                box = vectors
                print(f"box taken from {path}: {box}")
                break
    if box is not None:
        modeller.topology.setPeriodicBoxVectors(box)
    else:
        print("WARNING: no periodic box in any input and no --box-nm given.",
              file=sys.stderr)

    total = modeller.topology.getNumAtoms()
    out = Path(args.output)
    if total >= 100000 and out.suffix.lower() not in (".cif", ".pdbx", ".mmcif"):
        sys.exit(
            f"{total} atoms exceeds the 99,999-atom PDB limit — write a .cif instead "
            f"(e.g. {out.with_suffix('.cif')})."
        )

    # keep_ids: preserve the inputs' chain/residue numbering.
    write_pdb(modeller.topology, modeller.positions, out, keep_ids=True)
    print(f"\nwrote {out}: {total} atoms, "
          f"{modeller.topology.getNumResidues()} residues, "
          f"{modeller.topology.getNumChains()} chains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
