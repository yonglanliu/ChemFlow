#!/usr/bin/env python
"""Wrap a membrane PDB into its periodic box for OpenFE (keeps it a .pdb).

Fixes the complex-leg step-0 NaN caused by an UNWRAPPED frame (atoms outside the
box get imaged onto their periodic partners -> overlaps -> infinite energy). Uses
OpenMM to read/write so the output PDB is in the exact convention OpenFE's
`from_pdb_file` reads (works for >99,999-atom systems too).

The membrane box is usually NOT in the PDB (no CRYST1) — OpenFE takes it from
`complex_box_vectors` in rbfe.yaml. Pass the SAME values with --box-nm.

Run in your alchemforge/chemflow env (needs openmm):

    python wrap_pdb_for_openfe.py in.pdb out.pdb --box-nm 12.246,12.417,13.913
    #                                              ^ your complex_box_vectors (nm)

Then regenerate the JSON from the wrapped PDB and resubmit:
    # rbfe.yaml:  protein_pdb: out.pdb   (complex_box_vectors unchanged)
    chemflow openfe rbfe generate rbfe.yaml
"""

import argparse
import sys

import numpy as np
from openmm import Vec3, unit
from openmm.app import PDBFile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--box-nm", help="Lx,Ly,Lz in nm (orthorhombic) = your complex_box_vectors; "
                                     "overrides any CRYST1 in the PDB.")
    args = ap.parse_args()

    pdb = PDBFile(args.inp)
    pos = np.array(pdb.getPositions().value_in_unit(unit.nanometer))

    if args.box_nm:
        L = np.array([float(x) for x in args.box_nm.split(",")], dtype=float)
    else:
        bv = pdb.topology.getPeriodicBoxVectors()
        if bv is None:
            sys.exit("No box in the PDB (no CRYST1). Pass --box-nm Lx,Ly,Lz using your "
                     "complex_box_vectors from rbfe.yaml.")
        L = np.diag(np.array(bv.value_in_unit(unit.nanometer)))

    before = ((pos < 0) | (pos > L)).any(axis=1).mean() * 100
    for res in pdb.topology.residues():          # make each residue whole, then image in
        idx = [at.index for at in res.atoms()]
        ref = pos[idx[0]]
        d = pos[idx] - ref
        d -= L * np.round(d / L)
        pos[idx] = ref + d
        pos[idx] -= np.floor(pos[idx].mean(0) / L) * L

    span = pos.max(0) - pos.min(0)
    after = ((pos < 0) | (pos > L)).any(axis=1).mean() * 100
    print(f"box (nm):  {L}")
    print(f"span (nm): {span}")
    print(f"atoms outside box:  before {before:.1f}%  ->  after {after:.2f}%  (want ~0)")
    if (span > L + 1e-3).any():
        sys.exit(f"ERROR: span still > box {L}. The PDB coords and this box disagree "
                 "(wrong --box-nm, or a molecule larger than the box). Fix before running.")

    pdb.topology.setPeriodicBoxVectors(
        (Vec3(L[0], 0, 0), Vec3(0, L[1], 0), Vec3(0, 0, L[2])) * unit.nanometer)
    with open(args.out, "w") as fh:
        PDBFile.writeFile(pdb.topology, pos * unit.nanometer, fh, keepIds=True)
    print(f"wrote {args.out}")
    print("Next: protein_pdb: <this file> in rbfe.yaml, then "
          "`chemflow openfe rbfe generate rbfe.yaml` to rebuild the JSON, then resubmit.")


if __name__ == "__main__":
    main()
