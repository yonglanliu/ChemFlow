#!/usr/bin/env python
"""Wrap a membrane mmCIF into its periodic box for OpenFE (keeps it .cif).

Fixes the complex-leg step-0 NaN from an UNWRAPPED frame (atoms outside the box
get imaged onto their periodic partners -> overlaps -> infinite energy). Uses
OpenMM PDBxFile to read/write so it matches how OpenFE ingests the mmCIF, with no
PDB atom-serial limit.

The membrane box is usually taken from `complex_box_vectors` in rbfe.yaml, so pass
the SAME values with --box-nm (your run used 12.246,12.417,13.913).

Run in your alchemforge/chemflow env (needs openmm):

    python wrap_cif_for_openfe.py in.cif out.cif --box-nm 12.246,12.417,13.913

Then regenerate the JSON from the wrapped cif and resubmit:
    # rbfe.yaml:  protein_pdb: out.cif   (complex_box_vectors unchanged)
    chemflow openfe rbfe generate rbfe.yaml
"""

import argparse
import sys

import numpy as np
from openmm import Vec3, unit
from openmm.app import PDBxFile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--box-nm", help="Lx,Ly,Lz in nm (orthorhombic) = your complex_box_vectors; "
                                     "overrides any cell in the cif.")
    args = ap.parse_args()

    cif = PDBxFile(args.inp)
    pos = np.array(cif.getPositions().value_in_unit(unit.nanometer))

    if args.box_nm:
        L = np.array([float(x) for x in args.box_nm.split(",")], dtype=float)
    else:
        bv = cif.topology.getPeriodicBoxVectors()
        if bv is None:
            sys.exit("No cell in the cif. Pass --box-nm Lx,Ly,Lz using your "
                     "complex_box_vectors from rbfe.yaml.")
        L = np.diag(np.array(bv.value_in_unit(unit.nanometer)))

    before = ((pos < 0) | (pos > L)).any(axis=1).mean() * 100
    for res in cif.topology.residues():          # make each residue whole, then image in
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
    print(f"atoms outside box:  before {before:.1f}%  ->  after {after:.2f}%")
    # Whole molecules straddling a periodic face legitimately extend a little past the
    # box (span a bit > box is EXPECTED and fine). Only a gross overshoot means the
    # box is wrong or the coords don't belong to it.
    if (span > L * 1.4).any():
        sys.exit(f"ERROR: span {span} is >> box {L} — wrong --box-nm or mismatched coords.")
    if (span > L + 1e-3).any():
        print("note: span slightly exceeds the box — normal (whole molecules straddle the "
              "boundary); OpenMM images them correctly.")

    # Pre-flight clash check: any atom pair < 0.9 A apart under minimum image = overlap
    # (this is what caused the NaN before). Bonded atoms are ~1.0-1.5 A, so not flagged.
    clashes = _min_image_clashes(pos, L)
    print(f"minimum-image clashes (<0.9 A): {clashes}   (must be 0 for a stable start)")
    if clashes:
        print("WARNING: residual clashes remain — the frame may need a brief minimization, "
              "or a molecule is split. It will likely still NaN; inspect before a long run.")

    cif.topology.setPeriodicBoxVectors(
        (Vec3(L[0], 0, 0), Vec3(0, L[1], 0), Vec3(0, 0, L[2])) * unit.nanometer)
    with open(args.out, "w") as fh:
        PDBxFile.writeFile(cif.topology, pos * unit.nanometer, fh, keepIds=True)
    print(f"wrote {args.out}")
    print("Next: protein_pdb: <this file> in rbfe.yaml, then "
          "`chemflow openfe rbfe generate rbfe.yaml` to rebuild the JSON, then resubmit.")


def _min_image_clashes(pos, L, thr=0.09, cell=0.35):
    """Count atom pairs closer than ``thr`` nm under minimum image (cell list)."""
    from collections import defaultdict

    w = pos - np.floor(pos / L) * L
    nc = np.maximum((L / cell).astype(int), 1)
    cs = L / nc
    ci = (np.floor(w / cs).astype(int)) % nc
    grid = defaultdict(list)
    for i, (a, b, c) in enumerate(ci):
        grid[(int(a), int(b), int(c))].append(i)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    thr2 = thr * thr
    clashes = 0
    for (a, b, c), idxs in grid.items():
        neigh = []
        for dx, dy, dz in offs:
            k = ((a + dx) % nc[0], (b + dy) % nc[1], (c + dz) % nc[2])
            if k in grid:
                neigh.extend(grid[k])
        neigh = np.array(neigh)
        nb = w[neigh]
        for i in idxs:
            d = nb - w[i]
            d -= L * np.round(d / L)
            r2 = np.einsum("ij,ij->i", d, d)
            clashes += int(((r2 < thr2) & (neigh != i)).sum())
    return clashes // 2


if __name__ == "__main__":
    main()
