#!/usr/bin/env python
"""Wrap a membrane PDB into its periodic box for OpenFE (keeps it a .pdb).

Fixes the complex-leg step-0 NaN. Two failure modes are handled together:

  1. UNWRAPPED frame  -> atoms sit outside the box and get imaged onto their
     periodic partners -> nonbonded overlaps (e.g. ion-ion clashes) -> inf energy.
  2. SPLIT molecules  -> a molecule straddles a box face so part of it is imaged
     to the far side -> a bond is stretched ~one box length -> huge HarmonicBond/
     Angle forces -> inf energy. (This is what a per-RESIDUE wrap of a protein
     does: consecutive residues land on opposite faces and the peptide bond snaps
     across the box.)

So this wraps per WHOLE MOLECULE (connected component over the topology bonds):
each protein / lipid / ligand is first made whole by walking its bonds, then the
whole molecule is imaged as one unit so its centre sits inside the box. Molecules
stay intact (no stretched bonds) and nothing lands outside the box.

Uses OpenMM to read/write so the output PDB is in the exact convention OpenFE's
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
from collections import deque

import numpy as np
from openmm import Vec3, unit
from openmm.app import PDBFile


def _connected_components(n_atoms, bonds):
    """Group atoms into molecules (connected components) over the bond graph.

    Atoms with no bonds (e.g. ions, water O with implicit Hs removed) become their
    own singleton molecule — imaging a singleton is just a plain wrap.
    """
    adj = [[] for _ in range(n_atoms)]
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n_atoms, dtype=bool)
    comps = []
    order = []  # BFS visit order per component, so parents are placed before children
    for start in range(n_atoms):
        if seen[start]:
            continue
        comp = []
        bfs = []
        seen[start] = True
        q = deque([start])
        while q:
            a = q.popleft()
            comp.append(a)
            for b in adj[a]:
                if not seen[b]:
                    seen[b] = True
                    bfs.append((a, b))   # edge parent->child
                    q.append(b)
        comps.append(comp)
        order.append(bfs)
    return comps, order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--box-nm", help="Lx,Ly,Lz in nm (orthorhombic) = your complex_box_vectors; "
                                     "overrides any CRYST1 in the PDB.")
    args = ap.parse_args()

    pdb = PDBFile(args.inp)
    pos = np.array(pdb.getPositions().value_in_unit(unit.nanometer))
    n = len(pos)

    if args.box_nm:
        L = np.array([float(x) for x in args.box_nm.split(",")], dtype=float)
    else:
        bv = pdb.topology.getPeriodicBoxVectors()
        if bv is None:
            sys.exit("No box in the PDB (no CRYST1). Pass --box-nm Lx,Ly,Lz using your "
                     "complex_box_vectors from rbfe.yaml.")
        L = np.diag(np.array(bv.value_in_unit(unit.nanometer)))

    bonds = [(b[0].index, b[1].index) for b in pdb.topology.bonds()]
    before = ((pos < 0) | (pos > L)).any(axis=1).mean() * 100

    comps, order = _connected_components(n, bonds)
    n_multi = sum(1 for c in comps if len(c) > 1)
    bonded_atoms = n - sum(1 for c in comps if len(c) == 1)
    print(f"molecules: {len(comps)}  ({n_multi} multi-atom, "
          f"{bonded_atoms}/{n} atoms carry bonds)")
    if bonded_atoms < 0.3 * n:
        print("WARNING: most atoms have NO bonds in this PDB. Connectivity is probably "
              "missing (no CONECT records / non-standard residues). Whole-molecule "
              "unwrap cannot help those — add CONECT, or wrap from the .prmtop with "
              "cpptraj `autoimage` instead.", file=sys.stderr)

    # 1) make each molecule whole: walk bonds parent->child, image child next to parent.
    split_fixed = 0
    for bfs in order:
        for a, b in bfs:
            shift = L * np.round((pos[b] - pos[a]) / L)
            if np.any(shift):
                split_fixed += 1
            pos[b] -= shift
    # 2) image each whole molecule so its centroid sits inside the box.
    for comp in comps:
        idx = np.asarray(comp)
        pos[idx] -= np.floor(pos[idx].mean(0) / L) * L

    after = ((pos < 0) | (pos > L)).any(axis=1).mean() * 100
    # a molecule whose own extent exceeds the box = a bond still crossing the box
    worst = np.zeros(3)
    for comp in comps:
        idx = np.asarray(comp)
        ext = pos[idx].max(0) - pos[idx].min(0)
        worst = np.maximum(worst, ext)
    print(f"box (nm):          {L}")
    print(f"bonds re-joined:   {split_fixed}  (molecules that were split across a face)")
    print(f"largest molecule extent (nm): {worst}")
    print(f"atoms outside box: before {before:.1f}%  ->  after {after:.2f}%  (want ~0)")
    if (worst > L + 1e-3).any():
        sys.exit(f"ERROR: a molecule is still larger than the box {L} after unwrap. "
                 "Either the box is wrong (--box-nm) or connectivity is missing so a "
                 "molecule could not be made whole. Fix before running.")

    pdb.topology.setPeriodicBoxVectors(
        (Vec3(L[0], 0, 0), Vec3(0, L[1], 0), Vec3(0, 0, L[2])) * unit.nanometer)
    with open(args.out, "w") as fh:
        PDBFile.writeFile(pdb.topology, pos * unit.nanometer, fh, keepIds=True)
    print(f"wrote {args.out}")
    print("Next: protein_pdb: <this file> in rbfe.yaml, then "
          "`chemflow openfe rbfe generate rbfe.yaml` to rebuild the JSON, then resubmit.")


if __name__ == "__main__":
    main()
