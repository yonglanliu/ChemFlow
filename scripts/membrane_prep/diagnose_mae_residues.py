#!/usr/bin/env python
"""Diagnose missing hydrogens / split residues in a Maestro .mae.

Run with Schrodinger's python (it reads .mae directly):

    $SCHRODINGER/run python3 diagnose_mae_residues.py system.mae [RESNUM]

Reports, for the whole file:
  * protein residues that have NO hydrogens at all
  * residues whose atoms are NON-CONTIGUOUS in file order (these get split into
    two residues by any reader that groups by consecutive runs, which is how a
    residue ends up with heavy atoms but no H)
and, for RESNUM if given, every atom of that residue.
"""
import sys
from collections import defaultdict

from schrodinger import structure

AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "ACE", "NMA", "NME",
}

path = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else None

st = structure.StructureReader.read(path)
print(f"{path}: {st.atom_total} atoms")

# key -> list of (file_index, atom)
res_atoms = defaultdict(list)
for i, a in enumerate(st.atom):
    r = a.getResidue()
    key = (a.chain, int(a.resnum), (r.pdbres or "").strip())
    res_atoms[key].append((i, a))

no_h, split, hetero_no_h = [], [], []
for key, entries in res_atoms.items():
    chain, resnum, resn = key
    idx = [i for i, _ in entries]
    if max(idx) - min(idx) + 1 != len(idx):
        split.append((key, len(idx), min(idx), max(idx)))
    n_h = sum(1 for _, a in entries if (a.element or "").upper() == "H")
    if n_h == 0:
        (no_h if resn in AA else hetero_no_h).append((key, len(entries)))

print(f"\nresidues total: {len(res_atoms)}")
print(f"PROTEIN residues with NO hydrogens: {len(no_h)}")
for key, n in no_h[:20]:
    print(f"   chain {key[0]!r} resnum {key[1]} {key[2]}  ({n} atoms)")
if len(no_h) > 20:
    print(f"   ... and {len(no_h)-20} more")

print(f"\nNON-PROTEIN residues with no hydrogens: {len(hetero_no_h)}"
      "   (ions/waters-without-H are normal)")
for key, n in hetero_no_h[:10]:
    print(f"   chain {key[0]!r} resnum {key[1]} {key[2]}  ({n} atoms)")

print(f"\nresidues whose atoms are NON-CONTIGUOUS in file order: {len(split)}")
print("   (these are the ones a consecutive-run reader would split in two)")
for key, n, lo, hi in split[:20]:
    print(f"   chain {key[0]!r} resnum {key[1]} {key[2]}  {n} atoms spread over {lo}-{hi}")

if target is not None:
    print(f"\n--- every atom with resnum {target} ---")
    for key, entries in res_atoms.items():
        if str(key[1]) != str(target):
            continue
        print(f"chain {key[0]!r} {key[2]} resnum {key[1]}: {len(entries)} atoms")
        for i, a in entries:
            print(f"   idx {i:>7}  name {(a.pdbname or '').strip():<5} "
                  f"elem {a.element:<2}")
