#!/usr/bin/env python
"""Why does gufe reject this structure? Report residues with no connectivity.

    conda activate alchemforge
    python diagnose_connectivity.py complex.cif

gufe's ProteinComponent / ProteinMembraneComponent build an RDKit molecule from
the topology's BONDS and assign a formal charge per atom. Any atom with zero
connectivity is treated as a monatomic ion and looked up in a small ion table,
so an unbonded heavy atom fails with:

    ValueError: Unknown ion: <atom> in residue <RES> at index <n>

That is a CONNECTIVITY problem, not a chemistry one. Neither PDB (no CONECT)
nor mmCIF (no _struct_conn) stores bonds for non-standard residues, and OpenMM
only auto-bonds residues it has templates for.

This prints, per residue name: how many copies, how many atoms each, whether
they carry hydrogens, and how many have NO bonds — which tells you whether the
fix is "perceive the missing bonds" or "the hydrogens are genuinely absent".
"""

import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])

    from chemflow.fe.openmm_modeling.membrane_relax import read_structure

    # add_bonds=False: show the file AS IT IS, before any repair.
    structure = read_structure(path, add_bonds=False)
    top = structure.topology

    bonded = set()
    for bond in top.bonds():
        bonded.add(bond[0].index)
        bonded.add(bond[1].index)

    stats = defaultdict(lambda: {"copies": 0, "unbonded": 0, "atoms": set(),
                                 "no_h": 0})
    for res in top.residues():
        atoms = list(res.atoms())
        s = stats[res.name]
        s["copies"] += 1
        s["atoms"].add(len(atoms))
        if not any(a.index in bonded for a in atoms):
            s["unbonded"] += 1
        if not any(a.element is not None and a.element.symbol == "H"
                   for a in atoms):
            s["no_h"] += 1

    print(f"{path}: {top.getNumAtoms()} atoms, {top.getNumResidues()} residues, "
          f"{top.getNumBonds()} bonds\n")
    print(f"{'residue':<10}{'copies':>8}{'atoms':>12}{'no bonds':>10}{'no H':>8}   verdict")
    problems = 0
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["unbonded"]):
        sizes = ",".join(str(x) for x in sorted(s["atoms"])[:3])
        if s["unbonded"]:
            problems += s["unbonded"]
            single = max(s["atoms"]) == 1
            if single:
                verdict = "single atom - a real ion? gufe needs it in its ion table"
            elif s["no_h"] == s["copies"]:
                verdict = "NO HYDROGENS - add them (pdbfixer / Modeller.addHydrogens)"
            else:
                verdict = "bonds missing - perceive them (add_missing_intra_residue_bonds)"
        else:
            verdict = "ok"
        print(f"{name:<10}{s['copies']:>8}{sizes:>12}{s['unbonded']:>10}"
              f"{s['no_h']:>8}   {verdict}")

    print(f"\n{problems} residue copy/copies have no bonds at all.")
    if problems:
        print("gufe will raise 'Unknown ion: ...' on the first of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
