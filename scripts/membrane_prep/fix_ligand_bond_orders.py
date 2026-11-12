#!/usr/bin/env python
"""Restore ligand bond orders lost in a PDB round-trip, using an SDF template.

    conda activate alchemforge
    python fix_ligand_bond_orders.py wrapped_last.pdb ligands-template_charged.sdf \
        --resname 4DY --out 4DY_fixed.sdf

A PDB stores connectivity (CONECT) but NOT bond order, so every ligand bond
comes back single. Geometry-based perception guesses badly for things like CF3,
amides and aromatics. The reliable fix is RDKit's AssignBondOrdersFromTemplate:
take the bond orders from a molecule you trust (your SDF) and apply them to the
atoms from the structure (which carry the current coordinates).

The template and the PDB ligand must be the SAME molecule — same atoms, same
connectivity. Coordinates may differ freely; that is the point.

Write the result as .sdf (keeps bond orders) and convert to .mae if you want it
in Maestro:
    $SCHRODINGER/utilities/structconvert 4DY_fixed.sdf 4DY_fixed.mae
"""

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("structure", help="PDB/mmCIF holding the ligand coordinates")
    ap.add_argument("template", help="SDF with CORRECT bond orders")
    ap.add_argument("--resname", required=True,
                    help="ligand residue name in the structure (e.g. 4DY, UNK, LIG)")
    ap.add_argument("--template-name", default=None,
                    help="which SDF record to use (default: the one whose name "
                         "matches --resname, else the first)")
    ap.add_argument("--out", required=True, help="output .sdf")
    ap.add_argument("--all-copies", action="store_true",
                    help="write every copy of the residue (a multimer has one "
                         "ligand per chain), not just the first")
    args = ap.parse_args()

    from rdkit import Chem
    from rdkit.Chem import AllChem

    # --- template with the bond orders we trust -----------------------------
    supplier = Chem.SDMolSupplier(args.template, removeHs=False, sanitize=True)
    records = [m for m in supplier if m is not None]
    if not records:
        sys.exit(f"no readable molecule in {args.template}")
    template = None
    if args.template_name:
        template = next((m for m in records
                         if m.GetProp("_Name").strip() == args.template_name), None)
    if template is None:
        template = next((m for m in records
                         if m.GetProp("_Name").strip() == args.resname), None)
    if template is None:
        template = records[0]
    print(f"template: '{template.GetProp('_Name').strip()}' "
          f"({template.GetNumAtoms()} atoms) from {args.template}")

    # --- ligand copies out of the structure ---------------------------------
    flavor = 0  # keep CONECT-derived bonds
    structure = (Chem.MolFromPDBFile(args.structure, removeHs=False, sanitize=False,
                                     proximityBonding=True, flavor=flavor)
                 if not args.structure.lower().endswith((".cif", ".mmcif"))
                 else None)
    if structure is None:
        sys.exit("could not read the structure (mmCIF is not supported here — "
                 "pass the PDB)")

    wanted = args.resname.strip().upper()
    groups: dict[tuple, list[int]] = {}
    for atom in structure.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetResidueName().strip().upper() != wanted:
            continue
        key = (info.GetChainId(), info.GetResidueNumber())
        groups.setdefault(key, []).append(atom.GetIdx())
    if not groups:
        sys.exit(f"no residue named {wanted} found in {args.structure}")
    print(f"found {len(groups)} copy/copies of {wanted}: "
          + ", ".join(f"chain {c or '?'} resnum {n}" for c, n in groups))

    writer = Chem.SDWriter(args.out)
    written = 0
    for (chain, resnum), idxs in groups.items():
        frag = Chem.RWMol(structure)
        keep = set(idxs)
        for i in sorted((a.GetIdx() for a in structure.GetAtoms()
                         if a.GetIdx() not in keep), reverse=True):
            frag.RemoveAtom(i)
        frag = frag.GetMol()
        if frag.GetNumAtoms() != template.GetNumAtoms():
            print(f"  chain {chain} {resnum}: SKIPPED — {frag.GetNumAtoms()} atoms "
                  f"but the template has {template.GetNumAtoms()}. Are hydrogens "
                  "present in both?", file=sys.stderr)
            continue
        try:
            fixed = AllChem.AssignBondOrdersFromTemplate(template, frag)
        except Exception as exc:  # noqa: BLE001
            print(f"  chain {chain} {resnum}: FAILED — {exc}\n"
                  "    The template must be the same molecule with the same "
                  "connectivity; check protonation and that both have explicit H.",
                  file=sys.stderr)
            continue
        fixed.SetProp("_Name", f"{wanted}_{chain or 'X'}{resnum}")
        writer.write(fixed)
        written += 1
        print(f"  chain {chain} {resnum}: bond orders restored")
        if not args.all_copies:
            break
    writer.close()

    if not written:
        sys.exit("nothing written — see the messages above")
    print(f"\nwrote {written} ligand(s) -> {args.out}")
    print("Convert for Maestro with:\n"
          f"    $SCHRODINGER/utilities/structconvert {args.out} "
          f"{args.out.rsplit('.', 1)[0]}.mae")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
