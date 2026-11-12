"""Restore ligand bond orders lost in a PDB round-trip, from an SDF template.

``alchemforge fix-bond-orders STRUCTURE TEMPLATE.sdf --resname 4DY -o fixed.sdf``

A PDB stores connectivity (CONECT) but NOT bond order, so every ligand bond
comes back single. Geometry-based perception — Maestro's "Assign Bond Orders",
or Schrodinger's assignbondorders module — guesses, and gets CF3 groups, amides
and aromatic rings wrong often enough not to be trusted for a custom ligand.

An SDF *does* store bond orders, so if you already have one for the ligand
(the file you fed the pipeline), transfer them onto the structure's coordinates
with RDKit's ``AssignBondOrdersFromTemplate``. The template must be the same
molecule with the same connectivity and the same hydrogens; coordinates may
differ freely, which is the whole point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_templates(path):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    records = [m for m in supplier if m is not None]
    if not records:
        raise ValueError(f"no readable molecule in {path}")
    return records


def _pick_template(records, resname, explicit=None, n_atoms=None):
    """Choose the SDF record for a residue: explicit name > name match > atom count.

    Falling back to atom count matters because a structure's residue name often
    does not match the SDF's molecule name — e.g. a ligand written as ``UNK``
    whose template is named ``LIG``. Picking record[0] blindly silently pairs a
    ligand with the wrong molecule.
    """
    name = (explicit or resname).strip().upper()
    for mol in records:
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip().upper() == name:
            return mol
    if n_atoms is not None:
        sized = [m for m in records if m.GetNumAtoms() == n_atoms]
        if len(sized) == 1:
            print(f"  (no template named '{name}'; matched '"
                  f"{sized[0].GetProp('_Name').strip()}' by atom count {n_atoms})")
            return sized[0]
        if len(sized) > 1:
            raise ValueError(
                f"no template named '{name}' and {len(sized)} templates have "
                f"{n_atoms} atoms — disambiguate with --template-name"
            )
    return records[0]


_COVALENT_RADII_A = {  # Angstrom
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Se": 1.20, "Br": 1.20,
    "I": 1.39,
}


def _mol_from_atoms(atoms):
    """Build an RDKit molecule from parsed PDB atoms, bonds perceived by distance.

    Needed because RDKit's PDB reader rejects hybrid-36 serials. Bond ORDERS do
    not matter here — every bond is single and the template supplies the real
    orders — only the connectivity graph has to be right so the template can be
    matched onto it.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.RWMol()
    conf_pos = []
    for _name, element, x, y, z in atoms:
        mol.AddAtom(Chem.Atom(element))
        conf_pos.append((x, y, z))

    n = len(atoms)
    for i in range(n):
        ei = atoms[i][1]
        ri = _COVALENT_RADII_A.get(ei, 0.0)
        if ri <= 0:
            continue
        xi, yi, zi = conf_pos[i]
        for j in range(i + 1, n):
            rj = _COVALENT_RADII_A.get(atoms[j][1], 0.0)
            if rj <= 0:
                continue
            xj, yj, zj = conf_pos[j]
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2
            cut = 1.3 * (ri + rj)
            if d2 < cut * cut:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    mol = mol.GetMol()
    conf = Chem.Conformer(n)
    for i, (x, y, z) in enumerate(conf_pos):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                     ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                     ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    return mol


def _apply_template(template, atoms):
    """Return the ligand with the template's bond orders and the PDB coordinates.

    Uses a GRAPH match (AssignBondOrdersFromTemplate), so the atom order in the
    SDF need not match the order in the structure — copies of the same ligand
    are often written in different orders.
    """
    from rdkit.Chem import AllChem

    if len(atoms) != template.GetNumAtoms():
        raise ValueError(
            f"{len(atoms)} atoms in the structure vs {template.GetNumAtoms()} in "
            "the template — do both have explicit hydrogens, and is this the "
            "right template?"
        )
    frag = _mol_from_atoms(atoms)
    return AllChem.AssignBondOrdersFromTemplate(template, frag)


def _parse_pdb_residues(path, wanted):
    """Collect atoms of the named residues straight from the PDB text.

    RDKit's and OpenMM's PDB readers both reject hybrid-36 serials ("A0000"),
    which is what a >99,999-atom system is written with — and we do not need a
    full parse anyway. Only the residue name, chain, residue number, atom name
    and coordinates matter here, and those columns are fixed-width regardless of
    how the serial is encoded.

    Copies are split on CONTIGUOUS BLOCKS, not on (chain, resSeq): a PDB's
    resSeq wraps at 9999 and some writers reuse chain ids, so two copies can
    share an identifier and would otherwise be merged into one oversized
    "molecule". Each ligand's atoms are contiguous in the file, so a new block
    reliably means a new copy.

    Returns a list of (resname, chain, resseq, [(atom_name, element, x, y, z)...]).
    """
    copies = []
    current_key = None
    current = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            resname = line[17:21].strip().upper() or line[17:20].strip().upper()
            chain = line[21:22].strip()
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            key = (resname, chain, resseq, icode)
            if key != current_key:
                if current is not None and current_key[0] in wanted:
                    copies.append((current_key[0], current_key[1],
                                   current_key[2], current))
                current_key, current = key, []
            if resname not in wanted:
                continue
            atom_name = line[12:16].strip()
            try:
                x, y, z = (float(line[30:38]), float(line[38:46]),
                           float(line[46:54]))
            except ValueError:
                continue
            element = line[76:78].strip().title() if len(line) >= 78 else ""
            if not element:
                element = "".join(c for c in atom_name if c.isalpha())[:1].title()
            current.append((atom_name, element, x, y, z))
    if current is not None and current_key[0] in wanted and current:
        copies.append((current_key[0], current_key[1], current_key[2], current))
    return [c for c in copies if c[3]]


def fix_bond_orders(structure_path, template_path, resnames, output,
                    template_name=None, all_copies=True):
    """Write an SDF of the named residues with bond orders from the template.

    Returns the number of molecules written.
    """
    from rdkit import Chem

    structure_path = Path(structure_path)
    if structure_path.suffix.lower() in (".cif", ".pdbx", ".mmcif"):
        raise ValueError("mmCIF is not supported here — pass the PDB.")
    records = _load_templates(template_path)
    print(f"template file: {len(records)} molecule(s) — "
          + ", ".join(m.GetProp("_Name").strip() or "(unnamed)" for m in records))

    wanted = {r.strip().upper() for r in resnames}
    found = _parse_pdb_residues(structure_path, wanted)
    if not found:
        raise ValueError(f"none of {sorted(wanted)} found in {structure_path.name}")
    print(f"structure: {len(found)} ligand cop{'y' if len(found) == 1 else 'ies'} "
          "found — " + ", ".join(
              f"{r} chain {c or '?'} resnum {n} ({len(a)} atoms)"
              for r, c, n, a in found))

    writer = Chem.SDWriter(str(output))
    written, skipped = 0, 0
    try:
        for resname in resnames:
            name = resname.strip().upper()
            copies = [c for c in found if c[0] == name]
            n_obs = len(copies[0][3]) if copies else None
            template = _pick_template(records, name, template_name, n_obs)
            tname = template.GetProp("_Name").strip() or "(unnamed)"
            if template.GetNumConformers() == 0:
                print(f"{name}: template '{tname}' has no 3D conformer — skipping",
                      file=sys.stderr)
                continue
            if not copies:
                print(f"{name}: not found in {structure_path.name}", file=sys.stderr)
                continue
            print(f"{name}: {len(copies)} copy/copies, template '{tname}' "
                  f"({template.GetNumAtoms()} atoms)")
            for _res, chain, resseq, atoms in copies:
                try:
                    fixed = _apply_template(template, atoms)
                except Exception as exc:  # noqa: BLE001 - RDKit raises broadly
                    skipped += 1
                    print(f"  chain {chain or '?'} {resseq}: SKIPPED — {exc}",
                          file=sys.stderr)
                    continue
                fixed.SetProp("_Name", f"{name}_{chain or 'X'}{resseq}")
                writer.write(fixed)
                written += 1
                print(f"  chain {chain or '?'} {resseq}: bond orders applied "
                      f"({len(atoms)} atoms)")
                if not all_copies:
                    break
    finally:
        writer.close()

    print(f"\nsummary: {len(found)} copies found, {written} written"
          + (f", {skipped} skipped (see messages above)" if skipped else ""))
    if not written:
        raise ValueError("no ligand was written — see the messages above")
    return written


def run_cli(args: argparse.Namespace) -> int:
    resnames = [r for r in str(args.resname).replace(",", " ").split() if r]
    if not resnames:
        raise ValueError("--resname needs at least one residue name")
    n = fix_bond_orders(args.structure, args.template, resnames, args.output,
                        template_name=args.template_name,
                        all_copies=not args.first_only)
    print(f"\nwrote {n} ligand(s) -> {args.output}")
    print("For Maestro:\n"
          f"    $SCHRODINGER/utilities/structconvert {args.output} "
          f"{Path(args.output).with_suffix('.mae')}")
    return 0


def add_bond_orders_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``alchemforge fix-bond-orders`` command."""
    parser = subparsers.add_parser(
        "fix-bond-orders",
        help="Restore ligand bond orders lost in a PDB round-trip (SDF template)",
        description=(
            "PDB does not store bond orders, so ligands come back with every bond "
            "single. Transfer the correct orders from an SDF template onto the "
            "structure's coordinates instead of guessing them from geometry."
        ),
    )
    parser.add_argument("structure", metavar="STRUCTURE.pdb",
                        help="PDB holding the ligand coordinates")
    parser.add_argument("template", metavar="TEMPLATE.sdf",
                        help="SDF with correct bond orders")
    parser.add_argument("--resname", required=True,
                        help="ligand residue name(s), comma-separated (e.g. 4DY,UNK)")
    parser.add_argument("--template-name",
                        help="SDF record to use (default: the record whose name "
                             "matches --resname, else the first)")
    parser.add_argument("-o", "--output", required=True, metavar="OUT.sdf",
                        help="output SDF (bond orders are preserved in SDF)")
    parser.add_argument("--first-only", action="store_true",
                        help="only the first copy (default: every copy, one per chain)")
    parser.set_defaults(func=run_cli)
    return parser
