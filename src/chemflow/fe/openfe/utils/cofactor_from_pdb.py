"""Build identical-charge cofactor components from ONE template SDF + a complex PDB.

OpenFE's RFE protocol rejects isomorphic ``SmallMoleculeComponent``s whose partial
charges differ. When a cofactor appears as several copies in the binding site,
charging each copy independently gives divergent charges. Instead, define the
cofactor **once** — an OpenFF-charged template SDF (with hydrogens) — and rigidly
place that single definition onto every copy's coordinates found in the complex
PDB. All copies then share identical topology + partial charges and differ only in
pose, so the validation passes with no post-hoc harmonization.

Workflow (see :func:`build_cofactors_from_pdb`):
  1. Load the template as a charged RDKit molecule (H included).
  2. For each matching residue (by ``resnames``) in the complex PDB, extract its
     heavy-atom coordinates.
  3. Graph-match the template's heavy atoms to the copy, Kabsch-fit the whole
     template onto the copy's heavy atoms, and emit a SmallMoleculeComponent.
  4. :func:`strip_residues` removes those residues from the protein PDB so the
     ProteinComponent (protein force field) doesn't double-count them.

All heavy RDKit/OpenFE imports are local so this module imports without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_PDB_ATOM_RECORDS = ("ATOM  ", "HETATM")


def _resname_set(resnames) -> set[str]:
    if isinstance(resnames, str):
        resnames = [resnames]
    return {str(r).strip().upper() for r in resnames if str(r).strip()}


def strip_residues(complex_pdb: str | Path, resnames, out_pdb: str | Path) -> Path:
    """Write ``complex_pdb`` to ``out_pdb`` with the given residue names removed.

    Also drops any ``CONECT`` records that reference a removed atom serial, so the
    protein PDB has no dangling bonds for the ProteinComponent reader.
    """
    wanted = _resname_set(resnames)
    removed_serials: set[str] = set()
    kept: list[str] = []
    # First pass: collect serials of the atoms we will remove.
    lines = Path(complex_pdb).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line[:6] in _PDB_ATOM_RECORDS and line[17:20].strip().upper() in wanted:
            removed_serials.add(line[6:11].strip())
    # Second pass: drop those atoms and any CONECT touching them.
    for line in lines:
        if line[:6] in _PDB_ATOM_RECORDS and line[17:20].strip().upper() in wanted:
            continue
        if line.startswith("CONECT"):
            serials = {line[i:i + 5].strip() for i in range(6, len(line.rstrip()), 5)}
            if serials & removed_serials:
                continue
        kept.append(line)
    out_pdb = Path(out_pdb)
    out_pdb.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return out_pdb


def extract_cofactor_copies(complex_pdb: str | Path, resnames):
    """Return ``[(key, [(atom_name, element, (x,y,z), raw_pdb_line), ...]), ...]`` per copy.

    A copy is one residue instance, keyed by (resname, chain, resseq, icode). The
    raw PDB line is kept so RDKit can do its own bond perception on the real record.
    """
    wanted = _resname_set(resnames)
    copies: dict[tuple, list] = {}
    order: list[tuple] = []
    for line in Path(complex_pdb).read_text(encoding="utf-8").splitlines():
        if line[:6] not in _PDB_ATOM_RECORDS:
            continue
        resname = line[17:20].strip().upper()
        if resname not in wanted:
            continue
        key = (resname, line[21], line[22:26].strip(), line[26])
        name = line[12:16].strip()
        element = (line[76:78].strip() or name[:1]).capitalize()
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if key not in copies:
            copies[key] = []
            order.append(key)
        copies[key].append((name, element, xyz, line))
    return [(k, copies[k]) for k in order]


def _kabsch(mobile, target):
    """Rigid transform (R, t) mapping ``mobile`` onto ``target`` (both N×3)."""
    import numpy as np

    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    mc = mobile.mean(axis=0)
    tc = target.mean(axis=0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    trans = tc - rot @ mc
    return rot, trans


def _load_template_rdkit(template_sdf, charge_settings=None):
    """Load the template SDF as a charged RDKit mol (ensure charges if missing)."""
    import openfe  # noqa: PLC0415

    smc = openfe.SmallMoleculeComponent.from_sdf_file(str(template_sdf))
    from chemflow.fe.openfe.utils.partial_charges import (  # noqa: PLC0415
        ensure_partial_charges,
        has_complete_partial_charges,
    )
    if not has_complete_partial_charges(smc) and charge_settings is not None:
        smc = ensure_partial_charges(
            [smc],
            molecule_kind="cofactor",
            method=charge_settings.partial_charge_method,
            toolkit_backend=charge_settings.off_toolkit_backend,
            generate_n_conformers=charge_settings.number_of_conformers,
            nagl_model=charge_settings.nagl_model,
            processors=4,
        )[0]
    return smc.to_rdkit()


def build_cofactors_from_pdb(template_sdf, complex_pdb, resnames, *, charge_settings=None):
    """Return identical-charge cofactor ``SmallMoleculeComponent``s.

    Chemistry + partial charges come from the single ``template_sdf``; the pose of
    each copy comes from the matching residues in ``complex_pdb``. Raises
    ValueError if a copy cannot be graph-matched to the template.
    """
    import numpy as np  # noqa: PLC0415
    import openfe  # noqa: PLC0415
    from rdkit import Chem  # noqa: PLC0415

    template = _load_template_rdkit(template_sdf, charge_settings)
    conf = template.GetConformer()
    tmpl_coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(template.GetNumAtoms())]
    )
    heavy_orig = [
        i for i in range(template.GetNumAtoms())
        if template.GetAtomWithIdx(i).GetAtomicNum() > 1
    ]
    query = Chem.RemoveHs(template)  # heavy-only graph for the fallback match

    def _tmpl_name(i):
        atom = template.GetAtomWithIdx(i)
        info = atom.GetMonomerInfo()
        if info is not None and info.GetName().strip():
            return info.GetName().strip()
        if atom.HasProp("molFileAlias"):
            return atom.GetProp("molFileAlias").strip()
        return None

    tmpl_names = [_tmpl_name(i) for i in range(template.GetNumAtoms())]
    names_usable = (
        all(n for n in tmpl_names) and len(set(tmpl_names)) == len(tmpl_names)
    )

    copies = extract_cofactor_copies(complex_pdb, resnames)
    if not copies:
        raise ValueError(
            f"No residues named {sorted(_resname_set(resnames))} found in {complex_pdb}"
        )

    out = []
    for key, atoms in copies:
        copy_coords = {a[0]: np.array(a[2]) for a in atoms}  # all atoms incl H
        mol = Chem.Mol(template)
        mol_conf = mol.GetConformer()

        # Primary: the PDB cofactor has hydrogens, so if every template atom name is
        # present in this copy, place the template on the EXACT crystal coordinates
        # (heavy + H) — highest fidelity, no superposition.
        if names_usable and all(n in copy_coords for n in tmpl_names):
            for i, name in enumerate(tmpl_names):
                mol_conf.SetAtomPosition(i, copy_coords[name].tolist())
        else:
            # Fallback: match the template's heavy-atom graph to the copy (ignoring
            # bond order/aromaticity) and rigidly fit the whole template onto the
            # copy's heavy atoms (template internal geometry).
            copy_mol = _copy_as_rdkit(atoms)
            match = _connectivity_match(query, copy_mol) if copy_mol is not None else ()
            if not match or len(match) != len(heavy_orig):
                raise ValueError(
                    f"cofactor copy {key}: could not match template to the PDB residue "
                    f"by atom name or graph. Ensure the template SDF is the same molecule "
                    f"as resname {key[0]} (ideally with matching PDB atom names)."
                )
            cc = copy_mol.GetConformer()
            matched_t = [tmpl_coords[heavy_orig[j]] for j in range(len(heavy_orig))]
            matched_q = [list(cc.GetAtomPosition(match[j])) for j in range(len(match))]
            rot, trans = _kabsch(matched_t, matched_q)
            placed = (tmpl_coords @ rot.T) + trans
            for i in range(mol.GetNumAtoms()):
                mol_conf.SetAtomPosition(i, placed[i].tolist())

        name = "".join(str(p) for p in key if str(p).strip()) or key[0]
        out.append(openfe.SmallMoleculeComponent.from_rdkit(mol, name=name))
    return out


def _copy_as_rdkit(atoms):
    """Build an RDKit mol for one cofactor copy from its real PDB records.

    Uses RDKit's own proximity bond perception (element covalent radii), which is far
    more reliable than a fixed distance cutoff for atoms like S/Cl/P. ``atoms`` entries
    are ``(name, element, xyz, raw_pdb_line)``.
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
    except Exception:
        return None
    lines = [a[3] for a in atoms if len(a) > 3 and a[1].upper() != "H"]
    if not lines:
        return None
    block = "\n".join(lines) + "\nEND\n"
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False, proximityBonding=True)
    if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        return None
    try:
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
        )
    except Exception:
        pass
    return mol


def _connectivity_match(query, copy_mol):
    """Match ``query`` (template heavy atoms) to ``copy_mol`` on element+connectivity.

    Both molecules are reduced to single bonds / no aromaticity / no formal charge so
    bond-order and aromaticity differences (common between a template SDF and a crystal
    PDB residue) don't prevent the match. Returns the atom-index tuple or ``()``.
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
    except Exception:
        return ()

    def _flatten(mol):
        m = Chem.RWMol(mol)
        for atom in m.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
            atom.SetNoImplicit(True)
        for bond in m.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        return m.GetMol()

    q = _flatten(query)
    c = _flatten(copy_mol)
    match = c.GetSubstructMatch(q, useChirality=False)
    return match

