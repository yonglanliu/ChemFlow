"""Turn a clicked atom on each ligand ("head") into a LinkInvent warhead.

LinkInvent takes one input line per design: ``<head1>|<head2>``, where each head
is a SMILES with a single ``*`` marking where the linker attaches. Clicking an
atom means: replace one hydrogen on that atom with ``*``. If a 3D pose is
available the exit vector (the direction the linker leaves) is recorded, and
when several hydrogens are possible the most solvent-exposed one is chosen.

RDKit is imported lazily so this module can be imported without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class HeadSpec:
    """One PROTAC end with its chosen attachment point."""

    name: str
    smiles: str                       # warhead SMILES containing exactly one '*'
    attach_atom: int                  # index of the clicked heavy atom (input numbering)
    mol: Any = None                   # RDKit Mol incl. the dummy; has a conformer if 3D
    exit_origin: tuple[float, float, float] | None = None
    exit_direction: tuple[float, float, float] | None = None
    has_3d: bool = False
    notes: list[str] = field(default_factory=list)


def load_molecule(path: str | Path, *, sdf_index: int = 0):
    """Read a ligand from .sdf/.mol/.pdb/.mol2 keeping explicit hydrogens.

    Sulfonyl / phosphoryl groups that were exported with an S=O or P=O as a single
    bond (common from PyMOL) are repaired on load via :func:`_fix_oxo_bond_orders`,
    so the fixed structure is used everywhere downstream (3D view, atom picking and
    the warhead SMILES).
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"ligand file not found: {path}")
    from rdkit import Chem

    suffix = path.suffix.lower()
    if suffix in (".sdf", ".sd"):
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mols = [m for m in supplier if m is not None]
        if not mols:
            raise ValueError(f"no readable molecule in {path}")
        if sdf_index >= len(mols):
            raise ValueError(f"{path} has only {len(mols)} molecule(s); index {sdf_index} requested")
        return _fix_oxo_bond_orders(mols[sdf_index])
    if suffix == ".mol":
        mol = Chem.MolFromMolFile(str(path), removeHs=False)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False)
    else:
        raise ValueError(f"unsupported ligand format: {suffix} (use .sdf, .mol, .mol2, .pdb)")
    if mol is None:
        raise ValueError(f"could not read a molecule from {path}")
    return _fix_oxo_bond_orders(mol)


def mol_from_smiles(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return mol


def _pick_hydrogen(work, candidates: Sequence[int], protein_xyz) -> int:
    """Choose the H whose direction is most open (needs a conformer + protein)."""
    if len(candidates) == 1 or not work.GetNumConformers() or protein_xyz is None or len(protein_xyz) == 0:
        return candidates[0]
    import numpy as np

    conf = work.GetConformer()
    best, best_dist = candidates[0], -1.0
    for idx in candidates:
        pos = conf.GetAtomPosition(idx)
        point = np.array([pos.x, pos.y, pos.z])
        nearest = float(np.min(np.linalg.norm(np.asarray(protein_xyz) - point, axis=1)))
        if nearest > best_dist:
            best, best_dist = idx, nearest
    return best


def _fix_oxo_bond_orders(mol):
    """Fix sulfonyl / phosphoryl groups exported with an S=O or P=O as a *single*
    bond (common from PyMOL), which leaves the central atom with an odd valence.

    RDKit then adds a spurious hydrogen to reach the next allowed valence, so a
    sulfonamide ``C-S(=O)(=O)-N`` can turn into ``[SH]`` and crash LinkInvent
    (its vocabulary has no ``[SH]``). This promotes terminal, neutral S–O / P–O
    single bonds to double until the central atom's valence is no longer odd.
    Atom indices are preserved (only bond orders change).
    """
    from rdkit import Chem

    rw = Chem.RWMol(mol)

    def _valence(a) -> int:
        return int(round(sum(b.GetBondTypeAsDouble() for b in a.GetBonds())))

    changed = False
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() not in (15, 16):          # P, S
            continue
        for bond in atom.GetBonds():
            if _valence(atom) % 2 == 0:
                break
            other = bond.GetOtherAtom(atom)
            if (other.GetAtomicNum() == 8 and other.GetDegree() == 1
                    and other.GetFormalCharge() == 0
                    and bond.GetBondType() == Chem.BondType.SINGLE):
                bond.SetBondType(Chem.BondType.DOUBLE)
                changed = True
    if not changed:
        return mol
    fixed = rw.GetMol()
    try:
        Chem.SanitizeMol(fixed)
    except Exception:
        return mol                                       # keep the original if the fix is invalid
    return fixed


def _normalize_implicit_hydrogens(mol) -> None:
    """Rewrite neutral, non-aromatic heavy atoms to use implicit hydrogens.

    After the AddHs -> set-dummy -> RemoveHs sequence, RDKit can leave a heavy
    atom flagged ``noImplicit`` with an explicit H count, so ``MolToSmiles`` emits
    bracket tokens such as ``[SH]`` or ``[OH]``. REINVENT4's LinkInvent vocabulary
    has the plain ``S`` / ``O`` / ``N`` tokens but not those bracketed H forms, so
    a warhead with a thiol/alcohol would crash the run. Clearing the flags on
    neutral, non-aromatic heavy atoms and re-sanitising restores the plain tokens.
    Aromatic and charged atoms are left untouched because their bracket tokens
    (e.g. ``[nH]``, ``[N+]``, ``[O-]``) are in the vocabulary.
    """
    from rdkit import Chem

    changed = False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        if atom.GetIsAromatic() or atom.GetFormalCharge() != 0:
            continue
        if atom.GetNoImplicit() or atom.GetNumExplicitHs():
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
            changed = True
    if changed:
        Chem.SanitizeMol(mol)


def make_head(mol, atom_index: int, *, name: str = "head", protein_xyz=None) -> HeadSpec:
    """Build a :class:`HeadSpec` by attaching ``*`` at ``atom_index``.

    ``atom_index`` is an atom of ``mol`` as given (an index into the structure
    the user clicked). Clicking a hydrogen attaches the linker where that H was;
    clicking a heavy atom uses one of its hydrogens.
    """
    from rdkit import Chem

    if atom_index < 0 or atom_index >= mol.GetNumAtoms():
        raise ValueError(f"{name}: atom index {atom_index} is out of range (0-{mol.GetNumAtoms() - 1})")
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError(f"{name}: the ligand already contains a dummy atom (*)")

    work = Chem.Mol(mol)
    work = _fix_oxo_bond_orders(work)
    has_explicit_h = any(a.GetAtomicNum() == 1 for a in work.GetAtoms())
    if not has_explicit_h:
        # AddHs appends hydrogens at the end, so the original indices are unchanged.
        work = Chem.AddHs(work, addCoords=bool(work.GetNumConformers()))

    atom = work.GetAtomWithIdx(atom_index)
    if atom.GetAtomicNum() == 1:
        h_idx = atom_index
        attach_idx = atom.GetNeighbors()[0].GetIdx()
    else:
        attach_idx = atom_index
        hydrogens = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        if not hydrogens:
            raise ValueError(
                f"{name}: atom {atom_index} ({atom.GetSymbol()}) has no free hydrogen, "
                "so a linker cannot be attached there. Pick an atom that carries an H."
            )
        h_idx = _pick_hydrogen(work, hydrogens, protein_xyz)

    has_3d = bool(work.GetNumConformers())
    origin = direction = None
    if has_3d:
        conf = work.GetConformer()
        a, h = conf.GetAtomPosition(attach_idx), conf.GetAtomPosition(h_idx)
        vec = (h.x - a.x, h.y - a.y, h.z - a.z)
        norm = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5 or 1.0
        origin = (a.x, a.y, a.z)
        direction = (vec[0] / norm, vec[1] / norm, vec[2] / norm)

    editable = Chem.RWMol(work)
    editable.GetAtomWithIdx(h_idx).SetAtomicNum(0)       # H -> dummy atom '*'
    capped = Chem.RemoveHs(editable)                     # drop the remaining explicit H
    dummies = [a for a in capped.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        raise ValueError(f"{name}: expected one attachment point, found {len(dummies)}")
    _normalize_implicit_hydrogens(capped)
    # Non-isomeric on purpose: REINVENT4's default LinkInvent vocabulary and its
    # SMILES randomisation do not use stereo tokens. The 3D mol keeps the pose.
    smiles = Chem.MolToSmiles(capped, isomericSmiles=False)
    if smiles.count("*") != 1:
        raise ValueError(f"{name}: unexpected warhead SMILES {smiles!r}")

    return HeadSpec(
        name=name,
        smiles=smiles,
        attach_atom=attach_idx,
        mol=capped,
        exit_origin=origin,
        exit_direction=direction,
        has_3d=has_3d,
    )


def warhead_line(head1: HeadSpec, head2: HeadSpec) -> str:
    """The LinkInvent input line: ``<head1>|<head2>``."""
    return f"{head1.smiles}|{head2.smiles}"
