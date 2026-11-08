"""Built-in example for the PROTAC linker designer.

"Try example" needs two ready ligands. Rather than ship binary structure files
or download/split a PROTAC from the PDB (fragile), we keep two warheads as SMILES
and let RDKit build clean 3D structures on first use. Each warhead names a
chemically sensible attachment point (where the linker would join), so the
desktop can pre-select it and the user can generate immediately.

The two example warheads are a real CRBN E3-ligase binder (pomalidomide, joined
at its aromatic amine, the usual PROTAC handle) and a simple aromatic target
binder (joined at the hydroxyl). They are illustrative: enough to show the
click-to-linker workflow, not a validated ternary complex.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExampleWarhead:
    key: str
    name: str
    smiles: str
    attach_smarts: str          # first matching atom is the attachment point
    note: str = ""


EXAMPLE_WARHEADS = (
    ExampleWarhead(
        key="pomalidomide",
        name="Pomalidomide (CRBN E3 binder)",
        smiles="Nc1cccc2c1C(=O)N(C3CCC(=O)NC3=O)C2=O",
        attach_smarts="[NX3;H2]",            # the aromatic amine — pomalidomide's usual PROTAC handle
        note="Real cereblon ligand; linker is attached at the 4-amino group.",
    ),
    ExampleWarhead(
        key="aryl_target",
        name="Example target binder (aryl)",
        smiles="Oc1ccc(cc1)c1ccccc1",         # 4-phenylphenol
        attach_smarts="[OX2H]",              # the phenol oxygen
        note="Illustrative target warhead; linker attached at the hydroxyl.",
    ),
)


def _attachment_index(mol, smarts: str) -> int:
    from rdkit import Chem

    query = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(query)
    if not matches:
        raise ValueError(f"attachment SMARTS {smarts!r} not found in example warhead")
    return int(matches[0][0])


def build_example_warhead(warhead: ExampleWarhead, dest_dir: str | Path, *, seed: int = 7):
    """Embed a warhead to 3D and write ``<key>.sdf``; return (path, attach_index).

    The written SDF's heavy-atom order is preserved on reload, so ``attach_index``
    stays valid for :func:`chemflow.generative.reinvent.linker.heads.make_head`.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    mol = Chem.MolFromSmiles(warhead.smiles)
    if mol is None:
        raise ValueError(f"example warhead {warhead.key!r} has an invalid SMILES")
    attach_index = _attachment_index(mol, warhead.attach_smarts)

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise RuntimeError(f"could not generate 3D coordinates for {warhead.key!r}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    path = dest_dir / f"{warhead.key}.sdf"
    writer = Chem.SDWriter(str(path))
    try:
        mol.SetProp("_Name", warhead.name)
        writer.write(mol)
    finally:
        writer.close()
    return path, attach_index


def ensure_example(dest_dir: str | Path) -> list[dict]:
    """Make sure both example warheads exist under ``dest_dir``.

    Returns ``[{key, name, path, attach_index, note}, ...]`` for head 1 then 2.
    Safe to call repeatedly; it rebuilds only missing files.
    """
    dest_dir = Path(dest_dir)
    result = []
    for warhead in EXAMPLE_WARHEADS:
        path, attach_index = build_example_warhead(warhead, dest_dir)
        result.append({
            "key": warhead.key, "name": warhead.name, "path": path,
            "attach_index": attach_index, "note": warhead.note,
        })
    return result
