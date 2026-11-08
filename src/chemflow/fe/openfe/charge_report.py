"""RDKit-only helpers to read per-ligand net charges from SDF files.

Shared by the ``openfe charge`` CLI report and the desktop Ligand-Prep table.
``formal_charges`` = net formal charge of the input; ``partial_charge_sums`` =
sum of assigned partial charges in a charged SDF (openff ``PartialCharge`` tag).
"""

from __future__ import annotations

from pathlib import Path


def formal_charges(sdf: str | Path) -> list[tuple[str, int]]:
    from rdkit import Chem

    out = []
    for i, mol in enumerate(Chem.SDMolSupplier(str(sdf), removeHs=False)):
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol{i + 1}"
        out.append((name or f"mol{i + 1}", Chem.GetFormalCharge(mol)))
    return out


def _partial_sum(mol) -> float | None:
    for key in mol.GetPropNames():
        if "partialcharge" in key.lower():
            try:
                return sum(float(x) for x in mol.GetProp(key).split())
            except ValueError:
                continue
    return None


def partial_charge_sums(sdf: str | Path) -> dict[str, float | None]:
    from rdkit import Chem

    sums: dict[str, float | None] = {}
    for i, mol in enumerate(Chem.SDMolSupplier(str(sdf), removeHs=False)):
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol{i + 1}"
        sums[name or f"mol{i + 1}"] = _partial_sum(mol)
    return sums


def molecule_svg(sdf: str | Path, name: str, size=(380, 300)) -> str:
    """2D depiction (SVG) of the molecule named ``name`` in ``sdf`` (empty if absent)."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    target = None
    for i, mol in enumerate(Chem.SDMolSupplier(str(sdf), removeHs=False)):
        if mol is None:
            continue
        nm = (mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol{i + 1}") or f"mol{i + 1}"
        if nm == name:
            target = mol
            break
    if target is None:
        return ""
    drawing = Chem.Mol(target)
    try:
        drawing = Chem.RemoveHs(drawing)
    except Exception:
        pass
    rdDepictor.Compute2DCoords(drawing)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, drawing)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.find("<svg"):]

