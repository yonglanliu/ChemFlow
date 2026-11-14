"""Before/after quality control for OpenFE-charged ligand SDF files."""

from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChargeQCRecord:
    index: int
    name: str
    original_formal_charge: int
    assigned_partial_charge: float | None
    charge_difference: float | None
    connectivity_preserved: bool
    status: str
    message: str
    original_svg: str
    assigned_svg: str


def _molecules(path: Path):
    from rdkit import Chem

    molecules = [
        molecule
        for molecule in Chem.SDMolSupplier(str(path), removeHs=False)
        if molecule is not None
    ]
    if not molecules:
        raise ValueError(f"No valid molecules were found in {path}")
    return molecules


def _name(molecule, index: int) -> str:
    if molecule.HasProp("_Name") and molecule.GetProp("_Name").strip():
        return molecule.GetProp("_Name").strip()
    return f"ligand_{index:03d}"


def _atom_partial_charge(atom) -> float | None:
    property_names = (
        "PartialCharge",
        "partial_charge",
        "_TriposPartialCharge",
        "_GasteigerCharge",
    )
    for property_name in property_names:
        if atom.HasProp(property_name):
            try:
                value = float(atom.GetProp(property_name))
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    return None


def _partial_charge_sum(molecule) -> float | None:
    values = []
    for atom in molecule.GetAtoms():
        value = _atom_partial_charge(atom)
        if value is None or not math.isfinite(value):
            return None
        values.append(value)
    return sum(values) if values else None


def _canonical(molecule) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True)


def _svg(molecule, *, show_partial_charges: bool = False) -> str:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    copy = Chem.Mol(molecule)
    if show_partial_charges:
        for atom in copy.GetAtoms():
            charge = _atom_partial_charge(atom)
            if charge is not None:
                atom.SetProp("atomNote", f"{charge:+.2f}")
    rdDepictor.Compute2DCoords(copy)
    drawer = rdMolDraw2D.MolDraw2DSVG(330, 240)
    drawer.DrawMolecule(copy)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def compare_charged_sdf(
    original_sdf: str | Path,
    charged_sdf: str | Path,
    *,
    tolerance: float = 0.01,
) -> list[ChargeQCRecord]:
    """Compare structures and integer/formal versus summed partial charges."""
    from rdkit import Chem

    original = _molecules(Path(original_sdf).expanduser().resolve())
    charged = _molecules(Path(charged_sdf).expanduser().resolve())
    if len(original) != len(charged):
        raise ValueError(
            f"Molecule count changed during charging: {len(original)} → {len(charged)}"
        )
    records = []
    for index, (before, after) in enumerate(zip(original, charged, strict=True), start=1):
        name = _name(before, index)
        formal = int(Chem.GetFormalCharge(before))
        assigned = _partial_charge_sum(after)
        preserved = _canonical(before) == _canonical(after)
        difference = assigned - formal if assigned is not None else None
        problems = []
        if not preserved:
            problems.append("connectivity or stereochemistry changed")
        if assigned is None:
            problems.append("atom partial charges were not found")
        elif abs(difference) > tolerance:
            problems.append(
                f"partial-charge sum differs from formal charge by {difference:+.4f} e"
            )
        records.append(
            ChargeQCRecord(
                index=index,
                name=name,
                original_formal_charge=formal,
                assigned_partial_charge=assigned,
                charge_difference=difference,
                connectivity_preserved=preserved,
                status="PASS" if not problems else "WARNING",
                message="All QC checks passed" if not problems else "; ".join(problems),
                original_svg=_svg(before),
                assigned_svg=_svg(after, show_partial_charges=True),
            )
        )
    return records


def render_charge_qc_html(
    original_sdf: str | Path,
    charged_sdf: str | Path,
    *,
    tolerance: float = 0.01,
) -> str:
    records = compare_charged_sdf(original_sdf, charged_sdf, tolerance=tolerance)
    passed = sum(record.status == "PASS" for record in records)
    cards = []
    for record in records:
        assigned = (
            "not found"
            if record.assigned_partial_charge is None
            else f"{record.assigned_partial_charge:+.4f} e"
        )
        cards.append(
            f'''<section class="card">
<div class="heading"><strong>{html.escape(record.name)}</strong>
<span class="{record.status.lower()}">{record.status}</span></div>
<div class="comparison"><div><h3>Original</h3>{record.original_svg}
<p>Formal charge: <b>{record.original_formal_charge:+d} e</b></p></div>
<div class="arrow">→</div><div><h3>Assigned</h3>{record.assigned_svg}
<p>Σ partial charges: <b>{assigned}</b></p></div></div>
<p class="message">{html.escape(record.message)}</p></section>'''
        )
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f4f8fc;color:#20384d;margin:20px}}
h1{{color:#123f76;margin-bottom:4px}}.summary{{color:#60788d;margin-bottom:18px}}
.card{{background:white;border:1px solid #c8dceb;border-radius:12px;padding:14px;margin:14px 0;box-shadow:0 2px 8px #2343}}
.heading{{display:flex;justify-content:space-between;font-size:18px}}.pass,.warning{{padding:4px 10px;border-radius:12px;font-size:12px}}
.pass{{background:#dff4e5;color:#176634}}.warning{{background:#fff0ce;color:#915d00}}
.comparison{{display:grid;grid-template-columns:1fr 56px 1fr;align-items:center;text-align:center}}
.comparison svg{{width:100%;max-width:330px;height:240px}}.arrow{{font-size:40px;color:#2f7ed8}}h3{{color:#315f7d}}
.message{{border-top:1px solid #e2edf5;padding-top:10px;color:#526b82}}
</style></head><body><h1>OpenFE Partial-Charge QC</h1>
<div class="summary">{passed}/{len(records)} compounds passed · tolerance ±{tolerance:.3f} e</div>
{''.join(cards)}</body></html>'''


def write_charge_qc(
    original_sdf: Path, charged_sdf: Path, html_path: Path, csv_path: Path
) -> None:
    records = compare_charged_sdf(original_sdf, charged_sdf)
    html_path.write_text(
        render_charge_qc_html(original_sdf, charged_sdf), encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "name",
                "formal_charge",
                "partial_charge_sum",
                "difference",
                "connectivity_preserved",
                "status",
                "message",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.index,
                    record.name,
                    record.original_formal_charge,
                    record.assigned_partial_charge,
                    record.charge_difference,
                    record.connectivity_preserved,
                    record.status,
                    record.message,
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_sdf", type=Path)
    parser.add_argument("charged_sdf", type=Path)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        write_charge_qc(args.original_sdf, args.charged_sdf, args.html, args.csv)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
