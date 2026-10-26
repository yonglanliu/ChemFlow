"""Charge-change classification and staging for GROMACS ligand RBFE edges."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChargePlan:
    """Auditable charge treatment generated for a ligand network."""

    method: str
    changes_csv: Path
    manifest_json: Path
    charge_changing_edges: int


def _integer_charge(value: str, label: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer net charge") from error
    rounded = round(numeric)
    if abs(numeric - rounded) > 1.0e-6:
        raise ValueError(f"{label} must be an integer net charge, got {value!r}")
    return rounded


def read_charge_manifest(path: Path) -> dict[str, int]:
    """Read the charge assignments produced by GROMACS ligand preparation."""
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"Ligand charge manifest is empty: {path}")
    charges: dict[str, int] = {}
    for number, row in enumerate(rows, start=2):
        name = (row.get("ligand_name") or row.get("name") or "").strip()
        if not name:
            raise ValueError(f"Missing ligand_name in {path} line {number}")
        if name in charges:
            raise ValueError(f"Duplicate ligand_name in charge manifest: {name}")
        value = row.get("net_charge")
        if value is None or not value.strip():
            raise ValueError(f"Missing net_charge for ligand {name!r}")
        charges[name] = _integer_charge(value, f"net_charge for {name!r}")
    return charges


def charges_from_sdf(path: Path) -> dict[str, int]:
    """Fallback to formal charges when no prepared-ligand manifest is supplied."""
    from rdkit import Chem

    charges: dict[str, int] = {}
    for number, molecule in enumerate(
        Chem.SDMolSupplier(str(path), removeHs=False), start=1
    ):
        if molecule is None:
            raise ValueError(f"Cannot read ligand record {number} from {path}")
        name = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
        if not name:
            raise ValueError(f"Ligand record {number} has no _Name")
        if name in charges:
            raise ValueError(f"Duplicate ligand name in SDF: {name}")
        charges[name] = int(Chem.GetFormalCharge(molecule))
    return charges


def _read_corrections(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    corrections: dict[str, tuple[float, float]] = {}
    required = ("edge_id", "complex_correction_kj_mol", "solvent_correction_kj_mol")
    for number, row in enumerate(rows, start=2):
        if any(not (row.get(key) or "").strip() for key in required):
            raise ValueError(
                f"Charge correction table line {number} must define: "
                + ", ".join(required)
            )
        try:
            corrections[row["edge_id"].strip()] = (
                float(row["complex_correction_kj_mol"]),
                float(row["solvent_correction_kj_mol"]),
            )
        except ValueError as error:
            raise ValueError(
                f"Non-numeric correction in {path} line {number}"
            ) from error
    return corrections


def _coion_itp(ion: str, force_field: str) -> tuple[str, str]:
    charmm = "charmm" in force_field.casefold()
    if ion == "NA":
        residue, atom_type, charge, mass = (
            ("NA", "SOD", 1.0, 22.98977)
            if charmm
            else ("NA", "Na", 1.0, 22.98977)
        )
    else:
        residue, atom_type, charge, mass = (
            ("CL", "CLA", -1.0, 35.453)
            if charmm
            else ("CL", "Cl", -1.0, 35.453)
        )
    text = (
        "; Generated alchemical co-ion topology.\n"
        "; State A is the physical ion; state B is neutral with unchanged LJ.\n"
        "[ moleculetype ]\n"
        f"{residue}_COION  1\n\n"
        "[ atoms ]\n"
        "; nr type resnr residue atom cgnr charge mass typeB chargeB massB\n"
        f"1 {atom_type} 1 {residue} {residue} 1 {charge:.1f} {mass:.5f} "
        f"{atom_type} 0.0 {mass:.5f}\n"
        "\n[ position_restraints ]\n"
        "; ai funct fcx fcy fcz; keep the alchemical ion away from the solute\n"
        "1 1 1000 1000 1000\n"
    )
    return residue, text


def build_charge_plan(
    *,
    edges_csv: Path,
    ligand_library: Path,
    output_directory: Path,
    method: str,
    charge_manifest: Path | None = None,
    correction_file: Path | None = None,
    force_field: str = "amber",
) -> ChargePlan:
    """Classify every edge and stage an explicit charge-change treatment."""
    if method not in {"reject", "coion", "correction"}:
        raise ValueError("Charge-change method must be reject, coion, or correction")
    charges = (
        read_charge_manifest(charge_manifest)
        if charge_manifest is not None
        else charges_from_sdf(ligand_library)
    )
    with edges_csv.open(newline="", encoding="utf-8") as handle:
        edges = list(csv.DictReader(handle))
    corrections = _read_corrections(correction_file)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    changing: list[dict[str, object]] = []
    for number, edge in enumerate(edges, start=1):
        edge_id = (edge.get("edge_id") or f"edge_{number:04d}").strip()
        ligand_a = (edge.get("ligand_a") or "").strip()
        ligand_b = (edge.get("ligand_b") or "").strip()
        missing = [name for name in (ligand_a, ligand_b) if name not in charges]
        if missing:
            raise ValueError(
                "Charge manifest has no assignment for network ligand(s): "
                + ", ".join(missing)
            )
        charge_a, charge_b = charges[ligand_a], charges[ligand_b]
        delta = charge_b - charge_a
        row: dict[str, object] = {
            "edge_id": edge_id,
            "ligand_a": ligand_a,
            "ligand_b": ligand_b,
            "charge_a": charge_a,
            "charge_b": charge_b,
            "delta_charge": delta,
            "treatment": "none" if delta == 0 else method,
            "coion": "",
            "coion_count": 0,
            "coion_charge_a": "",
            "coion_charge_b": "",
            "complex_correction_kj_mol": "",
            "solvent_correction_kj_mol": "",
            "ddg_correction_kj_mol": "",
        }
        if delta:
            changing.append(row)
            if method == "coion":
                # The physical counterion is neutralized as the ligand acquires
                # charge, preserving the total cell charge at every lambda.
                ion = "NA" if delta > 0 else "CL"
                edge_dir = output_directory / edge_id
                edge_dir.mkdir(parents=True, exist_ok=True)
                residue, topology = _coion_itp(ion, force_field)
                (edge_dir / "coion.itp").write_text(topology, encoding="utf-8")
                row["coion"] = residue
                row["coion_count"] = abs(delta)
                row["coion_charge_a"] = 1 if delta > 0 else -1
                row["coion_charge_b"] = 0
            elif method == "correction":
                if edge_id not in corrections:
                    raise ValueError(
                        f"Charge correction table has no values for {edge_id}"
                    )
                complex_value, solvent_value = corrections[edge_id]
                row["complex_correction_kj_mol"] = complex_value
                row["solvent_correction_kj_mol"] = solvent_value
                row["ddg_correction_kj_mol"] = complex_value - solvent_value
        rows.append(row)
    if changing and method == "reject":
        summary = ", ".join(
            f"{row['edge_id']} ({row['ligand_a']}->{row['ligand_b']}: "
            f"{int(row['delta_charge']):+d})"
            for row in changing
        )
        raise ValueError(
            "Charge-changing RBFE edges require co-ion handling or a validated "
            f"correction table: {summary}"
        )

    changes_csv = output_directory.parent / "network" / "charge_changes.csv"
    changes_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "edge_id", "ligand_a", "ligand_b", "charge_a", "charge_b",
        "delta_charge", "treatment", "coion", "coion_count",
        "coion_charge_a", "coion_charge_b",
        "complex_correction_kj_mol", "solvent_correction_kj_mol",
        "ddg_correction_kj_mol",
    ]
    with changes_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_json = output_directory.parent / ".alchemforge" / "charge_strategy.json"
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(
        json.dumps(
            {
                "method": method,
                "charge_source": str(charge_manifest or ligand_library),
                "charge_changing_edges": len(changing),
                "legs": ["complex", "solvent"],
                "requirements": (
                    [
                        "Insert the listed physical ion(s) into each leg.",
                        "Include the generated coion.itp in each leg topology.",
                        "Use the same ion selection protocol in complex and solvent.",
                        "Verify total A- and B-state charge before every grompp call.",
                    ]
                    if method == "coion"
                    else [
                        (
                            "Apply the supplied complex-minus-solvent correction only "
                            "after the uncorrected RBFE analysis."
                        )
                    ]
                    if method == "correction"
                    else []
                ),
                "edges": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ChargePlan(method, changes_csv, manifest_json, len(changing))
