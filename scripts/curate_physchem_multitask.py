#!/usr/bin/env python3
"""Build the model-ready physicochemical multitask dataset.

The archival grouped table retains assay qualifiers and source-specific rows.
This script keeps exact measurements, consolidates canonical-SMILES duplicates,
and deliberately separates ExpansionRx solubility from the NIH/NCATS pH 7.4
kinetic-solubility endpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NIH_NCATS_SOURCES = {
    "NIH_PubChem_AID_1996",
    "NCATS_AID_1645848",
}
EXPANSIONRX_SOURCES = {
    "ExpansionRX_train",
    "ExpansionRX_test",
}

TARGET_COLUMNS = [
    "LogD_pH7_4",
    "Log_KSOL_pH6_8_mol_L",
    "Log_KSOL_pH7_4_mol_L",
    "Log_KSOL_ExpansionRX_mol_L",
]


def _join_unique(values: pd.Series) -> str | float:
    unique: list[str] = []
    for value in values.dropna():
        for item in str(value).split(" | "):
            item = item.strip()
            if item and item not in unique:
                unique.append(item)
    return " | ".join(unique) if unique else np.nan


def curate(source_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_path, low_memory=False)
    required = {
        "Molecule_Name",
        "Canonical_SMILES",
        "source",
        "dataset_split",
        "LogD",
        "LogD_qualifier",
        "Log_KSOL_pH6_8_mol_L",
        "KSOL_pH6_8_qualifier",
        "Log_KSOL_pH7_4_mol_L",
        "KSOL_pH7_4_qualifier",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Archival table is missing required columns: {missing}")

    exact_logd = pd.to_numeric(frame["LogD"], errors="coerce").where(
        frame["LogD_qualifier"].eq("=")
    )
    exact_ksol_68 = pd.to_numeric(
        frame["Log_KSOL_pH6_8_mol_L"], errors="coerce"
    ).where(frame["KSOL_pH6_8_qualifier"].eq("="))
    exact_ksol_source = pd.to_numeric(
        frame["Log_KSOL_pH7_4_mol_L"], errors="coerce"
    ).where(frame["KSOL_pH7_4_qualifier"].eq("="))

    working = pd.DataFrame(
        {
            "Molecule Name": frame["Molecule_Name"],
            "SMILES": frame["Canonical_SMILES"],
            "source": frame["source"],
            "dataset_split": frame["dataset_split"],
            "LogD_pH7_4": exact_logd,
            "Log_KSOL_pH6_8_mol_L": exact_ksol_68,
            "Log_KSOL_pH7_4_mol_L": exact_ksol_source.where(
                frame["source"].isin(NIH_NCATS_SOURCES)
            ),
            "Log_KSOL_ExpansionRX_mol_L": exact_ksol_source.where(
                frame["source"].isin(EXPANSIONRX_SOURCES)
            ),
        }
    )
    working = working[working[TARGET_COLUMNS].notna().any(axis=1)].copy()

    grouped = working.groupby("SMILES", sort=False, dropna=False)
    curated = grouped[TARGET_COLUMNS].mean()
    curated.insert(0, "dataset_split", grouped["dataset_split"].apply(_join_unique))
    curated.insert(0, "source", grouped["source"].apply(_join_unique))
    curated.insert(0, "Molecule Name", grouped["Molecule Name"].apply(_join_unique))
    curated = curated.reset_index()
    curated = curated[
        [
            "Molecule Name",
            "SMILES",
            "source",
            "dataset_split",
            *TARGET_COLUMNS,
        ]
    ]

    if curated["SMILES"].isna().any() or curated["SMILES"].duplicated().any():
        raise AssertionError("Curated output must contain one nonmissing canonical SMILES per row.")
    if not np.isfinite(curated[TARGET_COLUMNS].stack().to_numpy(dtype=float)).all():
        raise AssertionError("Curated targets must be finite numeric values.")
    if (
        curated["Log_KSOL_pH7_4_mol_L"].notna()
        & curated["Log_KSOL_ExpansionRX_mol_L"].notna()
    ).any():
        raise AssertionError(
            "A structure unexpectedly contains both NIH/NCATS and ExpansionRx solubility."
        )
    return curated


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repository / "dataset/ADMET/grouped/physchem/physchem_multitask.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "dataset/curated/physchem_multitask.csv",
    )
    args = parser.parse_args()

    curated = curate(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(args.output, index=False)

    print(f"Saved {len(curated):,} unique structures to {args.output}")
    for column in TARGET_COLUMNS:
        print(f"  {column}: {curated[column].notna().sum():,} labels")


if __name__ == "__main__":
    main()
