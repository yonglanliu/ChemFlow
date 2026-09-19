"""Build a harmonized HLM training table from Biogen, TDC, and ExpansionRX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPOSITORY_ROOT / "dataset" / "ADMET"
OUTPUT_DIR = REPOSITORY_ROOT / "dataset" / "combined"


def canonicalize_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def prepare_biogen() -> pd.DataFrame:
    path = DATASET_ROOT / "Biogen_ADME" / "Biogen_ADME_selected.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["Log_HLM_CLint"].notna()].copy()
    frame = frame.rename(columns={"Internal ID": "compound_id"})
    frame["HLM_CLint_mL_min_kg"] = np.power(10.0, frame["Log_HLM_CLint"])
    frame["source"] = "Biogen_ADME"
    frame["original_split"] = pd.NA
    return frame[[
        "compound_id",
        "SMILES",
        "HLM_CLint_mL_min_kg",
        "Log_HLM_CLint",
        "source",
        "original_split",
    ]]


def prepare_tdc() -> pd.DataFrame:
    path = (
        DATASET_ROOT
        / "TDC_ADME"
        / "data"
        / "clearance_microsome_az_log_ml_min_kg.csv"
    )
    frame = pd.read_csv(path)
    frame = frame.loc[frame["Log_HLM_CLint"].notna()].copy()
    frame = frame.rename(columns={"ID": "compound_id"})
    frame["source"] = "TDC_Clearance_Microsome_AZ"
    frame["original_split"] = pd.NA
    return frame[[
        "compound_id",
        "SMILES",
        "HLM_CLint_mL_min_kg",
        "Log_HLM_CLint",
        "source",
        "original_split",
    ]]


def prepare_expansionrx() -> pd.DataFrame:
    path = DATASET_ROOT / "ExpansionRX" / "expansion_data_train.csv"
    frame = pd.read_csv(path)

    # Use log10(x), not expansionrx_train_df.csv's log10(1 + x), to match
    # the Biogen endpoint. Missing, zero, and explicitly censored measurements
    # are already absent from this cleaned log column.
    frame = frame.loc[frame["Log(HLM_Clint)"].notna()].copy()
    frame = frame.rename(columns={
        "Molecule Name": "compound_id",
        "Log(HLM_Clint)": "Log_HLM_CLint",
        "HLM CLint": "HLM_CLint_mL_min_kg",
        "split": "original_split",
    })
    frame["source"] = "ExpansionRX_OpenADMET"
    return frame[[
        "compound_id",
        "SMILES",
        "HLM_CLint_mL_min_kg",
        "Log_HLM_CLint",
        "source",
        "original_split",
    ]]


def build_combined_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = [prepare_expansionrx(), prepare_biogen(), prepare_tdc()]
    combined = pd.concat(frames, ignore_index=True)
    combined["original_SMILES"] = combined["SMILES"]
    combined["SMILES"] = combined["SMILES"].map(canonicalize_smiles)

    if combined["Log_HLM_CLint"].isna().any():
        raise ValueError("The combined target contains missing values.")
    if not np.isfinite(combined["Log_HLM_CLint"]).all():
        raise ValueError("The combined target contains non-finite values.")

    expected_log = np.log10(combined["HLM_CLint_mL_min_kg"])
    if not np.allclose(combined["Log_HLM_CLint"], expected_log, atol=1e-9):
        raise ValueError("One or more target values are not log10(mL/min/kg).")

    source_count = combined.groupby("SMILES")["source"].transform("nunique")
    combined["cross_source_overlap"] = source_count.gt(1)
    combined["at_source_minimum"] = combined.groupby("source")[
        "Log_HLM_CLint"
    ].transform(lambda values: values.eq(values.min()))
    combined["at_source_maximum"] = combined.groupby("source")[
        "Log_HLM_CLint"
    ].transform(lambda values: values.eq(values.max()))

    priority = {
        "ExpansionRX_OpenADMET": 0,
        "Biogen_ADME": 1,
        "TDC_Clearance_Microsome_AZ": 2,
    }
    combined["_source_priority"] = combined["source"].map(priority)
    combined = combined.sort_values(
        ["SMILES", "_source_priority", "compound_id"],
        kind="stable",
    ).reset_index(drop=True)

    duplicate_mask = combined.duplicated("SMILES", keep="first")
    removed = combined.loc[duplicate_mask].copy()
    training = combined.loc[~duplicate_mask].copy()

    output_columns = [
        "compound_id",
        "SMILES",
        "original_SMILES",
        "Log_HLM_CLint",
        "HLM_CLint_mL_min_kg",
        "source",
        "original_split",
        "cross_source_overlap",
        "at_source_minimum",
        "at_source_maximum",
    ]
    return (
        combined[output_columns],
        training[output_columns],
        removed[output_columns],
    )


def main() -> None:
    combined, training, removed = build_combined_tables()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_path = OUTPUT_DIR / "HLM_Biogen_TDC_ExpansionRX_all_measurements.csv"
    training_path = OUTPUT_DIR / "HLM_Biogen_TDC_ExpansionRX_training.csv"
    removed_path = OUTPUT_DIR / "HLM_Biogen_TDC_ExpansionRX_overlap_removed.csv"
    combined.to_csv(all_path, index=False)
    training.to_csv(training_path, index=False)
    removed.to_csv(removed_path, index=False)

    print(f"All measurements: {len(combined):,} -> {all_path}")
    print(f"Unique-molecule training rows: {len(training):,} -> {training_path}")
    print(f"Overlapping rows removed: {len(removed):,} -> {removed_path}")
    print("\nTraining rows by source:")
    print(training["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
