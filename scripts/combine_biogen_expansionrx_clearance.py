"""Build a compact, log10-transformed clearance training table.

The output combines Biogen with the unsplit ExpansionRX raw clearance data.
Censored measurements and nonpositive values are treated as missing labels
because they cannot be used as exact log10 regression targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLEARANCE_DIR = REPOSITORY_ROOT / "dataset" / "ADMET" / "curated" / "clearance"
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "dataset"
    / "combined"
    / "Biogen_ExpansionRX_clearance_training.csv"
)

RAW_TO_LOG_COLUMNS = {
    "HLM_CLint (mL/min/kg)": "Log_HLM_CLint",
    "RLM_CLint (mL/min/kg)": "Log_RLM_CLint",
    "MLM_CLint (mL/min/kg)": "Log_MLM_CLint",
}
LOG_COLUMNS = list(RAW_TO_LOG_COLUMNS.values())
OUTPUT_COLUMNS = ["Molecule Name", "SMILES", "source", *LOG_COLUMNS]


def canonicalize_smiles(smiles: str) -> str:
    """Return an isomeric canonical SMILES or raise for an invalid structure."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def prepare_source(
    path: Path,
    molecule_name_column: str,
    default_source: str,
) -> pd.DataFrame:
    """Read one source and convert exact positive clearance values to log10."""
    frame = pd.read_csv(path)
    if "SMILES" not in frame:
        raise KeyError(f"{path} does not contain a SMILES column.")
    if molecule_name_column not in frame:
        raise KeyError(f"{path} does not contain {molecule_name_column!r}.")

    prepared = pd.DataFrame(
        {
            "Molecule Name": frame[molecule_name_column].astype("string"),
            "SMILES": frame["SMILES"].map(canonicalize_smiles),
            "source": (
                frame["source"].fillna(default_source).astype("string")
                if "source" in frame
                else default_source
            ),
        }
    )
    for raw_column, log_column in RAW_TO_LOG_COLUMNS.items():
        if raw_column not in frame:
            prepared[log_column] = np.nan
            continue

        # errors="coerce" intentionally excludes interval-censored strings such
        # as "< 9.3" and "> 528.1" from exact-value regression targets.
        numeric = pd.to_numeric(frame[raw_column], errors="coerce")
        prepared[log_column] = np.log10(numeric.where(numeric.gt(0)))

    return prepared


def build_training_table() -> pd.DataFrame:
    """Combine sources, remove empty rows, and collapse repeated structures."""
    biogen = prepare_source(
        CLEARANCE_DIR / "Biogen_clearance.csv",
        molecule_name_column="Internal ID",
        default_source="Biogen_ADME_Fang_2023",
    )
    expansionrx = prepare_source(
        CLEARANCE_DIR / "ExpansionRX_raw_clearance.csv",
        molecule_name_column="Molecule Name",
        default_source="ExpansionRX_OpenADMET_raw",
    )
    combined = pd.concat([biogen, expansionrx], ignore_index=True)
    combined = combined.dropna(subset=LOG_COLUMNS, how="all")

    # Averaging in log space gives one row per structure and prevents exact
    # duplicates from being assigned to different train/validation/test splits.
    # All identifiers and sources attached to a repeated structure are retained.
    training = combined.groupby("SMILES", as_index=False, sort=False).agg(
        {
            "Molecule Name": lambda values: ";".join(
                dict.fromkeys(values.dropna().astype(str))
            ),
            "source": lambda values: ";".join(
                dict.fromkeys(values.dropna().astype(str))
            ),
            **{column: "mean" for column in LOG_COLUMNS},
        }
    )
    training = training.reindex(columns=OUTPUT_COLUMNS)

    values = training[LOG_COLUMNS].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("The transformed targets contain infinite values.")
    if training["SMILES"].duplicated().any():
        raise ValueError("The training table contains duplicated structures.")
    if training[LOG_COLUMNS].isna().all(axis=1).any():
        raise ValueError("The training table contains a row without a target.")

    return training


def main() -> None:
    training = build_training_table()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(training):,} unique molecules -> {OUTPUT_PATH}")
    print("Available log10 targets:")
    print(training[LOG_COLUMNS].notna().sum().to_string())


if __name__ == "__main__":
    main()
