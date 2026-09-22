#!/usr/bin/env python3
"""Combine curated Public3 and ExpansionRX Caco-2 training datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem


TARGET = "Log10_Caco_Papp_AB_cm_s"
OUTPUT_COLUMNS = ["Ligand ID", "SMILES", "InChIKey", TARGET, "Source"]


def _canonicalize(
    frame: pd.DataFrame,
    *,
    source: str,
    path: Path,
    require_target: bool = True,
) -> pd.DataFrame:
    required = {"Ligand ID", "SMILES", "InChIKey", TARGET}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    result = frame[list(required)].copy()
    molecules = result["SMILES"].map(
        lambda value: Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
    )
    invalid = molecules.isna()
    if invalid.any():
        raise ValueError(f"{path} contains {int(invalid.sum())} invalid SMILES.")

    result["SMILES"] = molecules.map(
        lambda molecule: Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
    )
    generated_keys = molecules.map(Chem.MolToInchiKey)
    mismatched = generated_keys.ne(result["InChIKey"])
    if mismatched.any():
        raise ValueError(
            f"{path} contains {int(mismatched.sum())} InChIKeys that do not "
            "match the canonicalized structures."
        )

    result["InChIKey"] = generated_keys
    result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")
    missing_targets = result[TARGET].isna()
    if require_target and missing_targets.any():
        print(
            f"{source}: removed {int(missing_targets.sum()):,} structures "
            f"without {TARGET}."
        )
        result = result.loc[~missing_targets].copy()
    if result["InChIKey"].duplicated().any():
        raise ValueError(f"{path} contains duplicate InChIKeys.")
    if result["Ligand ID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate Ligand IDs.")

    result["Source"] = source
    return result[OUTPUT_COLUMNS]


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    curated = repository / "dataset/curated"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public",
        type=Path,
        default=curated / "caco2_public_3source_training.csv",
    )
    parser.add_argument(
        "--expansion-train",
        type=Path,
        default=curated / "caco2_expansionrx_train.csv",
    )
    parser.add_argument(
        "--expansion-test",
        type=Path,
        default=curated / "caco2_expansionrx_test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=curated / "caco2_public3_expansionrx_train.csv",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=curated / "caco2_public3_expansionrx_overlap_audit.csv",
    )
    args = parser.parse_args()

    public = _canonicalize(
        pd.read_csv(args.public), source="Public3", path=args.public
    )
    expansion_train = _canonicalize(
        pd.read_csv(args.expansion_train),
        source="ExpansionRX_train",
        path=args.expansion_train,
    )
    expansion_test = _canonicalize(
        pd.read_csv(args.expansion_test),
        source="ExpansionRX_test",
        path=args.expansion_test,
        require_target=False,
    )

    test_keys = set(expansion_test["InChIKey"])
    leaked_public = public["InChIKey"].isin(test_keys)
    leaked_expansion = expansion_train["InChIKey"].isin(test_keys)
    if leaked_public.any() or leaked_expansion.any():
        raise ValueError(
            "Exact test-set leakage detected: "
            f"Public3={int(leaked_public.sum())}, "
            f"ExpansionRX_train={int(leaked_expansion.sum())}."
        )

    overlap = public.merge(
        expansion_train,
        on="InChIKey",
        how="inner",
        suffixes=("_Public3", "_ExpansionRX"),
    )
    if not overlap.empty:
        overlap["Difference_ExpansionRX_minus_Public3"] = (
            overlap[f"{TARGET}_ExpansionRX"] - overlap[f"{TARGET}_Public3"]
        )
        overlap["Absolute_difference"] = overlap[
            "Difference_ExpansionRX_minus_Public3"
        ].abs()
        overlap["Pair_standard_deviation"] = (
            overlap["Absolute_difference"] / (2.0**0.5)
        )
        overlap["Conflict_std_gt_0.3"] = overlap[
            "Pair_standard_deviation"
        ].gt(0.3)

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(args.audit_output, index=False)

    combined = pd.concat([public, expansion_train], ignore_index=True)
    if combined["InChIKey"].duplicated().any():
        raise ValueError(
            "Public3 and ExpansionRX training contain overlapping InChIKeys; "
            f"inspect {args.audit_output} before choosing a label policy."
        )
    if combined["Ligand ID"].duplicated().any():
        raise ValueError("The combined training dataset contains duplicate Ligand IDs.")

    combined = combined.sort_values("Ligand ID").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    print(f"Public3 training compounds: {len(public):,}")
    print(f"ExpansionRX training compounds: {len(expansion_train):,}")
    print(f"Exact cross-source overlaps: {len(overlap):,}")
    print(f"Independent test compounds: {len(expansion_test):,}")
    print(f"Combined training compounds: {len(combined):,}")
    print(f"Saved combined training data: {args.output}")
    print(f"Saved overlap audit: {args.audit_output}")


if __name__ == "__main__":
    main()
