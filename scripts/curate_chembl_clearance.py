#!/usr/bin/env python3
"""Curate intrinsic microsomal clearance activities exported from ChEMBL.

The ChEMBL activity export is expected to be semicolon-delimited. The raw file
is never modified. This script classifies every clearance activity, standardizes
structures to parent molecules, retains a conservative primary endpoint, and
writes accepted measurements, exclusions, assay-level estimates, aggregated
compound targets, and a compact training table.

The primary endpoint is exact intrinsic microsomal clearance in
``mL min-1 g-1``, kept separately for human, rat, and mouse. Activities whose
assay descriptions explicitly normalize clearance per gram of liver are
excluded because they cannot be treated as microsomal-protein-normalized values.
Per-kilogram scaled clearance is not mixed with this endpoint because conversion
requires species- and protocol-specific physiological scaling factors.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


DEFAULT_INPUT = Path("dataset/CHEMBL/CL.csv")
DEFAULT_OUTPUT_DIR = Path("dataset/CHEMBL/curated_clearance")
DEFAULT_MAX_STD = 0.3
TARGET = "Log10_CLint_microsome_mL_min_g"

SPECIES = {
    "Homo sapiens": "human",
    "Rattus norvegicus": "rat",
    "Mus musculus": "mouse",
}

# These are numerically equivalent: 1 mL/g == 1 uL/mg.
PER_GRAM_UNIT_FACTORS = {
    "ml.min-1.g-1": 1.0,
    "ml/min.g": 1.0,
    "ml/min/g": 1.0,
    "ul/mg/min": 1.0,
    "ul.min-1.mg-1": 1.0,
    "ul/min/mg": 1.0,
}

REQUIRED_COLUMNS = {
    "Molecule ChEMBL ID",
    "Smiles",
    "Standard Relation",
    "Standard Value",
    "Standard Units",
    "Data Validity Comment",
    "Potential Duplicate",
    "Assay ChEMBL ID",
    "Assay Description",
    "BAO Label",
    "Assay Organism",
    "Assay Tissue Name",
    "Assay Cell Type",
    "Assay Subcellular Fraction",
    "Document ChEMBL ID",
    "Source Description",
    "Document Journal",
    "Document Year",
}

AUDIT_COLUMNS = [
    "Molecule ChEMBL ID",
    "Ligand_ID",
    "Original_SMILES",
    "SMILES",
    "InChIKey",
    "Parent_InChIKey",
    "Species",
    "Assay_class",
    "Microsome_scope",
    "Measurement_context",
    "Clearance_type",
    "Normalization_basis",
    "Standard Relation",
    "Standard Value",
    "Standard Units",
    "CLint_microsome_mL_min_g",
    TARGET,
    "Assay ChEMBL ID",
    "Assay Description",
    "BAO Label",
    "Assay Organism",
    "Assay Tissue Name",
    "Assay Cell Type",
    "Assay Subcellular Fraction",
    "Data Validity Comment",
    "Potential Duplicate",
    "Document ChEMBL ID",
    "Source Description",
    "Document Journal",
    "Document Year",
    "Curation_status",
    "Exclusion_reason",
    "Extreme_value_flag",
]


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalized_token(value: object) -> str:
    return _clean_text(value).lower().replace("µ", "u").replace("μ", "u")


def _contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def classify_assay(row: pd.Series) -> str:
    """Assign an assay matrix without relying on Target Type alone."""
    description = _clean_text(row["Assay Description"])
    bao = _clean_text(row["BAO Label"])
    tissue = _clean_text(row["Assay Tissue Name"])
    cell = _clean_text(row["Assay Cell Type"])
    fraction = _clean_text(row["Assay Subcellular Fraction"])
    text = " | ".join((description, bao, tissue, cell, fraction))

    has_microsome = _contains(text, r"microsom")
    has_hepatocyte = _contains(text, r"hepatocyte|heparg|liver cells")
    has_cytosol = _contains(text, r"cytosol")
    has_s9 = _contains(text, r"(?:^|\W)s\s*9(?:\W|$)")
    has_plasma_or_blood = _contains(text, r"plasma|whole blood|\bblood\b|serum")
    organism_format = _contains(bao, r"organism-based")

    matrices = sum(
        (has_microsome, has_hepatocyte, has_cytosol, has_s9, has_plasma_or_blood)
    )
    if matrices > 1:
        return "mixed_or_ambiguous"
    if has_hepatocyte:
        return "hepatocyte"
    if has_cytosol:
        return "cytosol"
    if has_s9:
        return "s9_fraction"
    if has_plasma_or_blood:
        return "plasma_or_blood"
    if has_microsome:
        return "microsome"
    if organism_format:
        return "in_vivo_or_whole_organism"
    return "other_or_unspecified"


def classify_clearance_type(description: object) -> str:
    text = _clean_text(description)
    if _contains(text, r"unbound\s+(?:hepatic\s+)?intrinsic clearance|cl\s*int\s*,?\s*u"):
        return "unbound_intrinsic"
    if _contains(text, r"intrinsic clearance|\bcl\s*int\b"):
        return "intrinsic"
    if _contains(text, r"metabolic stability"):
        return "metabolic_stability_unspecified"
    if _contains(text, r"total(?: body)? clearance"):
        return "total_clearance"
    if _contains(text, r"clearance"):
        return "clearance_unspecified"
    return "unspecified"


def classify_microsome_scope(row: pd.Series) -> str:
    """Classify organ context; generic microsomes require manual review."""
    description = _clean_text(row["Assay Description"])
    tissue = _clean_text(row["Assay Tissue Name"])
    text = f"{description} | {tissue}"
    if _contains(
        text,
        r"kidney|renal|lung|intestinal|intestine|\bgut\b|brain|colon|jejun",
    ):
        return "non_liver"
    if _contains(text, r"\bliver\b|\bhepatic\b"):
        return "liver"
    return "unspecified"


def classify_measurement_context(description: object) -> str:
    """Separate general disappearance assays from pathway-specific kinetics."""
    text = _clean_text(description)
    if _contains(
        text,
        r"metabolite formation|glucuronidation|hydroxylation|dealkylation|"
        r"ratio of vmax|\bcyp\s*\d|cytochrome p450|ugt-mediated|"
        r"harboring cyp|in presence of cyp\w* inhibitor",
    ):
        return "enzyme_or_metabolite_specific"
    return "general_clearance"


def classify_normalization_basis(description: object) -> str:
    """Identify an explicitly stated mass denominator in an assay description."""
    text = _clean_text(description)
    if _contains(
        text,
        r"(?:\bper\s+)?\b(?:g|gram(?:s)?)\s+(?:of\s+)?liver\b",
    ):
        return "liver_mass"
    if _contains(
        text,
        r"\b(?:per\s+)?(?:(?:u|µ|μ|m)?g|(?:milli|micro)?gram(?:s)?)\s+"
        r"(?:of\s+)?(?:microsomal\s+)?protein\b",
    ):
        return "microsomal_protein"
    return "unspecified"


def standardize_structure(value: object) -> tuple[object, object, object, str]:
    """Return parent SMILES, full key, connectivity key, and an error string."""
    text = _clean_text(value)
    if not text:
        return pd.NA, pd.NA, pd.NA, "missing_smiles"
    try:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            return pd.NA, pd.NA, pd.NA, "invalid_smiles"
        molecule = rdMolStandardize.Cleanup(molecule)
        molecule = rdMolStandardize.FragmentParent(molecule)
        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        Chem.SanitizeMol(molecule)
        smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        key = Chem.MolToInchiKey(molecule)
        if not smiles or not key:
            return pd.NA, pd.NA, pd.NA, "structure_identifier_failed"
        return smiles, key, key.split("-")[0], ""
    except Exception:
        return pd.NA, pd.NA, pd.NA, "structure_standardization_failed"


def normalize_relation(value: object) -> str:
    return _clean_text(value).replace("'", "").replace('"', "")


def exclusion_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    structure_error = _clean_text(row["Structure_error"])
    if structure_error:
        reasons.append(structure_error)
    if row["Species"] not in SPECIES.values():
        reasons.append("species_not_human_rat_mouse")
    if row["Assay_class"] != "microsome":
        reasons.append(f"assay_class_{row['Assay_class']}")
    elif row["Microsome_scope"] != "liver":
        reasons.append(f"microsome_scope_{row['Microsome_scope']}")
    if row["Measurement_context"] != "general_clearance":
        reasons.append(f"measurement_context_{row['Measurement_context']}")
    if row["Clearance_type"] != "intrinsic":
        reasons.append(f"clearance_type_{row['Clearance_type']}")
    if row["Normalization_basis"] == "liver_mass":
        reasons.append("liver_mass_normalized")
    if row["Normalized_relation"] != "=":
        reasons.append("censored_or_missing_relation")
    if not np.isfinite(row["Numeric_standard_value"]):
        reasons.append("missing_or_non_numeric_value")
    elif row["Numeric_standard_value"] <= 0:
        reasons.append("nonpositive_value")
    if row["Normalized_unit"] not in PER_GRAM_UNIT_FACTORS:
        reasons.append("incompatible_or_missing_unit")
    if _clean_text(row["Data Validity Comment"]):
        reasons.append("data_validity_comment")
    duplicate = pd.to_numeric(
        pd.Series([row["Potential Duplicate"]]), errors="coerce"
    ).iloc[0]
    if np.isfinite(duplicate) and duplicate != 0:
        reasons.append("potential_duplicate")
    return reasons


def _join_unique(values: Iterable[object]) -> object:
    result: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in result:
            result.append(text)
    return " | ".join(result) if result else pd.NA


def load_and_classify(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", low_memory=False)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    frame = frame.copy()
    frame["Original_SMILES"] = frame["Smiles"]
    frame["Species"] = frame["Assay Organism"].map(SPECIES).fillna("other")
    frame["Assay_class"] = frame.apply(classify_assay, axis=1)
    frame["Microsome_scope"] = frame.apply(classify_microsome_scope, axis=1)
    frame["Measurement_context"] = frame["Assay Description"].map(
        classify_measurement_context
    )
    frame["Clearance_type"] = frame["Assay Description"].map(
        classify_clearance_type
    )
    frame["Normalization_basis"] = frame["Assay Description"].map(
        classify_normalization_basis
    )
    frame["Normalized_relation"] = frame["Standard Relation"].map(
        normalize_relation
    )
    frame["Normalized_unit"] = frame["Standard Units"].map(_normalized_token)
    frame["Numeric_standard_value"] = pd.to_numeric(
        frame["Standard Value"], errors="coerce"
    )

    RDLogger.DisableLog("rdApp.error")
    structures = frame["Smiles"].map(standardize_structure)
    RDLogger.EnableLog("rdApp.error")
    structure_frame = pd.DataFrame(
        structures.tolist(),
        columns=["SMILES", "InChIKey", "Parent_InChIKey", "Structure_error"],
        index=frame.index,
    )
    frame = pd.concat([frame, structure_frame], axis=1)

    reasons = frame.apply(exclusion_reasons, axis=1)
    frame["Exclusion_reason"] = reasons.map(lambda items: ";".join(items))
    frame["Curation_status"] = np.where(
        frame["Exclusion_reason"].eq(""), "accepted", "excluded"
    )

    factor = frame["Normalized_unit"].map(PER_GRAM_UNIT_FACTORS)
    frame["CLint_microsome_mL_min_g"] = (
        frame["Numeric_standard_value"] * factor
    )
    frame[TARGET] = np.nan
    positive = frame["CLint_microsome_mL_min_g"].gt(0)
    frame.loc[positive, TARGET] = np.log10(
        frame.loc[positive, "CLint_microsome_mL_min_g"]
    )
    frame.loc[frame["Curation_status"].ne("accepted"), TARGET] = np.nan
    # This is deliberately an audit flag, not an exclusion rule. Very low or
    # high exact values may be real, unit errors, or equality-encoded assay
    # limits and should be reviewed at the assay/source level.
    frame["Extreme_value_flag"] = np.where(
        frame["Curation_status"].eq("accepted")
        & (frame[TARGET].lt(-1.0) | frame[TARGET].gt(3.0)),
        "review_extreme_log_value",
        "",
    )

    unique_keys = sorted(frame["InChIKey"].dropna().unique())
    ligand_ids = {key: f"CF_CL_{index:06d}" for index, key in enumerate(unique_keys, 1)}
    frame["Ligand_ID"] = frame["InChIKey"].map(ligand_ids)
    return frame


def assay_level_estimates(accepted: pd.DataFrame) -> pd.DataFrame:
    keys = ["InChIKey", "Species", "Assay ChEMBL ID"]
    grouped = accepted.groupby(keys, sort=True, dropna=False)
    result = grouped[TARGET].agg(
        Assay_estimate="median",
        Assay_measurement_mean="mean",
        Assay_measurement_std="std",
        Assay_measurement_min="min",
        Assay_measurement_max="max",
        Assay_measurement_count="count",
    )
    for column in (
        "Ligand_ID",
        "SMILES",
        "Parent_InChIKey",
        "Assay Description",
        "Document ChEMBL ID",
        "Source Description",
        "Document Journal",
        "Document Year",
    ):
        result[column] = grouped[column].agg(_join_unique)
    return result.reset_index()


def compound_estimates(
    accepted: pd.DataFrame,
    assay_estimates: pd.DataFrame,
    max_std: float,
) -> pd.DataFrame:
    keys = ["InChIKey", "Species"]
    grouped = assay_estimates.groupby(keys, sort=True)
    result = grouped["Assay_estimate"].agg(
        **{
            TARGET: "median",
            f"{TARGET}_between_assay_mean": "mean",
            f"{TARGET}_between_assay_std": "std",
            f"{TARGET}_min": "min",
            f"{TARGET}_max": "max",
            "Assay_count": "count",
        }
    )
    for column in (
        "Ligand_ID",
        "SMILES",
        "Parent_InChIKey",
        "Assay ChEMBL ID",
        "Document ChEMBL ID",
        "Source Description",
        "Document Journal",
        "Document Year",
    ):
        result[column] = grouped[column].agg(_join_unique)

    raw_counts = accepted.groupby(keys).size().rename("Measurement_count")
    result = result.join(raw_counts)
    std_column = f"{TARGET}_between_assay_std"
    result["Eligible_for_pooling"] = (
        result["Assay_count"].eq(1)
        | result[std_column].isna()
        | result[std_column].le(max_std)
    )
    result["Aggregation_status"] = np.where(
        result["Eligible_for_pooling"], "pooled", "conflicting_assays"
    )
    result["CLint_microsome_mL_min_g"] = np.power(10.0, result[TARGET])
    result["Extreme_value_flag"] = np.where(
        result[TARGET].lt(-1.0) | result[TARGET].gt(3.0),
        "review_extreme_log_value",
        "",
    )
    return result.reset_index()


def exclusion_summary(classified: pd.DataFrame) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for reasons in classified.loc[
        classified["Curation_status"].eq("excluded"), "Exclusion_reason"
    ]:
        counter.update(str(reasons).split(";"))
    rows = [
        {"Exclusion_reason": reason, "Row_count": count}
        for reason, count in counter.most_common()
        if reason
    ]
    return pd.DataFrame(rows)


def assay_inventory(classified: pd.DataFrame) -> pd.DataFrame:
    """Create one row per assay so inclusion decisions can be reviewed."""
    keys = ["Assay ChEMBL ID"]
    grouped = classified.groupby(keys, sort=True, dropna=False)
    result = grouped.agg(
        Activity_count=("Molecule ChEMBL ID", "size"),
        Molecule_count=("Molecule ChEMBL ID", "nunique"),
        Accepted_activity_count=(
            "Curation_status",
            lambda values: int(values.eq("accepted").sum()),
        ),
        Assay_Description=("Assay Description", _join_unique),
        Species=("Species", _join_unique),
        Assay_class=("Assay_class", _join_unique),
        Microsome_scope=("Microsome_scope", _join_unique),
        Measurement_context=("Measurement_context", _join_unique),
        Clearance_type=("Clearance_type", _join_unique),
        Normalization_basis=("Normalization_basis", _join_unique),
        Standard_Units=("Standard Units", _join_unique),
        Standard_Relations=("Normalized_relation", _join_unique),
        Document_ChEMBL_ID=("Document ChEMBL ID", _join_unique),
        Source_Description=("Source Description", _join_unique),
        Document_Journal=("Document Journal", _join_unique),
        Document_Year=("Document Year", _join_unique),
        Exclusion_reasons=("Exclusion_reason", _join_unique),
    )
    result["Primary_endpoint_assay"] = result["Accepted_activity_count"].gt(0)
    return result.reset_index()


def multitask_table(training: pd.DataFrame) -> pd.DataFrame:
    """Pivot eligible human, rat, and mouse targets into a sparse wide table."""
    metadata = (
        training.groupby("InChIKey", sort=True)
        .agg(
            Ligand_ID=("Ligand_ID", _join_unique),
            SMILES=("SMILES", _join_unique),
            Parent_InChIKey=("Parent_InChIKey", _join_unique),
        )
        .reset_index()
    )
    values = training.pivot(index="InChIKey", columns="Species", values=TARGET)
    values = values.rename(
        columns={
            "human": "Log10_HLM_CLint_mL_min_g",
            "rat": "Log10_RLM_CLint_mL_min_g",
            "mouse": "Log10_MLM_CLint_mL_min_g",
        }
    ).reset_index()
    result = metadata.merge(values, on="InChIKey", how="left", validate="one_to_one")
    target_columns = [
        "Log10_HLM_CLint_mL_min_g",
        "Log10_RLM_CLint_mL_min_g",
        "Log10_MLM_CLint_mL_min_g",
    ]
    for column in target_columns:
        if column not in result:
            result[column] = np.nan
    return result[["Ligand_ID", "SMILES", "InChIKey", "Parent_InChIKey", *target_columns]]


def write_outputs(
    classified: pd.DataFrame,
    output_dir: Path,
    max_std: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted = classified.loc[classified["Curation_status"].eq("accepted")].copy()
    excluded = classified.loc[classified["Curation_status"].eq("excluded")].copy()
    assays = assay_level_estimates(accepted)
    compounds = compound_estimates(accepted, assays, max_std)
    training = compounds.loc[compounds["Eligible_for_pooling"]].copy()
    multitask = multitask_table(training)
    inventory = assay_inventory(classified)

    classified[AUDIT_COLUMNS].to_csv(
        output_dir / "chembl36_clearance_classified.csv", index=False
    )
    accepted[AUDIT_COLUMNS].to_csv(
        output_dir / "chembl36_microsomal_clearance_measurements.csv", index=False
    )
    excluded[AUDIT_COLUMNS].to_csv(
        output_dir / "chembl36_microsomal_clearance_excluded.csv", index=False
    )
    excluded.loc[excluded["Assay_class"].eq("microsome"), AUDIT_COLUMNS].to_csv(
        output_dir / "chembl36_microsomal_clearance_manual_review.csv",
        index=False,
    )
    assays.to_csv(
        output_dir / "chembl36_microsomal_clearance_assay_estimates.csv",
        index=False,
    )
    compounds.to_csv(
        output_dir / "chembl36_microsomal_clearance_curated.csv", index=False
    )
    training_columns = [
        "Ligand_ID",
        "SMILES",
        "InChIKey",
        "Parent_InChIKey",
        "Species",
        TARGET,
        "CLint_microsome_mL_min_g",
        "Measurement_count",
        "Assay_count",
        f"{TARGET}_between_assay_std",
        "Assay ChEMBL ID",
        "Document ChEMBL ID",
        "Source Description",
        "Extreme_value_flag",
    ]
    training[training_columns].to_csv(
        output_dir / "chembl36_microsomal_clearance_training.csv", index=False
    )
    multitask.to_csv(
        output_dir / "chembl36_microsomal_clearance_multitask.csv", index=False
    )
    inventory.to_csv(
        output_dir / "chembl36_clearance_assay_inventory.csv", index=False
    )
    exclusion_summary(classified).to_csv(
        output_dir / "chembl36_microsomal_clearance_exclusion_summary.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {"Stage": "Raw ChEMBL clearance activities", "Rows": len(classified)},
            {"Stage": "Accepted exact measurements", "Rows": len(accepted)},
            {"Stage": "Excluded activities", "Rows": len(excluded)},
            {"Stage": "Assay-level estimates", "Rows": len(assays)},
            {"Stage": "Compound-species estimates", "Rows": len(compounds)},
            {"Stage": "Eligible training rows", "Rows": len(training)},
            {"Stage": "Unique multitask molecules", "Rows": len(multitask)},
        ]
    )
    summary.to_csv(output_dir / "chembl36_microsomal_clearance_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("\nEligible training rows by species:")
    print(training["Species"].value_counts().to_string())
    print(f"\nOutputs: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-std", type=float, default=DEFAULT_MAX_STD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_std < 0:
        raise ValueError("--max-std must be non-negative.")
    classified = load_and_classify(args.input)
    write_outputs(classified, args.output_dir, args.max_std)


if __name__ == "__main__":
    main()
