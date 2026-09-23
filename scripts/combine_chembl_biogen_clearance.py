#!/usr/bin/env python3
"""Combine curated ChEMBL and Biogen microsomal-clearance data safely.

ChEMBL values retained by the upstream curation are recorded in mL/min/g and
interpreted as microsomal-protein-normalized values; assays explicitly reported
per gram of liver are excluded upstream. Biogen values are scaled per kilogram
body weight. This script applies explicit, configurable species scaling factors
to the ChEMBL values before pooling. It also standardizes every structure,
aggregates sources by InChIKey and species, rejects cross-source disagreements
above a fixed log10 standard-deviation threshold, and prevents ExpansionRX train
or test compounds from entering the combined pretraining dataset.

ExpansionRX is written to separate train and test files and is never pooled
with ChEMBL or Biogen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from curate_chembl_clearance import standardize_structure


TARGET = "Log10_CLint_mL_min_kg"
SPECIES_COLUMNS = {
    "human": "Log10_HLM_CLint_mL_min_kg",
    "rat": "Log10_RLM_CLint_mL_min_kg",
    "mouse": "Log10_MLM_CLint_mL_min_kg",
}


def molecular_weight(smiles: object) -> float:
    """Calculate average molecular weight from a standardized parent SMILES."""
    molecule = Chem.MolFromSmiles(str(smiles))
    return float(Descriptors.MolWt(molecule)) if molecule is not None else np.nan


def _join_unique(values: pd.Series) -> str:
    items = sorted({str(value) for value in values.dropna() if str(value).strip()})
    return ";".join(items)


def _first_unique(values: pd.Series) -> str:
    """Choose one deterministic representation for identifier-equivalent rows."""
    items = sorted({str(value) for value in values.dropna() if str(value).strip()})
    return items[0] if items else ""


def standardize_smiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Return parent structures using the same rules as ChEMBL curation."""
    frame = frame.copy()
    structures = frame["SMILES"].map(standardize_structure)
    standardized = pd.DataFrame(
        structures.tolist(),
        columns=["Canonical_SMILES", "InChIKey", "Parent_InChIKey", "Structure_error"],
        index=frame.index,
    )
    frame = pd.concat([frame, standardized], axis=1)
    frame["Original_SMILES"] = frame["SMILES"]
    frame["SMILES"] = frame["Canonical_SMILES"]
    return frame.drop(columns="Canonical_SMILES")


def prepare_chembl(path: Path, scaling: dict[str, float]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "Ligand_ID",
        "SMILES",
        "InChIKey",
        "Parent_InChIKey",
        "Species",
        "Log10_CLint_microsome_mL_min_g",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ChEMBL file is missing columns: {missing}")

    frame = frame.loc[frame["Species"].isin(scaling)].copy()
    frame["Scaling_factor_g_microsomal_protein_per_kg"] = frame["Species"].map(
        scaling
    )
    frame[TARGET] = pd.to_numeric(
        frame["Log10_CLint_microsome_mL_min_g"], errors="coerce"
    ) + np.log10(frame["Scaling_factor_g_microsomal_protein_per_kg"])
    frame["Source"] = "ChEMBL36"
    frame["Source_record_ID"] = frame["Assay ChEMBL ID"]
    frame["Original_target_name"] = "Log10_CLint_microsome_mL_min_g"
    frame["Original_target_value"] = frame[
        "Log10_CLint_microsome_mL_min_g"
    ]
    frame["Scaling_note"] = (
        "ChEMBL mL/min/g value interpreted as microsomal-protein-normalized "
        "CLint and scaled to mL/min/kg using the species factor in this row"
    )
    return frame[
        [
            "Ligand_ID",
            "SMILES",
            "InChIKey",
            "Parent_InChIKey",
            "Species",
            TARGET,
            "Source",
            "Source_record_ID",
            "Original_target_name",
            "Original_target_value",
            "Scaling_factor_g_microsomal_protein_per_kg",
            "Scaling_note",
        ]
    ]


def prepare_biogen(path: Path) -> pd.DataFrame:
    frame = standardize_smiles(pd.read_csv(path))
    if frame["Structure_error"].ne("").any():
        errors = int(frame["Structure_error"].ne("").sum())
        print(f"Warning: excluding {errors} invalid Biogen structures.")
    frame = frame.loc[frame["Structure_error"].eq("")].copy()

    rows: list[pd.DataFrame] = []
    for species, column in (
        ("human", "HLM_CLint (mL/min/kg)"),
        ("rat", "RLM_CLint (mL/min/kg)"),
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        selected = frame.loc[values.gt(0)].copy()
        selected[TARGET] = np.log10(values.loc[selected.index])
        selected["Species"] = species
        selected["Source"] = "Biogen_ADME_Fang_2023"
        selected["Source_record_ID"] = selected["Internal ID"].astype(str)
        selected["Original_target_name"] = column
        selected["Original_target_value"] = values.loc[selected.index]
        selected["Scaling_factor_g_microsomal_protein_per_kg"] = np.nan
        selected["Scaling_note"] = "Already reported in mL/min/kg"
        selected["Ligand_ID"] = pd.NA
        rows.append(selected)

    result = pd.concat(rows, ignore_index=True)
    return result[
        [
            "Ligand_ID",
            "SMILES",
            "InChIKey",
            "Parent_InChIKey",
            "Species",
            TARGET,
            "Source",
            "Source_record_ID",
            "Original_target_name",
            "Original_target_value",
            "Scaling_factor_g_microsomal_protein_per_kg",
            "Scaling_note",
        ]
    ]


def assign_ligand_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve ChEMBL IDs and assign noncolliding IDs to Biogen-only molecules."""
    frame = frame.copy()
    existing = (
        frame.dropna(subset=["Ligand_ID"])
        .drop_duplicates("InChIKey")
        .set_index("InChIKey")["Ligand_ID"]
        .to_dict()
    )
    numeric_ids = pd.Series(existing.values(), dtype="string").str.extract(
        r"CF_CL_(\d+)", expand=False
    )
    next_id = int(pd.to_numeric(numeric_ids, errors="coerce").max() or 0) + 1
    for key in sorted(set(frame["InChIKey"].dropna()) - set(existing)):
        existing[key] = f"CF_CL_{next_id:06d}"
        next_id += 1
    frame["Ligand_ID"] = frame["InChIKey"].map(existing)
    return frame


def aggregate_sources(frame: pd.DataFrame, max_std: float) -> pd.DataFrame:
    keys = ["InChIKey", "Species"]
    grouped = frame.groupby(keys, sort=True)
    result = grouped[TARGET].agg(
        **{
            TARGET: "median",
            f"{TARGET}_source_mean": "mean",
            f"{TARGET}_source_std": "std",
            f"{TARGET}_source_min": "min",
            f"{TARGET}_source_max": "max",
            "Source_value_count": "count",
        }
    )
    for column in (
        "Ligand_ID",
        "Parent_InChIKey",
        "Source",
        "Source_record_ID",
    ):
        result[column] = grouped[column].agg(_join_unique)
    result["SMILES"] = grouped["SMILES"].agg(_first_unique)
    result["Source_count"] = grouped["Source"].nunique()
    std_column = f"{TARGET}_source_std"
    result["Eligible_for_training"] = (
        result["Source_count"].eq(1)
        | result[std_column].isna()
        | result[std_column].le(max_std)
    )
    result["Aggregation_status"] = np.where(
        result["Eligible_for_training"], "pooled", "conflicting_sources"
    )
    result["CLint_mL_min_kg"] = np.power(10.0, result[TARGET])
    return result.reset_index()


def prepare_expansionrx(
    clearance_path: Path,
    membership_path: Path,
    split: str,
) -> pd.DataFrame:
    """Recover all clearance endpoints while preserving the published split."""
    clearance = pd.read_csv(clearance_path)
    membership = pd.read_csv(membership_path, usecols=["Molecule Name"])
    if membership["Molecule Name"].duplicated().any():
        raise ValueError(f"Duplicate molecule names in split membership: {membership_path}")
    frame = clearance.loc[
        clearance["Molecule Name"].isin(set(membership["Molecule Name"]))
    ].copy()
    missing_names = set(membership["Molecule Name"]) - set(frame["Molecule Name"])
    if missing_names:
        print(
            f"Note: {len(missing_names)} {split} molecules have no HLM, RLM, or MLM "
            "clearance measurement and are absent from the clearance output."
        )
    frame = standardize_smiles(frame)
    frame = frame.loc[frame["Structure_error"].eq("")].copy()
    frame["MolecularWeight"] = frame["SMILES"].map(molecular_weight)
    frame["Ligand_ID"] = frame["Molecule Name"]
    frame["Source"] = "ExpansionRX"
    frame["Dataset_split"] = split
    target_columns = []
    for short_name, source_column in (
        ("HLM", "HLM_CLint (mL/min/kg)"),
        ("RLM", "RLM_CLint (mL/min/kg)"),
        ("MLM", "MLM_CLint (mL/min/kg)"),
    ):
        values = pd.to_numeric(frame[source_column], errors="coerce")
        target_column = f"Log10_{short_name}_CLint_mL_min_kg"
        target_columns.append(target_column)
        frame[target_column] = np.nan
        positive = values.gt(0)
        frame.loc[positive, target_column] = np.log10(values.loc[positive])
    no_clearance_label = frame[target_columns].isna().all(axis=1)
    if no_clearance_label.any():
        print(
            f"Excluding {int(no_clearance_label.sum())} {split} molecules with no "
            "positive HLM, RLM, or MLM clearance label."
        )
        frame = frame.loc[~no_clearance_label].copy()
    return frame[
        [
            "Ligand_ID",
            "SMILES",
            "InChIKey",
            "Parent_InChIKey",
            "Source",
            "Dataset_split",
            "MolecularWeight",
            "Log10_HLM_CLint_mL_min_kg",
            "Log10_RLM_CLint_mL_min_kg",
            "Log10_MLM_CLint_mL_min_kg",
        ]
    ]


def make_multitask(training: pd.DataFrame) -> pd.DataFrame:
    metadata = (
        training.groupby("InChIKey", sort=True)
        .agg(
            Ligand_ID=("Ligand_ID", _join_unique),
            SMILES=("SMILES", _first_unique),
            Parent_InChIKey=("Parent_InChIKey", _join_unique),
            Source=("Source", _join_unique),
            MolecularWeight=("MolecularWeight", "first"),
        )
        .reset_index()
    )
    values = training.pivot(index="InChIKey", columns="Species", values=TARGET)
    values = values.rename(columns=SPECIES_COLUMNS).reset_index()
    result = metadata.merge(values, on="InChIKey", how="left", validate="one_to_one")
    for column in SPECIES_COLUMNS.values():
        if column not in result:
            result[column] = np.nan
    return result[
        [
            "Ligand_ID",
            "SMILES",
            "InChIKey",
            "Parent_InChIKey",
            "Source",
            "MolecularWeight",
            *SPECIES_COLUMNS.values(),
        ]
    ]


def make_cl3(cl1: pd.DataFrame, cl2: pd.DataFrame) -> pd.DataFrame:
    """Combine CL-1 and CL-2 while retaining dataset provenance."""
    columns = [
        "Ligand_ID",
        "SMILES",
        "InChIKey",
        "Parent_InChIKey",
        "Source",
        "MolecularWeight",
        *SPECIES_COLUMNS.values(),
    ]
    cl1_part = cl1[columns].copy()
    cl1_part.insert(0, "Dataset", "CL-1")
    cl2_part = cl2[columns].copy()
    cl2_part.insert(0, "Dataset", "CL-2")
    result = pd.concat([cl1_part, cl2_part], ignore_index=True)
    duplicated = result["InChIKey"].duplicated(keep=False)
    if duplicated.any():
        keys = sorted(result.loc[duplicated, "InChIKey"].unique())
        raise ValueError(
            "CL-1 and CL-2 must be structure-disjoint before constructing CL-3; "
            f"found {len(keys)} overlapping InChIKeys."
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chembl",
        type=Path,
        default=Path(
            "dataset/CHEMBL/curated_clearance/"
            "chembl36_microsomal_clearance_training.csv"
        ),
    )
    parser.add_argument(
        "--biogen",
        type=Path,
        default=Path("dataset/ADMET/grouped/clearance/Biogen_clearance.csv"),
    )
    parser.add_argument(
        "--expansionrx-all-clearance",
        type=Path,
        default=Path(
            "dataset/ADMET/grouped/clearance/ExpansionRX_raw_clearance.csv"
        ),
    )
    parser.add_argument(
        "--expansionrx-train-membership",
        type=Path,
        default=Path(
            "dataset/ADMET/ExpansionRX/expansionrx_train_df.csv"
        ),
    )
    parser.add_argument(
        "--expansionrx-test-membership",
        type=Path,
        default=Path(
            "dataset/ADMET/ExpansionRX/expansionrx_test_df.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/curated/clearance_chembl_biogen"),
    )
    parser.add_argument("--max-source-std", type=float, default=0.3)
    parser.add_argument("--minimum-mw", type=float, default=150.0)
    parser.add_argument("--maximum-mw", type=float, default=700.0)
    parser.add_argument("--human-scaling-factor", type=float, default=0.856)
    parser.add_argument("--rat-scaling-factor", type=float, default=1.8)
    parser.add_argument("--mouse-scaling-factor", type=float, default=3.9375)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_source_std < 0:
        raise ValueError("--max-source-std cannot be negative.")
    if args.minimum_mw < 0 or args.maximum_mw <= args.minimum_mw:
        raise ValueError("The molecular-weight interval is invalid.")
    scaling = {
        "human": args.human_scaling_factor,
        "rat": args.rat_scaling_factor,
        "mouse": args.mouse_scaling_factor,
    }
    if any(value <= 0 for value in scaling.values()):
        raise ValueError("All species scaling factors must be positive.")

    chembl = prepare_chembl(args.chembl, scaling)
    biogen = prepare_biogen(args.biogen)
    source_rows = assign_ligand_ids(pd.concat([chembl, biogen], ignore_index=True))
    aggregated = aggregate_sources(source_rows, args.max_source_std)
    aggregated["MolecularWeight"] = aggregated["SMILES"].map(molecular_weight)
    aggregated["Eligible_molecular_weight"] = aggregated["MolecularWeight"].between(
        args.minimum_mw,
        args.maximum_mw,
        inclusive="both",
    )

    expansion_train_all = prepare_expansionrx(
        args.expansionrx_all_clearance,
        args.expansionrx_train_membership,
        "train",
    )
    expansion_test_all = prepare_expansionrx(
        args.expansionrx_all_clearance,
        args.expansionrx_test_membership,
        "test",
    )
    expansion_train_all["Eligible_molecular_weight"] = expansion_train_all[
        "MolecularWeight"
    ].between(args.minimum_mw, args.maximum_mw, inclusive="both")
    expansion_test_all["Eligible_molecular_weight"] = expansion_test_all[
        "MolecularWeight"
    ].between(args.minimum_mw, args.maximum_mw, inclusive="both")
    expansion_train = expansion_train_all.loc[
        expansion_train_all["Eligible_molecular_weight"]
    ].drop(columns="Eligible_molecular_weight")
    expansion_test = expansion_test_all.loc[
        expansion_test_all["Eligible_molecular_weight"]
    ].drop(columns="Eligible_molecular_weight")
    expansion_keys = set(expansion_train["InChIKey"]) | set(expansion_test["InChIKey"])
    overlap = aggregated.loc[aggregated["InChIKey"].isin(expansion_keys)].copy()
    molecular_weight_excluded = aggregated.loc[
        ~aggregated["Eligible_molecular_weight"]
    ].copy()
    training = aggregated.loc[
        aggregated["Eligible_for_training"]
        & aggregated["Eligible_molecular_weight"]
        & ~aggregated["InChIKey"].isin(expansion_keys)
    ].copy()
    multitask = make_multitask(training)
    cl3 = make_cl3(multitask, expansion_train)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows.to_csv(args.output_dir / "chembl_biogen_source_values.csv", index=False)
    aggregated.to_csv(args.output_dir / "chembl_biogen_aggregated_audit.csv", index=False)
    molecular_weight_excluded.to_csv(
        args.output_dir / "chembl_biogen_molecular_weight_excluded.csv", index=False
    )
    aggregated.loc[aggregated["Aggregation_status"].eq("conflicting_sources")].to_csv(
        args.output_dir / "chembl_biogen_conflicting_sources.csv", index=False
    )
    overlap.to_csv(args.output_dir / "expansionrx_overlap_removed.csv", index=False)
    training.to_csv(args.output_dir / "chembl_biogen_clearance_training_long.csv", index=False)
    multitask.to_csv(
        args.output_dir / "chembl_biogen_clearance_training_multitask.csv", index=False
    )
    expansion_train.to_csv(args.output_dir / "expansionrx_clearance_train.csv", index=False)
    expansion_test.to_csv(args.output_dir / "expansionrx_clearance_test.csv", index=False)
    cl3.to_csv(args.output_dir / "cl3_clearance_training_multitask.csv", index=False)
    pd.concat(
        [
            expansion_train_all.loc[~expansion_train_all["Eligible_molecular_weight"]],
            expansion_test_all.loc[~expansion_test_all["Eligible_molecular_weight"]],
        ],
        ignore_index=True,
    ).to_csv(
        args.output_dir / "expansionrx_molecular_weight_excluded.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "Species": species,
                "Scaling_factor_g_microsomal_protein_per_kg": factor,
                "Interpretation": (
                    "Multiply ChEMBL mL/min/g values interpreted as "
                    "microsomal-protein-normalized clearance by this factor "
                    "to obtain mL/min/kg; explicitly liver-mass-normalized "
                    "assays were excluded upstream"
                ),
            }
            for species, factor in scaling.items()
        ]
    ).to_csv(args.output_dir / "scaling_assumptions.csv", index=False)

    summary = pd.DataFrame(
        [
            {"Stage": "ChEMBL source values", "Rows": len(chembl)},
            {"Stage": "Biogen source values", "Rows": len(biogen)},
            {"Stage": "Compound-species aggregates", "Rows": len(aggregated)},
            {
                "Stage": "Cross-source conflicts excluded",
                "Rows": int((~aggregated["Eligible_for_training"]).sum()),
            },
            {
                "Stage": "CL-1 molecular-weight targets excluded",
                "Rows": len(molecular_weight_excluded),
            },
            {
                "Stage": "CL-1 unique molecules excluded by molecular weight",
                "Rows": molecular_weight_excluded["InChIKey"].nunique(),
            },
            {"Stage": "ExpansionRX overlaps removed", "Rows": len(overlap)},
            {"Stage": "Combined training targets", "Rows": len(training)},
            {"Stage": "Combined unique molecules", "Rows": len(multitask)},
            {"Stage": "ExpansionRX train molecules", "Rows": len(expansion_train)},
            {"Stage": "ExpansionRX test molecules", "Rows": len(expansion_test)},
            {"Stage": "CL-3 unique training molecules", "Rows": len(cl3)},
        ]
    )
    summary.to_csv(args.output_dir / "curation_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nCombined targets by species:")
    print(training["Species"].value_counts().to_string())
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
