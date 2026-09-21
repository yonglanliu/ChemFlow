#!/usr/bin/env python3
"""Curate and merge three public Caco-2 apparent-permeability datasets.

The inputs are the official supplementary workbooks for Wang et al. (2016),
Wang and Chen (2020), and Wang et al. (2020). Structures are converted to
canonical isomeric SMILES and full standard InChIKeys before aggregation.

The first two papers report log10(Papp in cm/s). Wang et al. (2020) report
log10(Papp x 10^6); those values are shifted by -6 to obtain log10(cm/s).
Conflicting measurements are retained and summarized rather than silently
discarded. ``Eligible_for_pooling`` identifies structures whose measurement
standard deviation is at most the configurable threshold (default 0.3 log10
units), or structures having only one measurement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


TARGET = "Log10_Caco_Papp_AB_cm_s"
DEFAULT_MAX_STD = 0.3


def _canonical_smiles(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    molecule = Chem.MolFromSmiles(str(value).strip())
    if molecule is None:
        return np.nan
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _inchi_key(canonical_smiles: object) -> str | float:
    if pd.isna(canonical_smiles):
        return np.nan
    molecule = Chem.MolFromSmiles(str(canonical_smiles))
    if molecule is None:
        return np.nan
    key = Chem.MolToInchiKey(molecule)
    return key if key else np.nan


def _join_unique(values: pd.Series) -> str | float:
    unique: list[str] = []
    for value in values.dropna():
        for item in str(value).split(" | "):
            item = item.strip()
            if item and item not in unique:
                unique.append(item)
    return " | ".join(unique) if unique else np.nan


def _first_sorted(values: pd.Series) -> str | float:
    unique = sorted({str(value) for value in values.dropna()})
    return unique[0] if unique else np.nan


def _standardize(
    frame: pd.DataFrame,
    *,
    smiles: pd.Series,
    values: pd.Series,
    names: pd.Series,
    identifiers: pd.Series,
    source: str,
    subset: pd.Series,
) -> pd.DataFrame:
    canonical = smiles.map(_canonical_smiles)
    result = pd.DataFrame(
        {
            "Source Molecule Name": names.astype("string"),
            "Source ID": identifiers.astype("string"),
            "SMILES": canonical,
            "InChIKey": canonical.map(_inchi_key),
            "source": source,
            "source_subset": subset.astype("string"),
            TARGET: pd.to_numeric(values, errors="coerce"),
        },
        index=frame.index,
    )
    finite = np.isfinite(result[TARGET].to_numpy(dtype=float, na_value=np.nan))
    return result.loc[
        result["SMILES"].notna() & result["InChIKey"].notna() & finite
    ].copy()


def load_wang_2016(path: Path) -> pd.DataFrame:
    modeling = pd.read_excel(path, sheet_name="SI1")
    modeling_subset = modeling["Dataset"].map(
        {"Tr": "model_training", "Te": "model_test"}
    )
    modeling = _standardize(
        modeling,
        smiles=modeling["smi"],
        values=modeling["logPapp"],
        names=modeling["name"],
        identifiers=modeling["NO"],
        source="Wang_2016",
        subset=modeling_subset,
    )

    validation = pd.read_excel(path, sheet_name="SI5")
    validation = _standardize(
        validation,
        smiles=validation["SMILES"],
        values=validation["value"],
        names=validation["Name"],
        identifiers=validation["X..CAS"],
        source="Wang_2016",
        subset=pd.Series("external_validation", index=validation.index),
    )
    return pd.concat([modeling, validation], ignore_index=True)


def load_wang_chen_2020(path: Path) -> pd.DataFrame:
    source = pd.read_excel(path, sheet_name="Sheet1")
    return _standardize(
        source,
        smiles=source["SMILES"],
        values=source["logPapp"],
        names=pd.Series(pd.NA, index=source.index),
        identifiers=source["id or name"],
        source="Wang_Chen_2020",
        subset=pd.Series("modeling", index=source.index),
    )


def load_wang_2020(path: Path) -> pd.DataFrame:
    try:
        source = pd.read_excel(path, sheet_name="Supplementary table-LogPapp")
    except ImportError as error:
        raise ImportError(
            "Reading the original Wang 2020 .xls workbook requires xlrd. "
            "Install it with `python -m pip install xlrd`."
        ) from error

    raw_papp = pd.to_numeric(source["Caco-2Papp(× 10x6cm/s)"], errors="coerce")
    reported_log = pd.to_numeric(source["LogPapp Value"], errors="coerce")
    comparable = raw_papp.gt(0) & reported_log.notna()
    if comparable.any() and not np.allclose(
        np.log10(raw_papp[comparable]),
        reported_log[comparable],
        atol=5e-5,
        rtol=0,
    ):
        raise ValueError("Wang 2020 Papp and LogPapp columns are inconsistent.")

    # The reported value is log10(Papp x 10^6); shift it to log10(cm/s).
    return _standardize(
        source,
        smiles=source["SMILES"],
        values=reported_log - 6.0,
        names=source["Molecule Name"],
        identifiers=source["Molecule ID"],
        source="Wang_2020",
        subset=source["Data split"],
    )


def aggregate_source(measurements: pd.DataFrame) -> pd.DataFrame:
    grouped = measurements.groupby("InChIKey", sort=True)
    result = grouped[TARGET].agg(["mean", "std", "min", "max", "count"])
    result = result.rename(
        columns={
            "mean": TARGET,
            "std": f"{TARGET}_measurement_std",
            "min": f"{TARGET}_min",
            "max": f"{TARGET}_max",
            "count": "Measurement_count",
        }
    )
    result.insert(0, "SMILES", grouped["SMILES"].apply(_first_sorted))
    result.insert(0, "source_subset", grouped["source_subset"].apply(_join_unique))
    result.insert(0, "source", grouped["source"].apply(_join_unique))
    result.insert(0, "Source ID", grouped["Source ID"].apply(_join_unique))
    result.insert(
        0,
        "Source Molecule Name",
        grouped["Source Molecule Name"].apply(_join_unique),
    )
    return result.reset_index()


def merge_sources(
    raw_measurements: pd.DataFrame,
    source_tables: list[pd.DataFrame],
    max_std: float,
) -> pd.DataFrame:
    # Average each paper first, then give every paper equal weight for compounds
    # present in multiple collections.
    source_estimates = pd.concat(source_tables, ignore_index=True)
    grouped = source_estimates.groupby("InChIKey", sort=True)
    merged = grouped[TARGET].agg(["mean", "std", "min", "max", "count"])
    merged = merged.rename(
        columns={
            "mean": TARGET,
            "std": f"{TARGET}_between_source_std",
            "min": f"{TARGET}_source_min",
            "max": f"{TARGET}_source_max",
            "count": "Source_count",
        }
    )
    merged.insert(0, "SMILES", grouped["SMILES"].apply(_first_sorted))
    merged.insert(0, "source_subset", grouped["source_subset"].apply(_join_unique))
    merged.insert(0, "source", grouped["source"].apply(_join_unique))
    merged.insert(0, "Source ID", grouped["Source ID"].apply(_join_unique))
    merged.insert(
        0,
        "Source Molecule Name",
        grouped["Source Molecule Name"].apply(_join_unique),
    )

    raw_grouped = raw_measurements.groupby("InChIKey", sort=True)[TARGET]
    merged["Measurement_count"] = raw_grouped.count()
    merged[f"{TARGET}_measurement_std"] = raw_grouped.std()
    merged[f"{TARGET}_measurement_min"] = raw_grouped.min()
    merged[f"{TARGET}_measurement_max"] = raw_grouped.max()
    variation = merged[f"{TARGET}_measurement_std"]
    merged["Eligible_for_pooling"] = variation.isna() | variation.le(max_std)

    merged = merged.reset_index()
    merged.insert(
        0,
        "Ligand ID",
        [f"CF_caco2_{index}" for index in range(1, len(merged) + 1)],
    )
    return merged


def validate(frame: pd.DataFrame, *, require_names: bool = False) -> None:
    if frame[["SMILES", "InChIKey", TARGET]].isna().any().any():
        raise AssertionError("Curated output contains a missing structure or target.")
    if frame["InChIKey"].duplicated().any():
        raise AssertionError("Curated output contains duplicate full InChIKeys.")
    generated = frame["SMILES"].map(
        lambda smiles: Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    )
    if not generated.equals(frame["InChIKey"]):
        raise AssertionError("An InChIKey does not match its canonical SMILES.")
    if require_names:
        expected = [f"CF_caco2_{index}" for index in range(1, len(frame) + 1)]
        if frame["Ligand ID"].tolist() != expected:
            raise AssertionError("ChemFlow Caco-2 identifiers are not sequential.")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    default_source = (
        repository
        / "dataset/ADMET/grouped/permeability/public_caco2/original"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument(
        "--source-output-dir",
        type=Path,
        default=default_source.parent,
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=repository / "dataset/curated/caco2_public_3source.csv",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=(
            repository / "dataset/curated/caco2_public_3source_training.csv"
        ),
        help="Training table after excluding measurements above --max-std.",
    )
    parser.add_argument("--max-std", type=float, default=DEFAULT_MAX_STD)
    args = parser.parse_args()
    if args.max_std < 0:
        raise ValueError("--max-std must be nonnegative.")

    raw_by_source = {
        "Wang_2016": load_wang_2016(
            args.source_dir / "Wang_2016_Caco2_supporting.xlsx"
        ),
        "Wang_Chen_2020": load_wang_chen_2020(
            args.source_dir / "Wang_Chen_2020_Caco2_supporting.xlsx"
        ),
        "Wang_2020": load_wang_2020(
            args.source_dir / "Wang_2020_Caco2_supporting.xls"
        ),
    }
    source_tables = {
        name: aggregate_source(values) for name, values in raw_by_source.items()
    }
    for table in source_tables.values():
        validate(table)

    args.source_output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in source_tables.items():
        output = args.source_output_dir / f"{name}_caco2_curated.csv"
        table.to_csv(output, index=False)
        print(
            f"Saved {len(table):,} unique structures to {output} "
            f"from {len(raw_by_source[name]):,} valid measurements"
        )

    raw = pd.concat(raw_by_source.values(), ignore_index=True)
    merged = merge_sources(raw, list(source_tables.values()), args.max_std)
    validate(merged, require_names=True)
    args.merged_output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.merged_output, index=False)
    training_columns = ["Ligand ID", "SMILES", "InChIKey", TARGET]
    training = merged.loc[
        merged["Eligible_for_pooling"],
        training_columns,
    ].copy()
    validate(training)
    args.training_output.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(args.training_output, index=False)
    overlaps = int((merged["Source_count"] > 1).sum())
    conflicts = int((~merged["Eligible_for_pooling"]).sum())
    print(f"Saved {len(merged):,} unique structures to {args.merged_output}")
    print(f"  Structures represented by multiple papers: {overlaps:,}")
    print(
        f"  Structures with measurement SD > {args.max_std:g}: "
        f"{conflicts:,} (flagged, not removed)"
    )
    print(
        f"Saved {len(training):,} training structures to {args.training_output} "
        f"after excluding {conflicts:,} high-variability structures"
    )


if __name__ == "__main__":
    main()
