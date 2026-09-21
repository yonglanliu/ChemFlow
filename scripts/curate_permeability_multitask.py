#!/usr/bin/env python3
"""Build a model-ready, structure-level permeability multitask dataset.

The source assays measure different biological or artificial-membrane systems,
so they remain separate targets. ExpansionRX Caco-2 Papp A->B and the public
TDC Caco2_Wang endpoint are deliberately pooled into one literature-augmented
Caco-2 Papp head, with source provenance retained. Caco-2 efflux ratio remains
a separate companion head. Permeability coefficients reported in units of
10^-6 cm/s are converted to log10(cm/s), while dimensionless efflux ratios are
converted to log10(ratio). Sources that already provide standardized log-scale
responses are retained on that scale. Censored, zero, and nonnumeric
measurements are excluded from regression targets rather than treated as exact
observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem


REGRESSION_TARGETS = [
    "Log10_MDR1_MDCK_ER",
    "Log10_Caco_Papp_AB_cm_s",
    "Log10_Caco_ER",
    "Log10_PAMPA_BBB_cm_s",
    "Log10_PAMPA_pH5_5_cm_s",
    "Log10_PAMPA_pH7_4_cm_s",
]
CLASSIFICATION_TARGETS = ["PAMPA_NCATS_high_permeability"]
TARGET_COLUMNS = REGRESSION_TARGETS + CLASSIFICATION_TARGETS
CACO2_TARGETS = ["Log10_Caco_Papp_AB_cm_s", "Log10_Caco_ER"]


def _canonical_smiles(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    molecule = Chem.MolFromSmiles(str(value).strip())
    if molecule is None:
        return np.nan
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _inchi_key(canonical_smiles: object) -> str | float:
    """Generate a full standard InChIKey from a canonical SMILES string."""
    if pd.isna(canonical_smiles):
        return np.nan
    molecule = Chem.MolFromSmiles(str(canonical_smiles))
    if molecule is None:
        return np.nan
    value = Chem.MolToInchiKey(molecule)
    return value if value else np.nan


def _structure_columns(values: pd.Series) -> dict[str, pd.Series]:
    canonical = values.map(_canonical_smiles)
    return {
        "SMILES": canonical,
        "InChIKey": canonical.map(_inchi_key),
    }


def _first_sorted(values: pd.Series) -> str | float:
    unique = sorted({str(value) for value in values.dropna()})
    return unique[0] if unique else np.nan


def _positive_numeric(values: pd.Series) -> pd.Series:
    """Return exact positive measurements; '<', '>', and text become missing."""
    text = values.astype("string").str.strip()
    exact = ~text.str.match(r"^[<>]", na=False)
    numeric = pd.to_numeric(text.where(exact), errors="coerce")
    # Materialize nullable values as ordinary NaN before applying log10. This
    # avoids NumPy evaluating masked zero values and emitting divide warnings.
    numeric = pd.Series(
        numeric.to_numpy(dtype=float, na_value=np.nan), index=values.index
    )
    return numeric.where(numeric > 0)


def _log_permeability(values: pd.Series) -> pd.Series:
    # Source values are coefficients in 10^-6 cm/s.
    return np.log10(_positive_numeric(values) * 1e-6)


def _log_ratio(values: pd.Series) -> pd.Series:
    return np.log10(_positive_numeric(values))


def _join_unique(values: pd.Series) -> str | float:
    unique: list[str] = []
    for value in values.dropna():
        for item in str(value).split(" | "):
            item = item.strip()
            if item and item not in unique:
                unique.append(item)
    return " | ".join(unique) if unique else np.nan


def _empty_targets(length: int) -> dict[str, np.ndarray]:
    return {column: np.full(length, np.nan) for column in TARGET_COLUMNS}


def _biogen(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    structures = _structure_columns(source["SMILES"])
    targets = _empty_targets(len(source))
    targets["Log10_MDR1_MDCK_ER"] = _log_ratio(source["MDR1_MDCK_ER"])
    return pd.DataFrame(
        {
            "Molecule Name": source["Internal ID"],
            "Source ID": source["Vendor ID"].astype("string"),
            **structures,
            "source": source["source"],
            "resource": "Biogen_ADME_Fang_2023",
            "source_split": np.nan,
            **targets,
        }
    )


def _expansionrx(path: Path, source_name: str) -> pd.DataFrame:
    source = pd.read_csv(path)
    structures = _structure_columns(source["SMILES"])
    targets = _empty_targets(len(source))
    targets["Log10_Caco_Papp_AB_cm_s"] = _log_permeability(
        source["Caco_Papp_AB (10^-6 cm/s)"]
    )
    targets["Log10_Caco_ER"] = _log_ratio(source["Caco_ER"])
    return pd.DataFrame(
        {
            "Molecule Name": source["Molecule Name"],
            "Source ID": source["Molecule Name"],
            **structures,
            "source": source_name,
            "resource": "OpenADMET_ExpansionRX",
            "source_split": source["dataset_split"],
            **targets,
        }
    )


def _ncats(
    path: Path,
    measurement_column: str,
    target_column: str,
    source_name: str,
) -> pd.DataFrame:
    source = pd.read_csv(path)
    smiles_column = (
        "SMILES" if "SMILES" in source else "PUBCHEM_EXT_DATASOURCE_SMILES"
    )
    structures = _structure_columns(source[smiles_column])
    targets = _empty_targets(len(source))
    targets[target_column] = _log_permeability(source[measurement_column])
    return pd.DataFrame(
        {
            "Molecule Name": source["PUBCHEM_CID"].map(lambda value: f"CID:{value}"),
            "Source ID": source["PUBCHEM_SID"].map(lambda value: f"SID:{value}"),
            **structures,
            "source": source_name,
            "resource": "ADME_NCATS",
            "source_split": np.nan,
            **targets,
        }
    )


def _tdc_caco2(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    structures = _structure_columns(source["SMILES"])
    targets = _empty_targets(len(source))
    # TDC Caco2_Wang Y is already log10 apparent permeability in cm/s. It is
    # pooled with ExpansionRX Papp A->B for the planned public-data comparison,
    # while the source columns preserve its heterogeneous literature origin.
    targets["Log10_Caco_Papp_AB_cm_s"] = pd.to_numeric(
        source["Y"], errors="coerce"
    )
    return pd.DataFrame(
        {
            "Molecule Name": pd.NA,
            "Source ID": [f"TDC_Caco2_Wang:{index}" for index in source.index],
            **structures,
            "source": "TDC_Caco2_Wang",
            "resource": "TDC_Caco2_Wang",
            "source_split": np.nan,
            **targets,
        }
    )


def _tdc_pampa(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    structures = _structure_columns(source["SMILES"])
    labels = pd.to_numeric(source["Y"], errors="coerce")
    invalid = labels.notna() & ~labels.isin([0, 1])
    if invalid.any():
        raise ValueError(f"Non-binary PAMPA_NCATS labels in {path}")
    targets = _empty_targets(len(source))
    targets["PAMPA_NCATS_high_permeability"] = labels
    return pd.DataFrame(
        {
            "Molecule Name": pd.NA,
            "Source ID": [f"TDC_PAMPA_NCATS:{index}" for index in source.index],
            **structures,
            "source": "TDC_PAMPA_NCATS",
            "resource": "TDC_PAMPA_NCATS",
            "source_split": np.nan,
            **targets,
        }
    )


def curate(source_dir: Path) -> pd.DataFrame:
    frames = [
        _biogen(source_dir / "Biogen_permeability.csv"),
        _expansionrx(
            source_dir / "ExpansionRX_train_permeability.csv",
            "ExpansionRX_train",
        ),
        _expansionrx(
            source_dir / "ExpansionRX_test_permeability.csv",
            "ExpansionRX_test",
        ),
        _ncats(
            source_dir / "PAMPA-BBB_NCATS.csv",
            "PAMPA-BBB (x 10^-6 cm/s)",
            "Log10_PAMPA_BBB_cm_s",
            "NCATS_PAMPA_BBB",
        ),
        _ncats(
            source_dir / "PAMPA5.5_NCATS.csv",
            "PAMPA5.5  (x 10-6 cm/sec)",
            "Log10_PAMPA_pH5_5_cm_s",
            "NCATS_PAMPA_pH5_5",
        ),
        _ncats(
            source_dir / "PAMPA7.4_NCATS.csv",
            "PAMPA7.4 (x 10-6 cm/sec)",
            "Log10_PAMPA_pH7_4_cm_s",
            "NCATS_PAMPA_pH7_4",
        ),
        _tdc_caco2(source_dir / "TDC/Caco2_Wang.csv"),
        _tdc_pampa(source_dir / "TDC/PAMPA_NCATS.csv"),
    ]
    working = pd.concat(frames, ignore_index=True)
    working = working.loc[
        working["SMILES"].notna()
        & working["InChIKey"].notna()
        & working[TARGET_COLUMNS].notna().any(axis=1)
    ].copy()

    # Do not silently average assay-specific ExpansionRX observations with
    # public literature observations if an overlapping structure appears in a
    # future source update. There are currently no such overlaps.
    caco2_labeled = working.loc[
        working["Log10_Caco_Papp_AB_cm_s"].notna(),
        ["InChIKey", "resource"],
    ].drop_duplicates()
    cross_resource = caco2_labeled.groupby("InChIKey")["resource"].nunique()
    if (cross_resource > 1).any():
        examples = cross_resource[cross_resource > 1].index[:5].tolist()
        raise ValueError(
            "Caco-2 Papp structures occur in both ExpansionRX and public "
            "sources; resolve them without cross-assay averaging. Examples: "
            f"{examples}"
        )

    grouped = working.groupby("InChIKey", sort=False, dropna=False)
    curated = grouped[TARGET_COLUMNS].mean()
    curated.insert(0, "SMILES", grouped["SMILES"].apply(_first_sorted))
    curated.insert(0, "source_split", grouped["source_split"].apply(_join_unique))
    curated.insert(0, "source", grouped["source"].apply(_join_unique))
    curated.insert(0, "Source ID", grouped["Source ID"].apply(_join_unique))
    curated.insert(
        0, "Molecule Name", grouped["Molecule Name"].apply(_join_unique)
    )
    curated = curated.reset_index()
    curated = curated[
        [
            "Molecule Name",
            "Source ID",
            "SMILES",
            "InChIKey",
            "source",
            "source_split",
            *TARGET_COLUMNS,
        ]
    ]

    if curated[["SMILES", "InChIKey"]].isna().any().any():
        raise AssertionError("Output must contain valid canonical SMILES and InChIKeys.")
    if curated["InChIKey"].duplicated().any():
        raise AssertionError("Output must contain one row per full InChIKey.")
    values = curated[TARGET_COLUMNS].stack().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("All retained target values must be finite.")
    for column in CLASSIFICATION_TARGETS:
        labels = curated[column].dropna()
        if not labels.isin([0, 1]).all():
            raise AssertionError(
                f"Conflicting duplicate or non-binary labels found in {column}."
            )
    return curated


def _assign_caco2_ligand_ids(
    frame: pd.DataFrame,
    reference_path: Path,
) -> tuple[pd.DataFrame, int]:
    """Reuse public Caco-2 IDs and allocate noncolliding IDs to new structures."""
    if not reference_path.is_file():
        raise FileNotFoundError(
            "The authoritative public Caco-2 ligand-ID table was not found: "
            f"{reference_path}. Run scripts/curate_public_caco2.py first."
        )
    reference = pd.read_csv(reference_path, usecols=["Ligand ID", "InChIKey"])
    if reference[["Ligand ID", "InChIKey"]].isna().any().any():
        raise ValueError(f"Missing ligand IDs or InChIKeys in {reference_path}")
    if reference["Ligand ID"].duplicated().any():
        raise ValueError(f"Duplicate ligand IDs in {reference_path}")
    if reference["InChIKey"].duplicated().any():
        raise ValueError(f"Duplicate InChIKeys in {reference_path}")

    pattern = re.compile(r"^CF_caco2_(\d+)$")
    numbers = reference["Ligand ID"].map(
        lambda value: int(match.group(1))
        if (match := pattern.fullmatch(str(value)))
        else np.nan
    )
    if numbers.isna().any():
        invalid = reference.loc[numbers.isna(), "Ligand ID"].head(5).tolist()
        raise ValueError(f"Invalid public Caco-2 ligand IDs: {invalid}")

    identifier_by_key = dict(zip(reference["InChIKey"], reference["Ligand ID"]))
    assigned = frame["InChIKey"].map(identifier_by_key)
    reused_count = int(assigned.notna().sum())
    next_number = int(numbers.max()) + 1 if len(numbers) else 1
    for index in assigned[assigned.isna()].index:
        assigned.loc[index] = f"CF_caco2_{next_number}"
        next_number += 1

    result = frame.copy()
    result.insert(0, "Ligand ID", assigned)
    if result["Ligand ID"].duplicated().any():
        raise AssertionError("A Caco-2 ligand ID maps to multiple structures.")
    shared = result["InChIKey"].isin(identifier_by_key)
    expected = result.loc[shared, "InChIKey"].map(identifier_by_key)
    if not expected.equals(result.loc[shared, "Ligand ID"]):
        raise AssertionError("A shared Caco-2 structure did not reuse its public ID.")
    return result, reused_count


def _build_caco2_ligand_registry(
    public_reference_path: Path,
    caco2: pd.DataFrame,
) -> pd.DataFrame:
    """Create one authoritative one-to-one Ligand ID/InChIKey registry."""
    public = pd.read_csv(
        public_reference_path,
        usecols=["Ligand ID", "InChIKey", "SMILES"],
    )
    combined = pd.concat(
        [public, caco2[["Ligand ID", "InChIKey", "SMILES"]]],
        ignore_index=True,
    ).drop_duplicates()
    key_id_counts = combined.groupby("InChIKey")["Ligand ID"].nunique()
    id_key_counts = combined.groupby("Ligand ID")["InChIKey"].nunique()
    if (key_id_counts > 1).any():
        raise AssertionError("A Caco-2 structure maps to multiple ligand IDs.")
    if (id_key_counts > 1).any():
        raise AssertionError("A Caco-2 ligand ID maps to multiple structures.")
    registry = combined.drop_duplicates("InChIKey").copy()
    registry["_number"] = registry["Ligand ID"].str.extract(
        r"^CF_caco2_(\d+)$", expand=False
    ).astype(int)
    registry = registry.sort_values("_number").drop(columns="_number")
    return registry.reset_index(drop=True)


def curate_caco2(
    source_dir: Path,
    ligand_reference: Path,
) -> tuple[pd.DataFrame, int]:
    """Build the focused ExpansionRX Caco-2 Papp/ER table.

    TDC Caco2_Wang is deliberately excluded because all of its unique
    structures are already represented in the separately curated public
    three-source Caco-2 table.
    """
    working = pd.concat(
        [
            _expansionrx(
                source_dir / "ExpansionRX_train_permeability.csv",
                "ExpansionRX_train",
            ),
            _expansionrx(
                source_dir / "ExpansionRX_test_permeability.csv",
                "ExpansionRX_test",
            ),
        ],
        ignore_index=True,
    )
    working = working.loc[
        working["SMILES"].notna()
        & working["InChIKey"].notna()
        & working[CACO2_TARGETS].notna().any(axis=1)
    ].copy()
    grouped = working.groupby("InChIKey", sort=False, dropna=False)
    caco2 = grouped[CACO2_TARGETS].mean()
    caco2.insert(0, "SMILES", grouped["SMILES"].apply(_first_sorted))
    caco2.insert(0, "source_split", grouped["source_split"].apply(_join_unique))
    caco2.insert(0, "source", grouped["source"].apply(_join_unique))
    caco2.insert(0, "Source ID", grouped["Source ID"].apply(_join_unique))
    caco2.insert(0, "Molecule Name", grouped["Molecule Name"].apply(_join_unique))
    caco2 = caco2.reset_index()
    # Retain names supplied by the source datasets for traceability. Public
    # structures reuse the IDs in the three-source table; only structures not
    # found there receive new IDs above the public table's largest identifier.
    caco2 = caco2.rename(columns={"Molecule Name": "Source Molecule Name"})
    caco2 = caco2.sort_values("InChIKey", kind="stable").reset_index(drop=True)
    caco2, reused_count = _assign_caco2_ligand_ids(caco2, ligand_reference)
    metadata = [
        "Ligand ID",
        "Source Molecule Name",
        "Source ID",
        "SMILES",
        "InChIKey",
        "source",
        "source_split",
    ]
    caco2 = caco2[metadata + CACO2_TARGETS]
    if caco2[["SMILES", "InChIKey"]].isna().any().any():
        raise AssertionError(
            "Caco-2 output must contain valid canonical SMILES and InChIKeys."
        )
    if caco2["InChIKey"].duplicated().any():
        raise AssertionError("Caco-2 output must contain unique full InChIKeys.")
    if caco2["Ligand ID"].isna().any() or caco2["Ligand ID"].duplicated().any():
        raise AssertionError("Caco-2 output must contain unique ligand IDs.")
    values = caco2[CACO2_TARGETS].stack().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("All retained Caco-2 values must be finite.")
    return caco2, reused_count


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repository / "dataset/ADMET/grouped/permeability",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "dataset/curated/permeability_multitask.csv",
    )
    parser.add_argument(
        "--caco2-output",
        type=Path,
        default=repository / "dataset/curated/caco2_papp_er.csv",
    )
    parser.add_argument(
        "--caco2-train-output",
        type=Path,
        default=repository / "dataset/curated/caco2_expansionrx_train.csv",
        help="Model-ready ExpansionRX training partition to write.",
    )
    parser.add_argument(
        "--caco2-test-output",
        type=Path,
        default=repository / "dataset/curated/caco2_expansionrx_test.csv",
        help="Independent model-ready ExpansionRX test partition to write.",
    )
    parser.add_argument(
        "--caco2-ligand-reference",
        type=Path,
        default=repository / "dataset/curated/caco2_public_3source.csv",
        help=(
            "Authoritative Ligand ID/InChIKey mapping. Generate it with "
            "scripts/curate_public_caco2.py before running this script."
        ),
    )
    parser.add_argument(
        "--caco2-ligand-registry",
        type=Path,
        default=repository / "dataset/curated/caco2_ligand_registry.csv",
        help="Combined one-to-one Ligand ID/InChIKey registry to write.",
    )
    args = parser.parse_args()

    curated = curate(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(args.output, index=False)
    caco2, reused_count = curate_caco2(
        args.source_dir,
        args.caco2_ligand_reference,
    )
    args.caco2_output.parent.mkdir(parents=True, exist_ok=True)
    caco2.to_csv(args.caco2_output, index=False)
    model_columns = ["Ligand ID", "SMILES", "InChIKey", *CACO2_TARGETS]
    caco2_train = caco2.loc[
        caco2["source_split"].eq("train"), model_columns
    ].copy()
    caco2_test = caco2.loc[
        caco2["source_split"].eq("test"), model_columns
    ].copy()
    if len(caco2_train) + len(caco2_test) != len(caco2):
        raise AssertionError("Every ExpansionRX row must belong to train or test.")
    args.caco2_train_output.parent.mkdir(parents=True, exist_ok=True)
    args.caco2_test_output.parent.mkdir(parents=True, exist_ok=True)
    caco2_train.to_csv(args.caco2_train_output, index=False)
    caco2_test.to_csv(args.caco2_test_output, index=False)
    registry = _build_caco2_ligand_registry(
        args.caco2_ligand_reference,
        caco2,
    )
    args.caco2_ligand_registry.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.caco2_ligand_registry, index=False)

    print(f"Saved {len(curated):,} unique structures to {args.output}")
    for column in TARGET_COLUMNS:
        kind = "classification" if column in CLASSIFICATION_TARGETS else "regression"
        print(f"  {column} [{kind}]: {curated[column].notna().sum():,} labels")
    print(f"Saved {len(caco2):,} Caco-2 structures to {args.caco2_output}")
    print(f"  Reused public Caco-2 ligand IDs: {reused_count:,}")
    print(f"  Assigned IDs to additional structures: {len(caco2) - reused_count:,}")
    print(f"  ExpansionRX training rows: {len(caco2_train):,} -> {args.caco2_train_output}")
    print(f"  ExpansionRX test rows: {len(caco2_test):,} -> {args.caco2_test_output}")
    print(
        f"  Ligand registry: {args.caco2_ligand_registry} "
        f"({len(registry):,} unique structures)"
    )
    for column in CACO2_TARGETS:
        print(f"  {column}: {caco2[column].notna().sum():,} labels")


if __name__ == "__main__":
    main()
