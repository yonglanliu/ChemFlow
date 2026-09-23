#!/usr/bin/env python3
"""Create cluster-disjoint CL-3 train/validation splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import MiniBatchKMeans


TARGETS = (
    "Log10_HLM_CLint_mL_min_kg",
    "Log10_RLM_CLint_mL_min_kg",
    "Log10_MLM_CLint_mL_min_kg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("dataset/curated/clearance_chembl_biogen")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "cl3_clearance_training_multitask.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "splits",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=root / "expansionrx_clearance_test.csv",
        help="Independent CL-Test dataset to label with split='test'.",
    )
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=10000)
    return parser.parse_args()


def fingerprints(smiles: pd.Series, radius: int, n_bits: int) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=True,
    )
    matrix = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for row_index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES at row {row_index}: {value}")
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[row_index])
    return matrix


def cluster_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame[["cluster_id", "Dataset", *TARGETS]].copy()
    for target in TARGETS:
        working[f"{target}_present"] = working[target].notna().astype(int)
    working["CL-1_count"] = working["Dataset"].eq("CL-1").astype(int)
    working["CL-2_count"] = working["Dataset"].eq("CL-2").astype(int)
    aggregations: dict[str, tuple[str, str]] = {
        "molecules": ("cluster_id", "size"),
        "CL-1_count": ("CL-1_count", "sum"),
        "CL-2_count": ("CL-2_count", "sum"),
    }
    for target in TARGETS:
        aggregations[f"{target}_count"] = (f"{target}_present", "sum")
    return working.groupby("cluster_id", sort=True).agg(**aggregations).reset_index()


def choose_validation_clusters(
    statistics: pd.DataFrame,
    fraction: float,
    *,
    seed: int,
    trials: int,
) -> set[int]:
    """Search random cluster orderings for size- and task-balanced holdouts."""
    metric_columns = [
        "molecules",
        "CL-1_count",
        "CL-2_count",
        *(f"{target}_count" for target in TARGETS),
    ]
    values = statistics[metric_columns].to_numpy(dtype=float)
    totals = values.sum(axis=0)
    cluster_ids = statistics["cluster_id"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    best_score = np.inf
    best_clusters: set[int] | None = None

    for _ in range(trials):
        order = rng.permutation(len(cluster_ids))
        cumulative = np.cumsum(values[order], axis=0)
        target_rows = fraction * totals[0]
        candidate_positions = {
            max(0, int(np.searchsorted(cumulative[:, 0], target_rows)) - 1),
            min(len(order) - 1, int(np.searchsorted(cumulative[:, 0], target_rows))),
        }
        for position in candidate_positions:
            selected = cumulative[position]
            realized = selected / totals
            # Molecular count is primary; endpoint and source balance prevent a
            # sparse multitask endpoint from being concentrated in one split.
            weights = np.asarray([4.0, 1.0, 1.0, 1.5, 1.5, 1.5])
            score = float(np.sum(weights * np.square(realized - fraction)))
            if score < best_score:
                best_score = score
                best_clusters = set(cluster_ids[order[: position + 1]].tolist())

    if not best_clusters:
        raise RuntimeError("Failed to choose validation clusters.")
    return best_clusters


def split_summary(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    rows = []
    for split in ("train", "val"):
        selected = frame.loc[frame["split"].eq(split)]
        row: dict[str, object] = {
            "Requested_val_fraction": fraction,
            "Split": split,
            "Molecules": len(selected),
            "Fraction": len(selected) / len(frame),
            "Clusters": selected["cluster_id"].nunique(),
            "CL-1_molecules": int(selected["Dataset"].eq("CL-1").sum()),
            "CL-2_molecules": int(selected["Dataset"].eq("CL-2").sum()),
        }
        for target in TARGETS:
            row[f"{target}_labels"] = int(selected[target].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be at least two.")
    if args.n_bits < 1 or args.radius < 0 or args.search_trials < 1:
        raise ValueError("Fingerprint and search settings are invalid.")

    frame = pd.read_csv(args.input)
    required = {"SMILES", "InChIKey", "Dataset", *TARGETS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if frame["InChIKey"].duplicated().any():
        raise ValueError("CL-3 must contain one row per InChIKey.")

    matrix = fingerprints(frame["SMILES"], args.radius, args.n_bits)
    n_clusters = min(args.n_clusters, len(frame))
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=args.seed,
        batch_size=1024,
        max_iter=300,
        n_init=10,
        reassignment_ratio=0.01,
    )
    frame = frame.copy()
    frame["cluster_id"] = model.fit_predict(matrix).astype(int)
    statistics = cluster_statistics(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = args.output_dir / "cl3_kmeans_cluster_assignments.csv"
    frame[["Ligand_ID", "InChIKey", "cluster_id"]].to_csv(
        assignment_path, index=False
    )
    statistics.to_csv(args.output_dir / "cl3_kmeans_cluster_summary.csv", index=False)

    test_frame = pd.read_csv(args.test_input)
    test_required = {"SMILES", "InChIKey", *TARGETS}
    test_missing = sorted(test_required - set(test_frame.columns))
    if test_missing:
        raise ValueError(f"CL-Test is missing required columns: {test_missing}")
    test_frame = test_frame.copy()
    test_frame["split_val_0.1"] = "test"
    test_frame["split_val_0.2"] = "test"
    consolidated_test_path = args.output_dir / "cl_test_splits.csv"
    test_frame.to_csv(consolidated_test_path, index=False)

    summaries = []
    consolidated = frame.copy()
    for fraction in (0.1, 0.2):
        validation_clusters = choose_validation_clusters(
            statistics,
            fraction,
            seed=args.seed + int(fraction * 1000),
            trials=args.search_trials,
        )
        split_frame = frame.copy()
        split_frame["split"] = np.where(
            split_frame["cluster_id"].isin(validation_clusters), "val", "train"
        )
        consolidated[f"split_val_{fraction:.1f}"] = split_frame["split"]
        summary = split_summary(split_frame, fraction)
        summaries.append(summary)

        train_clusters = set(
            split_frame.loc[split_frame["split"].eq("train"), "cluster_id"]
        )
        val_clusters = set(
            split_frame.loc[split_frame["split"].eq("val"), "cluster_id"]
        )
        if train_clusters & val_clusters:
            raise AssertionError("A molecular cluster was assigned to both splits.")
        print(f"\nPrepared {fraction:.0%} validation assignment.")
        print(summary.to_string(index=False))

    summary_path = args.output_dir / "cl3_kmeans_split_summary.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    consolidated_paths = []
    for dataset_name, file_prefix in (
        (None, "cl3"),
        ("CL-1", "cl1"),
        ("CL-2", "cl2"),
    ):
        output = args.output_dir / f"{file_prefix}_kmeans_splits.csv"
        selected = (
            consolidated
            if dataset_name is None
            else consolidated.loc[consolidated["Dataset"].eq(dataset_name)]
        )
        selected.to_csv(output, index=False)
        consolidated_paths.append(output)
    print(f"\nSaved cluster assignments: {assignment_path}")
    print(f"Saved split summary: {summary_path}")
    for output in consolidated_paths:
        print(f"Saved consolidated split dataset: {output}")
    print(f"Saved consolidated test dataset: {consolidated_test_path}")


if __name__ == "__main__":
    main()
