#!/usr/bin/env python3
"""Compare molecular representations for one traditional ML model."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METRICS = ("R2", "MAE", "RMSE", "Pearson", "Spearman", "Kendall")
REPRESENTATIONS = OrderedDict(
    [
        ("ecfp4", "ECFP4"),
        ("ecfp4_rdkit2d", "ECFP4 + RDKit2D"),
        ("rdkit2d", "RDKit2D"),
        (
            "ecfp4_rdkit2d_erg_avalon",
            "ECFP4 + RDKit2D + ErG + Avalon",
        ),
    ]
)


def calculate_metrics(y_true, y_pred):
    """Calculate regression and rank-correlation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 2:
        raise ValueError("At least two finite observations are required.")

    correlations = {}
    for name, function in (
        ("Pearson", pearsonr),
        ("Spearman", spearmanr),
        ("Kendall", kendalltau),
    ):
        try:
            correlations[name] = float(function(y_true, y_pred).statistic)
        except Exception:
            correlations[name] = np.nan

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        **correlations,
    }


def load_prediction_file(path):
    """Load current ChemFlow output, with aliases for older column names."""
    frame = pd.read_csv(path)
    aliases = {
        "y_true": "true_value",
        "y_test": "true_value",
        "y_pred": "predicted_value",
    }
    frame = frame.rename(
        columns={old: new for old, new in aliases.items() if new not in frame.columns}
    )
    required = {"true_value", "predicted_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def align_representations(frames):
    """Inner-align representations so every metric uses the same compounds."""
    first_frame = next(iter(frames.values()))
    preferred_keys = ["Molecule Name", "SMILES", "test_row"]
    keys = [
        column
        for column in preferred_keys
        if all(column in frame.columns for frame in frames.values())
    ]
    if not keys:
        lengths = {name: len(frame) for name, frame in frames.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "Prediction files have different row counts and no common molecule "
                f"identifier columns: {lengths}"
            )
        keys = ["_comparison_row"]
        frames = {
            name: frame.assign(_comparison_row=np.arange(len(frame)))
            for name, frame in frames.items()
        }
        first_frame = next(iter(frames.values()))

    if any(frame.duplicated(keys).any() for frame in frames.values()):
        raise ValueError(f"Molecule identifier columns are not unique: {keys}")

    metadata_columns = keys + [
        column
        for column in ("Molecule Name", "SMILES", "test_row")
        if column not in keys and column in first_frame.columns
    ]
    aligned = first_frame[metadata_columns].copy()
    initial_sizes = {name: len(frame) for name, frame in frames.items()}

    for representation, frame in frames.items():
        selected = frame[keys + ["true_value", "predicted_value"]].rename(
            columns={
                "true_value": f"true__{representation}",
                "predicted_value": f"pred__{representation}",
            }
        )
        aligned = aligned.merge(selected, on=keys, how="inner", validate="one_to_one")

    if aligned.empty:
        raise ValueError("No common test molecules were found across representations.")

    truth_columns = [f"true__{name}" for name in frames]
    reference_truth = aligned[truth_columns[0]].to_numpy(dtype=float)
    for column in truth_columns[1:]:
        candidate = aligned[column].to_numpy(dtype=float)
        if not np.allclose(reference_truth, candidate, equal_nan=True):
            raise ValueError(f"True values differ across representations: {column}")

    if any(size != len(aligned) for size in initial_sizes.values()):
        print(
            "Warning: comparison uses the common test-set intersection: "
            f"{len(aligned)} rows from original sizes {initial_sizes}."
        )
    return aligned


def paired_bootstrap(aligned, representations, n_bootstrap, seed):
    """Use identical resampled molecules for every representation."""
    y_true = aligned[f"true__{representations[0]}"].to_numpy(dtype=float)
    predictions = {
        name: aligned[f"pred__{name}"].to_numpy(dtype=float)
        for name in representations
    }
    finite = np.isfinite(y_true)
    for values in predictions.values():
        finite &= np.isfinite(values)
    y_true = y_true[finite]
    predictions = {name: values[finite] for name, values in predictions.items()}
    if len(y_true) < 2:
        raise ValueError("Not enough common finite observations for bootstrapping.")

    point_estimates = {
        name: calculate_metrics(y_true, values)
        for name, values in predictions.items()
    }
    bootstrap_values = {
        name: {metric: [] for metric in METRICS} for name in representations
    }
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        for name, values in predictions.items():
            estimates = calculate_metrics(y_true[indices], values[indices])
            for metric in METRICS:
                bootstrap_values[name][metric].append(estimates[metric])

    return y_true, point_estimates, bootstrap_values


def summarize_bootstrap(
    task,
    model_name,
    representation_labels,
    n_samples,
    point_estimates,
    bootstrap_values,
):
    rows = []
    for representation, label in representation_labels.items():
        for metric in METRICS:
            values = np.asarray(
                bootstrap_values[representation][metric], dtype=float
            )
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "Task": task,
                    "Model": model_name,
                    "Representation": label,
                    "Representation_key": representation,
                    "Metric": metric,
                    "Estimate": point_estimates[representation][metric],
                    "CI_2.5%": np.percentile(values, 2.5),
                    "CI_97.5%": np.percentile(values, 97.5),
                    "N_test": n_samples,
                    "N_bootstrap": len(values),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary, task, model_name, output_path):
    """Plot each metric separately because errors and correlations differ."""
    labels = list(dict.fromkeys(summary["Representation"]))
    short_labels = {
        "ECFP4": "ECFP4",
        "ECFP4 + RDKit2D": "ECFP4\n+RDKit2D",
        "RDKit2D": "RDKit2D",
        "ECFP4 + RDKit2D + ErG + Avalon": "All four",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, metric in zip(axes.ravel(), METRICS):
        metric_data = summary.loc[summary["Metric"] == metric].set_index(
            "Representation"
        ).loc[labels]
        estimates = metric_data["Estimate"].to_numpy(dtype=float)
        lower = metric_data["CI_2.5%"].to_numpy(dtype=float)
        upper = metric_data["CI_97.5%"].to_numpy(dtype=float)
        x = np.arange(len(labels))
        axis.errorbar(
            x,
            estimates,
            yerr=np.vstack([estimates - lower, upper - estimates]),
            fmt="o",
            capsize=4,
            markersize=7,
            linewidth=1.4,
        )
        axis.set_xticks(x, [short_labels.get(label, label) for label in labels])
        axis.set_title(metric)
        axis.grid(axis="y", linestyle=":", alpha=0.4)
        axis.spines[["top", "right"]].set_visible(False)
        if metric in {"MAE", "RMSE"}:
            axis.set_ylabel("Error (lower is better)")
        else:
            axis.set_ylabel("Score (higher is better)")

    fig.suptitle(
        f"{task} {model_name}: molecular-representation comparison\n"
        f"paired bootstrap 95% confidence intervals",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compare_task(root, output_dir, task, model_key, model_label, n_bootstrap, seed):
    frames = {}
    for representation in REPRESENTATIONS:
        path = (
            root
            / f"ml_{task}"
            / representation
            / model_key
            / f"{model_key}_test_predictions.csv"
        )
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        frames[representation] = load_prediction_file(path)
        print(f"Loaded {task} {representation}: {path} ({len(frames[representation])} rows)")

    aligned = align_representations(frames)
    y_true, points, bootstraps = paired_bootstrap(
        aligned=aligned,
        representations=list(REPRESENTATIONS),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary = summarize_bootstrap(
        task=task,
        model_name=model_label,
        representation_labels=REPRESENTATIONS,
        n_samples=len(y_true),
        point_estimates=points,
        bootstrap_values=bootstraps,
    )
    summary_path = output_dir / f"{task}_{model_key}_representation_metrics.csv"
    figure_path = output_dir / f"{task}_{model_key}_representation_comparison.png"
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, task, model_label, figure_path)

    table = summary.pivot(
        index="Representation", columns="Metric", values="Estimate"
    ).loc[list(REPRESENTATIONS.values()), list(METRICS)]
    print(f"\n{task} {model_label} ({len(y_true)} common test molecules):")
    print(table.round(4).to_string())
    print(f"Saved: {summary_path}")
    print(f"Saved: {figure_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/liuy48/model_training/adme/clearance"),
        help="Training directory containing ml_HLM, ml_MLM, and ml_RLM.",
    )
    parser.add_argument("--model", default="lightgbm", help="Model output folder key.")
    parser.add_argument("--model-label", default="LightGBM")
    parser.add_argument("--tasks", nargs="+", default=["HLM", "MLM", "RLM"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.root / "representation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for task_index, task in enumerate(args.tasks):
        summaries.append(
            compare_task(
                root=args.root,
                output_dir=output_dir,
                task=task,
                model_key=args.model,
                model_label=args.model_label,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + task_index,
            )
        )
    combined_path = output_dir / f"all_tasks_{args.model}_representation_metrics.csv"
    pd.concat(summaries, ignore_index=True).to_csv(combined_path, index=False)
    print(f"\nSaved combined results: {combined_path}")


if __name__ == "__main__":
    main()
