#!/usr/bin/env python3
"""Compare molecular representations for one traditional ML model."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METRICS = ("R2", "MAE", "RMSE", "Pearson", "Spearman", "Kendall")
LOWER_IS_BETTER = {"MAE", "RMSE"}
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
SHORT_LABELS = {
    "ECFP4": "ECFP4",
    "ECFP4 + RDKit2D": "ECFP4\n+RDKit2D",
    "RDKit2D": "RDKit2D",
    "ECFP4 + RDKit2D + ErG + Avalon": "All four",
}


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
    # A structure is the most reliable cross-run identity. Molecule names and
    # row numbers are useful fallbacks, but some training exports regenerate
    # them after filtering or splitting.
    preferred_keys = ["SMILES", "Molecule Name", "test_row"]
    keys = []
    key_overlap = 0
    rejected_keys = []
    for column in preferred_keys:
        if not all(column in frame.columns for frame in frames.values()):
            continue
        # Use one stable identifier rather than a composite of every metadata
        # column. In particular, test_row can be regenerated after a feature
        # set drops invalid molecules and therefore must not invalidate an
        # otherwise valid Molecule Name or SMILES match.
        if any(frame[column].isna().any() for frame in frames.values()):
            continue
        if any(frame[column].duplicated().any() for frame in frames.values()):
            continue
        common_values = set(first_frame[column])
        for frame in frames.values():
            common_values.intersection_update(frame[column])
        overlap = len(common_values)
        if overlap == 0:
            rejected_keys.append(f"{column}: no overlap")
            continue

        # An identifier is trustworthy only when the observed target agrees
        # for every matched molecule. This detects regenerated sequential names
        # or row IDs that happen to overlap numerically across different splits.
        reference = first_frame[[column, "true_value"]].rename(
            columns={"true_value": "true__reference"}
        )
        truth_matches = True
        for frame in frames.values():
            candidate = reference.merge(
                frame[[column, "true_value"]].rename(
                    columns={"true_value": "true__candidate"}
                ),
                on=column,
                how="inner",
                validate="one_to_one",
            )
            if not np.allclose(
                candidate["true__reference"].to_numpy(dtype=float),
                candidate["true__candidate"].to_numpy(dtype=float),
                equal_nan=True,
            ):
                truth_matches = False
                break
        if not truth_matches:
            rejected_keys.append(f"{column}: matched targets differ")
            continue
        if overlap > key_overlap:
            keys = [column]
            key_overlap = overlap

    if not keys:
        lengths = {name: len(frame) for name, frame in frames.items()}
        common_identifiers = [
            column
            for column in preferred_keys
            if all(column in frame.columns for frame in frames.values())
        ]
        if common_identifiers:
            raise ValueError(
                "No unique, non-missing molecule identifier has common values "
                "across representations. This usually means the models used "
                f"different test splits. Checked: {common_identifiers}; sizes: "
                f"{lengths}; rejected: {rejected_keys}"
            )
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
    else:
        print(f"Aligning representations by {keys[0]} ({key_overlap} common rows).")

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


def summarize_paired_differences(
    task,
    model_name,
    representation_labels,
    point_estimates,
    bootstrap_values,
):
    """Compare every representation pair using matched bootstrap replicates."""
    rows = []
    for representation_a, representation_b in combinations(representation_labels, 2):
        for metric in METRICS:
            values_a = np.asarray(
                bootstrap_values[representation_a][metric], dtype=float
            )
            values_b = np.asarray(
                bootstrap_values[representation_b][metric], dtype=float
            )
            finite = np.isfinite(values_a) & np.isfinite(values_b)
            raw_differences = values_a[finite] - values_b[finite]
            if raw_differences.size == 0:
                lower = upper = probability_better = p_value = float("nan")
            else:
                improvements = (
                    -raw_differences
                    if metric in LOWER_IS_BETTER
                    else raw_differences
                )
                lower, upper = np.percentile(improvements, [2.5, 97.5])
                probability_better = float(
                    (
                        np.count_nonzero(improvements > 0)
                        + 0.5 * np.count_nonzero(improvements == 0)
                    )
                    / improvements.size
                )
                lower_tail = (np.count_nonzero(improvements <= 0) + 1) / (
                    improvements.size + 1
                )
                upper_tail = (np.count_nonzero(improvements >= 0) + 1) / (
                    improvements.size + 1
                )
                p_value = float(min(1.0, 2.0 * min(lower_tail, upper_tail)))

            estimate_a = float(point_estimates[representation_a][metric])
            estimate_b = float(point_estimates[representation_b][metric])
            raw_difference = estimate_a - estimate_b
            improvement = (
                -raw_difference if metric in LOWER_IS_BETTER else raw_difference
            )
            rows.append(
                {
                    "Task": task,
                    "Model": model_name,
                    "Metric": metric,
                    "Representation_A": representation_labels[representation_a],
                    "Representation_A_key": representation_a,
                    "Representation_B": representation_labels[representation_b],
                    "Representation_B_key": representation_b,
                    "Estimate_A": estimate_a,
                    "Estimate_B": estimate_b,
                    "Difference_A_minus_B": raw_difference,
                    "Improvement_A_over_B": improvement,
                    "Improvement_CI_2.5%": lower,
                    "Improvement_CI_97.5%": upper,
                    "Probability_A_better": probability_better,
                    "P_value_two_sided": p_value,
                    "N_bootstrap": int(raw_differences.size),
                }
            )
    return pd.DataFrame(rows)


def add_holm_adjustment(pairwise):
    """Adjust pairwise p-values within each task and metric family."""
    adjusted = pd.Series(np.nan, index=pairwise.index, dtype=float)
    for _, group in pairwise.groupby(["Task", "Metric"], sort=False):
        finite = group["P_value_two_sided"].dropna().sort_values()
        running_maximum = 0.0
        number = len(finite)
        for rank, (index, p_value) in enumerate(finite.items()):
            candidate = min(1.0, float(p_value) * (number - rank))
            running_maximum = max(running_maximum, candidate)
            adjusted.loc[index] = running_maximum
    result = pairwise.copy()
    result["P_value_Holm"] = adjusted
    result["Significant_Holm_0.05"] = result["P_value_Holm"] < 0.05
    return result


def plot_combined_summary(summary, tasks, model_name, output_path):
    """Plot grouped metric estimates for every task in one figure."""
    labels = list(REPRESENTATIONS.values())
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.75, len(labels)))
    column_count = min(3, len(tasks))
    row_count = int(np.ceil(len(tasks) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7.0 * column_count, 4.8 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    metric_positions = np.arange(len(METRICS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(labels))
    for task_index, task in enumerate(tasks):
        axis = axes.ravel()[task_index]
        for label, color, offset in zip(labels, colors, offsets):
            estimates = []
            lower_errors = []
            upper_errors = []
            for metric in METRICS:
                selection = summary.loc[
                    (summary["Task"] == task)
                    & (summary["Metric"] == metric)
                    & (summary["Representation"] == label)
                ]
                if selection.empty:
                    estimates.append(np.nan)
                    lower_errors.append(0.0)
                    upper_errors.append(0.0)
                    continue
                row = selection.iloc[0]
                estimate = float(row["Estimate"])
                lower = float(row["CI_2.5%"])
                upper = float(row["CI_97.5%"])
                estimates.append(estimate)
                lower_errors.append(max(0.0, estimate - lower))
                upper_errors.append(max(0.0, upper - estimate))
            axis.errorbar(
                metric_positions + offset,
                estimates,
                yerr=np.vstack([lower_errors, upper_errors]),
                fmt="o",
                color=color,
                capsize=3,
                markersize=6,
                linewidth=1.2,
                label=label,
            )
        axis.set_xticks(metric_positions, METRICS, rotation=25, ha="right")
        axis.set_title(str(task), fontweight="bold")
        axis.set_ylabel("Metric value")
        axis.grid(axis="both", linestyle=":", alpha=0.35)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axhline(0.0, color="0.65", linewidth=0.8)

    for axis in axes.ravel()[len(tasks) :]:
        axis.set_visible(False)

    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=min(len(labels), 4),
        frameon=False,
    )
    figure.suptitle(
        f"{model_name}: molecular-representation performance\n"
        "paired bootstrap 95% confidence intervals",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _significance_level(p_value):
    if not np.isfinite(p_value) or p_value >= 0.05:
        return 0
    if p_value < 0.001:
        return 3
    if p_value < 0.01:
        return 2
    return 1


def plot_pairwise_heatmaps(pairwise, tasks, metric, model_name, output_path):
    """Plot directional pairwise significance matrices for all tasks."""
    labels = list(REPRESENTATIONS.values())
    short_labels = [SHORT_LABELS.get(label, label).replace("\n", " ") for label in labels]
    column_count = min(3, len(tasks))
    row_count = int(np.ceil(len(tasks) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.4 * column_count, 5.0 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    colors = [
        "#9e2f4f",
        "#de7894",
        "#f4c6d2",
        "#f7f7f7",
        "#b8e3c2",
        "#58b77b",
        "#086838",
    ]
    color_map = ListedColormap(colors)
    normalizer = BoundaryNorm(np.arange(-3.5, 4.5, 1.0), color_map.N)

    for task_index, task in enumerate(tasks):
        axis = axes.ravel()[task_index]
        matrix = np.zeros((len(labels), len(labels)), dtype=float)
        annotations = np.full((len(labels), len(labels)), "NS", dtype=object)
        np.fill_diagonal(annotations, "—")
        task_rows = pairwise.loc[
            (pairwise["Task"] == task) & (pairwise["Metric"] == metric)
        ]
        for _, row in task_rows.iterrows():
            index_a = labels.index(row["Representation_A"])
            index_b = labels.index(row["Representation_B"])
            improvement = float(row["Improvement_A_over_B"])
            p_value = float(row["P_value_Holm"])
            level = _significance_level(p_value)
            direction = 0 if level == 0 else (level if improvement > 0 else -level)
            matrix[index_a, index_b] = direction
            matrix[index_b, index_a] = -direction
            label = (
                "NS"
                if level == 0
                else "p<0.001"
                if p_value < 0.001
                else "p<0.01"
                if p_value < 0.01
                else "p<0.05"
            )
            annotations[index_a, index_b] = label
            annotations[index_b, index_a] = label

        axis.imshow(matrix, cmap=color_map, norm=normalizer, aspect="equal")
        axis.set_xticks(np.arange(len(labels)), short_labels, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(labels)), short_labels)
        axis.set_title(str(task), fontweight="bold")
        for row_index in range(len(labels)):
            for column_index in range(len(labels)):
                value = matrix[row_index, column_index]
                text_color = "white" if abs(value) == 3 else "#222222"
                axis.text(
                    column_index,
                    row_index,
                    annotations[row_index, column_index],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )
        axis.set_xlabel("Column representation")
        axis.set_ylabel("Row representation")

    for axis in axes.ravel()[len(tasks) :]:
        axis.set_visible(False)

    legend = [
        Patch(facecolor="#086838", label="row better, p<0.001"),
        Patch(facecolor="#58b77b", label="row better, p<0.01"),
        Patch(facecolor="#b8e3c2", label="row better, p<0.05"),
        Patch(facecolor="#f7f7f7", edgecolor="0.7", label="not significant"),
        Patch(facecolor="#f4c6d2", label="column better, p<0.05"),
        Patch(facecolor="#de7894", label="column better, p<0.01"),
        Patch(facecolor="#9e2f4f", label="column better, p<0.001"),
    ]
    figure.legend(
        handles=legend,
        loc="outside center right",
        frameon=False,
        fontsize=8,
    )
    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
    figure.suptitle(
        f"{model_name}: pairwise representation differences ({metric}; {direction})\n"
        "Holm-adjusted paired-bootstrap significance",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def compare_task(root, output_dir, task, model_key, model_label, n_bootstrap, seed):
    frames = {}
    for representation in REPRESENTATIONS:
        path = (
            root
            / f"{task}_ml"
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
    summary.to_csv(summary_path, index=False)
    pairwise = summarize_paired_differences(
        task=task,
        model_name=model_label,
        representation_labels=REPRESENTATIONS,
        point_estimates=points,
        bootstrap_values=bootstraps,
    )
    pairwise_path = output_dir / f"{task}_{model_key}_paired_differences.csv"
    add_holm_adjustment(pairwise).to_csv(pairwise_path, index=False)

    table = summary.pivot(
        index="Representation", columns="Metric", values="Estimate"
    ).loc[list(REPRESENTATIONS.values()), list(METRICS)]
    print(f"\n{task} {model_label} ({len(y_true)} common test molecules):")
    print(table.round(4).to_string())
    print(f"Saved: {summary_path}")
    print(f"Saved: {pairwise_path}")
    return summary, pairwise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Training directory containing folders such as HLM_ml, MLM_ml, and RLM_ml.",
    )
    parser.add_argument("--model", default="lightgbm", help="Model output folder key.")
    parser.add_argument("--model-label", default="LightGBM")
    parser.add_argument("--tasks", nargs="+", default=["HLM", "MLM", "RLM"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pairwise-metric",
        choices=METRICS,
        default="RMSE",
        help="Metric shown in the pairwise significance heatmap.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be at least 1.")
    output_dir = args.output_dir or args.root / "representation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    pairwise_results = []
    for task_index, task in enumerate(args.tasks):
        summary, pairwise = compare_task(
            root=args.root,
            output_dir=output_dir,
            task=task,
            model_key=args.model,
            model_label=args.model_label,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + task_index,
        )
        summaries.append(summary)
        pairwise_results.append(pairwise)
    combined = pd.concat(summaries, ignore_index=True)
    combined_path = output_dir / f"all_tasks_{args.model}_representation_metrics.csv"
    combined.to_csv(combined_path, index=False)
    pairwise = add_holm_adjustment(pd.concat(pairwise_results, ignore_index=True))
    pairwise_path = output_dir / f"all_tasks_{args.model}_paired_differences.csv"
    pairwise.to_csv(pairwise_path, index=False)
    figure_path = output_dir / f"all_tasks_{args.model}_representation_comparison.png"
    plot_combined_summary(
        combined,
        tasks=args.tasks,
        model_name=args.model_label,
        output_path=figure_path,
    )
    heatmap_path = (
        output_dir
        / f"all_tasks_{args.model}_paired_{args.pairwise_metric.lower()}_heatmap.png"
    )
    plot_pairwise_heatmaps(
        pairwise,
        tasks=args.tasks,
        metric=args.pairwise_metric,
        model_name=args.model_label,
        output_path=heatmap_path,
    )
    print(f"\nSaved combined results: {combined_path}")
    print(f"Saved paired analysis: {pairwise_path}")
    print(f"Saved combined figure: {figure_path}")
    print(f"Saved combined figure: {figure_path.with_suffix('.pdf')}")
    print(f"Saved pairwise heatmap: {heatmap_path}")
    print(f"Saved pairwise heatmap: {heatmap_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
