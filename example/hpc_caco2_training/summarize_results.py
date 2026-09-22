#!/usr/bin/env python3
"""Summarize and bootstrap Caco-2 test results across model strategies."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET = "Log10_Caco_Papp_AB_cm_s"
METRICS = ("R2", "RMSE", "MAE", "Pearson", "Spearman", "Kendall")
MODEL_ORDER = ("CheMeleon", "Graphormer", "ChemBERTa")
STRATEGY_ORDER = ("S1 Public3", "S2 ExpansionRX", "S3 Joint")
CONDITION_ORDER = (
    ("scaffold", 0.1),
    ("scaffold", 0.2),
    ("random", 0.1),
    ("random", 0.2),
)
RUN_PATTERN = re.compile(
    r"^(?P<model>chemeleon|graphormer|chemberta)_"
    r"(?P<strategy>S[123]_(?:public3|expansionrx|joint))"
    r"(?:_freeze(?P<freeze>\d+))?_"
    r"(?P<split>scaffold|random)_val_"
    r"(?P<val>0(?:\.\d+)?)_seed(?P<seed>\d+)$"
)
MODEL_LABELS = {
    "chemeleon": "CheMeleon",
    "graphormer": "Graphormer",
    "chemberta": "ChemBERTa",
}
STRATEGY_LABELS = {
    "S1_public3": "S1 Public3",
    "S2_expansionrx": "S2 ExpansionRX",
    "S3_joint": "S3 Joint",
}


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate all requested regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if y_true.size < 2:
        return {metric: float("nan") for metric in METRICS}

    result = {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }
    for name, function in (
        ("Pearson", pearsonr),
        ("Spearman", spearmanr),
        ("Kendall", kendalltau),
    ):
        try:
            result[name] = float(function(y_true, y_pred).statistic)
        except Exception:
            result[name] = float("nan")
    return result


def prediction_columns(frame: pd.DataFrame, target: str) -> tuple[str, str]:
    """Recognize ChemFlow's model-specific prediction column conventions."""
    true_candidates = (
        f"{target}_true",
        f"true_{target}",
        "true_value",
        "y_true",
        "y_test",
    )
    pred_candidates = (
        f"{target}_prediction",
        f"pred_{target}",
        "predicted_value",
        "y_pred",
    )
    true_column = next((name for name in true_candidates if name in frame), None)
    pred_column = next((name for name in pred_candidates if name in frame), None)
    if true_column is None or pred_column is None:
        raise ValueError(
            "Could not identify true/prediction columns. Available columns: "
            f"{list(frame.columns)}"
        )
    return true_column, pred_column


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Bootstrap matched test molecules with replacement."""
    rng = np.random.default_rng(seed)
    values = {metric: np.full(n_bootstrap, np.nan) for metric in METRICS}
    for replicate in range(n_bootstrap):
        selected = rng.integers(0, len(y_true), size=len(y_true))
        estimates = calculate_metrics(y_true[selected], y_pred[selected])
        for metric in METRICS:
            values[metric][replicate] = estimates[metric]
    return values


def discover_runs(root: Path) -> list[tuple[Path, re.Match[str]]]:
    discovered = []
    for prediction_path in sorted(root.glob("*/test_predictions.csv")):
        match = RUN_PATTERN.fullmatch(prediction_path.parent.name)
        if match is not None:
            discovered.append((prediction_path, match))
    return discovered


def analyze(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = discover_runs(args.root)
    if not runs:
        raise FileNotFoundError(
            f"No matching test_predictions.csv files were found below {args.root}."
        )

    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    bootstrap_rows: list[pd.DataFrame] = []
    for run_index, (prediction_path, match) in enumerate(runs):
        metadata = match.groupdict()
        frame = pd.read_csv(prediction_path)
        true_column, pred_column = prediction_columns(frame, args.target)
        y_true = pd.to_numeric(frame[true_column], errors="coerce").to_numpy(float)
        y_pred = pd.to_numeric(frame[pred_column], errors="coerce").to_numpy(float)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[finite], y_pred[finite]
        if y_true.size < 2:
            print(f"Skipping {prediction_path}: fewer than two finite predictions.")
            continue

        model = MODEL_LABELS[metadata["model"]]
        strategy = STRATEGY_LABELS[metadata["strategy"]]
        split = metadata["split"]
        val_fraction = float(metadata["val"])
        seed = int(metadata["seed"])
        point = calculate_metrics(y_true, y_pred)
        draws = bootstrap_metrics(
            y_true,
            y_pred,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        common = {
            "Model": model,
            "Strategy": strategy,
            "Split": split,
            "Val_fraction": val_fraction,
            "Seed": seed,
            "Freeze_epochs": (
                int(metadata["freeze"]) if metadata["freeze"] is not None else 0
            ),
            "N_test": int(y_true.size),
            "Run": prediction_path.parent.name,
            "Prediction_file": str(prediction_path),
        }
        wide = dict(common)
        for metric in METRICS:
            finite_draws = draws[metric][np.isfinite(draws[metric])]
            lower, upper = (
                np.percentile(finite_draws, [2.5, 97.5])
                if finite_draws.size
                else (np.nan, np.nan)
            )
            wide[metric] = point[metric]
            wide[f"{metric}_CI_low"] = float(lower)
            wide[f"{metric}_CI_high"] = float(upper)
            long_rows.append(
                {
                    **common,
                    "Metric": metric,
                    "Estimate": point[metric],
                    "CI_2.5%": float(lower),
                    "CI_97.5%": float(upper),
                    "N_bootstrap": int(finite_draws.size),
                }
            )
            bootstrap_rows.append(
                pd.DataFrame(
                    {
                        "Run": prediction_path.parent.name,
                        "Model": model,
                        "Strategy": strategy,
                        "Split": split,
                        "Val_fraction": val_fraction,
                        "Metric": metric,
                        "Bootstrap": np.arange(args.n_bootstrap),
                        "Value": draws[metric],
                    }
                )
            )
        wide_rows.append(wide)
        print(
            f"Loaded {model} | {strategy} | {split} {val_fraction:g} | "
            f"n={y_true.size}: {prediction_path}"
        )

    if not wide_rows:
        raise ValueError("No run contained enough valid test predictions.")
    return (
        pd.DataFrame(wide_rows),
        pd.DataFrame(long_rows),
        pd.concat(bootstrap_rows, ignore_index=True),
    )


def plot_summary(long_table: pd.DataFrame, output: Path) -> None:
    """Plot bootstrap 95% intervals for every split condition and metric."""
    colors = dict(zip(STRATEGY_ORDER, ("#2878b5", "#e07a1f", "#2a9d55")))
    offsets = dict(zip(STRATEGY_ORDER, (-0.22, 0.0, 0.22)))
    figure, axes = plt.subplots(
        len(CONDITION_ORDER),
        len(METRICS),
        figsize=(25, 14),
        constrained_layout=True,
        squeeze=False,
    )
    x = np.arange(len(MODEL_ORDER), dtype=float)
    for row_index, (split, val_fraction) in enumerate(CONDITION_ORDER):
        condition = long_table.loc[
            (long_table["Split"] == split)
            & np.isclose(long_table["Val_fraction"], val_fraction)
        ]
        for column_index, metric in enumerate(METRICS):
            axis = axes[row_index, column_index]
            for strategy in STRATEGY_ORDER:
                estimates, lower_errors, upper_errors = [], [], []
                for model in MODEL_ORDER:
                    selected = condition.loc[
                        (condition["Metric"] == metric)
                        & (condition["Strategy"] == strategy)
                        & (condition["Model"] == model)
                    ]
                    if selected.empty:
                        estimates.append(np.nan)
                        lower_errors.append(0.0)
                        upper_errors.append(0.0)
                        continue
                    row = selected.iloc[0]
                    estimate = float(row["Estimate"])
                    estimates.append(estimate)
                    lower_errors.append(max(0.0, estimate - float(row["CI_2.5%"])))
                    upper_errors.append(max(0.0, float(row["CI_97.5%"])-estimate))
                axis.errorbar(
                    x + offsets[strategy],
                    estimates,
                    yerr=np.vstack([lower_errors, upper_errors]),
                    fmt="o",
                    capsize=3,
                    color=colors[strategy],
                    label=strategy,
                )
            axis.set_xticks(x, MODEL_ORDER, rotation=25, ha="right")
            axis.grid(axis="y", linestyle=":", alpha=0.4)
            axis.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                axis.set_title(metric, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(
                    f"{split.capitalize()} split\nval={val_fraction:g}\nMetric value"
                )
            if metric in {"R2", "Pearson", "Spearman", "Kendall"}:
                axis.axhline(0.0, color="0.65", linewidth=0.8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle(
        "Caco-2 ExpansionRX independent-test performance\n"
        "points and molecule-bootstrap 95% confidence intervals",
        fontsize=17,
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/liuy48/model_training/adme/adsorption/caco2"),
    )
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be at least 1.")
    output_dir = args.output_dir or args.root / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    wide, long, bootstrap = analyze(args)
    sort_columns = ["Model", "Strategy", "Split", "Val_fraction", "Seed"]
    wide = wide.sort_values(sort_columns).reset_index(drop=True)
    long = long.sort_values(sort_columns + ["Metric"]).reset_index(drop=True)

    wide_path = output_dir / "caco2_test_metrics_table.csv"
    long_path = output_dir / "caco2_test_metrics_bootstrap_ci.csv"
    bootstrap_path = output_dir / "caco2_test_bootstrap_samples.csv.gz"
    figure_path = output_dir / "caco2_test_bootstrap_comparison.png"
    wide.to_csv(wide_path, index=False)
    long.to_csv(long_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False, compression="gzip")
    plot_summary(long, figure_path)

    print(f"Saved wide metric table: {wide_path}")
    print(f"Saved bootstrap CI table: {long_path}")
    print(f"Saved bootstrap samples: {bootstrap_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved figure: {figure_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
