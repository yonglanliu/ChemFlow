#!/usr/bin/env python3
"""Plot clearance distributions for combined and ExpansionRX datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ENDPOINTS = {
    "HLM": "Log10_HLM_CLint_mL_min_kg",
    "RLM": "Log10_RLM_CLint_mL_min_kg",
    "MLM": "Log10_MLM_CLint_mL_min_kg",
}
DATASETS = (
    "ChEMBL + Biogen",
    "ExpansionRX train",
    "ExpansionRX test",
)
COLORS = {
    "ChEMBL + Biogen": "#0072B2",
    "ExpansionRX train": "#E69F00",
    "ExpansionRX test": "#009E73",
}
LINESTYLES = {
    "ChEMBL + Biogen": "-",
    "ExpansionRX train": "--",
    "ExpansionRX test": "-.",
}


def finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.array([], dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def distribution_summary(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for endpoint, column in ENDPOINTS.items():
        for dataset in DATASETS:
            values = finite_values(frames[dataset], column)
            rows.append(
                {
                    "Endpoint": endpoint,
                    "Dataset": dataset,
                    "N": len(values),
                    "Mean": np.mean(values) if len(values) else np.nan,
                    "SD": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                    "Median": np.median(values) if len(values) else np.nan,
                    "Q1": np.percentile(values, 25) if len(values) else np.nan,
                    "Q3": np.percentile(values, 75) if len(values) else np.nan,
                    "Minimum": np.min(values) if len(values) else np.nan,
                    "Maximum": np.max(values) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def density_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(values) < 2 or np.isclose(np.std(values), 0.0):
        return np.zeros_like(grid)
    return gaussian_kde(values)(grid)


def plot_distributions(
    frames: dict[str, pd.DataFrame], output_path: Path
) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 24,
            "font.family": "DejaVu Sans",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(21, 6.8), sharey=False)

    for axis, (endpoint, column) in zip(axes, ENDPOINTS.items()):
        endpoint_values = {
            dataset: finite_values(frame, column)
            for dataset, frame in frames.items()
        }
        available = [values for values in endpoint_values.values() if len(values)]
        if not available:
            axis.text(0.5, 0.5, "No measurements", ha="center", va="center")
            continue

        pooled = np.concatenate(available)
        span = float(np.max(pooled) - np.min(pooled))
        padding = max(0.12, span * 0.06)
        grid = np.linspace(np.min(pooled) - padding, np.max(pooled) + padding, 500)

        for dataset in DATASETS:
            values = endpoint_values[dataset]
            if not len(values):
                continue
            density = density_curve(values, grid)
            color = COLORS[dataset]
            label = f"{dataset} (n={len(values):,})"
            axis.plot(
                grid,
                density,
                color=color,
                linewidth=3.0,
                linestyle=LINESTYLES[dataset],
                label=label,
                zorder=3,
            )
            axis.fill_between(grid, density, color=color, alpha=0.10, zorder=2)
            median = float(np.median(values))
            axis.axvline(
                median,
                color=color,
                linewidth=1.8,
                linestyle=LINESTYLES[dataset],
                alpha=0.9,
                zorder=4,
            )

        missing = [
            dataset for dataset in DATASETS if not len(endpoint_values[dataset])
        ]
        if missing:
            axis.text(
                0.98,
                0.95,
                "Not measured:\n" + "\n".join(missing),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=13,
                color="#555555",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "alpha": 0.9,
                },
            )

        axis.set_title(f"{endpoint} intrinsic clearance", fontweight="bold", pad=14)
        axis.set_xlabel(r"$\log_{10}$(CL$_{int}$ [mL min$^{-1}$ kg$^{-1}$])")
        axis.set_ylabel("Probability density")
        axis.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", frameon=False)

    figure.suptitle(
        "Clearance distributions across training and independent test datasets",
        fontweight="bold",
        y=1.02,
    )
    figure.text(
        0.5,
        -0.015,
        "Curves show kernel density estimates; vertical lines indicate medians.",
        ha="center",
        fontsize=14,
        color="#444444",
    )
    figure.tight_layout(w_pad=2.8)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("dataset/curated/clearance_chembl_biogen")
    parser.add_argument(
        "--combined",
        type=Path,
        default=root / "chembl_biogen_clearance_training_multitask.csv",
    )
    parser.add_argument(
        "--expansionrx-train",
        type=Path,
        default=root / "expansionrx_clearance_train.csv",
    )
    parser.add_argument(
        "--expansionrx-test",
        type=Path,
        default=root / "expansionrx_clearance_test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "clearance_distribution_comparison.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = {
        "ChEMBL + Biogen": pd.read_csv(args.combined),
        "ExpansionRX train": pd.read_csv(args.expansionrx_train),
        "ExpansionRX test": pd.read_csv(args.expansionrx_test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = distribution_summary(frames)
    summary_path = args.output.with_name("clearance_distribution_summary.csv")
    summary.to_csv(summary_path, index=False)
    plot_distributions(frames, args.output)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved figure: {args.output.resolve()}")
    print(f"Saved figure: {args.output.with_suffix('.pdf').resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
