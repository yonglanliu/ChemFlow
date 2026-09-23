#!/usr/bin/env python3
"""Plot molecular-weight distributions for the curated clearance datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import gaussian_kde


DATASET_ORDER = ("CL-1 retained", "CL-1 removed", "CL-2", "CL-Test")
COLORS = {
    "CL-1 retained": "#0072B2",
    "CL-1 removed": "#CC79A7",
    "CL-2": "#E69F00",
    "CL-Test": "#009E73",
}
LINESTYLES = {
    "CL-1 retained": "-",
    "CL-1 removed": ":",
    "CL-2": "--",
    "CL-Test": "-.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("dataset/curated/clearance_chembl_biogen")
    parser.add_argument(
        "--cl1",
        type=Path,
        default=root / "chembl_biogen_clearance_training_multitask.csv",
    )
    parser.add_argument(
        "--cl1-removed",
        type=Path,
        default=root / "chembl_biogen_molecular_weight_excluded.csv",
    )
    parser.add_argument(
        "--cl2",
        type=Path,
        default=root / "expansionrx_clearance_train.csv",
    )
    parser.add_argument(
        "--cl-test",
        type=Path,
        default=root / "expansionrx_clearance_test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "clearance_molecular_weight_distribution.png",
    )
    parser.add_argument("--cutoff", type=float, default=700.0)
    return parser.parse_args()


def molecular_weights(path: Path) -> np.ndarray:
    frame = pd.read_csv(path).drop_duplicates("InChIKey")
    values: list[float] = []
    for smiles in frame["SMILES"]:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is not None:
            values.append(float(Descriptors.MolWt(molecule)))
    return np.asarray(values, dtype=float)


def summarize(values_by_dataset: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for dataset, values in values_by_dataset.items():
        rows.append(
            {
                "Dataset": dataset,
                "N": len(values),
                "Mean_MW": np.mean(values),
                "SD_MW": np.std(values, ddof=1),
                "Median_MW": np.median(values),
                "Q1_MW": np.percentile(values, 25),
                "Q3_MW": np.percentile(values, 75),
                "Minimum_MW": np.min(values),
                "Maximum_MW": np.max(values),
                "N_600_to_700": np.sum((values >= 600) & (values <= 700)),
                "N_over_700": np.sum(values > 700),
                "Percent_over_700": 100 * np.mean(values > 700),
                "N_over_1000": np.sum(values > 1000),
            }
        )
    return pd.DataFrame(rows)


def create_figure(
    values_by_dataset: dict[str, np.ndarray], output: Path, cutoff: float
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 20,
            "axes.linewidth": 1.8,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
        }
    )
    figure, (density_axis, tail_axis) = plt.subplots(
        1,
        2,
        figsize=(17, 7.5),
        gridspec_kw={"width_ratios": (1.35, 1.0), "wspace": 0.25},
    )

    grid = np.linspace(100, 1100, 750)
    for dataset in DATASET_ORDER:
        values = values_by_dataset[dataset]
        # Focus the KDE on the region containing conventional small molecules;
        # the full high-MW tail is retained in the survival panel.
        central = values[values <= 1100]
        density = gaussian_kde(central)(grid)
        density_axis.plot(
            grid,
            density,
            color=COLORS[dataset],
            linestyle=LINESTYLES[dataset],
            linewidth=3,
            label=f"{dataset} (n={len(values):,})",
        )
        density_axis.fill_between(grid, density, color=COLORS[dataset], alpha=0.08)

        ordered = np.sort(values)
        survival = (len(ordered) - np.arange(len(ordered))) / len(ordered)
        tail_axis.step(
            ordered,
            survival,
            where="post",
            color=COLORS[dataset],
            linestyle=LINESTYLES[dataset],
            linewidth=2.6,
            label=dataset,
        )

    density_axis.axvline(
        cutoff,
        color="#B2182B",
        linestyle=":",
        linewidth=2.5,
        label=f"Proposed cutoff ({cutoff:g} Da)",
    )
    density_axis.set_xlim(100, 1100)
    density_axis.set_xlabel("Molecular weight (Da)")
    density_axis.set_ylabel("Probability density")
    density_axis.set_title("Retained and removed distributions", fontweight="bold")
    density_axis.legend(loc="upper right", frameon=False, fontsize=14)
    density_axis.grid(axis="y", linestyle=":", alpha=0.3)

    tail_axis.axvline(cutoff, color="#B2182B", linestyle=":", linewidth=2.5)
    tail_axis.set_xscale("log")
    tail_axis.set_yscale("log")
    tail_axis.set_xlim(100, 6000)
    tail_axis.set_ylim(8e-5, 1.05)
    tail_axis.set_xlabel("Molecular weight (Da, log scale)")
    tail_axis.set_ylabel("Fraction with MW ≥ x")
    tail_axis.set_title("Full molecular-weight tail", fontweight="bold")
    tail_axis.grid(which="both", linestyle=":", alpha=0.25)

    for axis in (density_axis, tail_axis):
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Molecular-weight distributions of the clearance datasets",
        fontsize=25,
        fontweight="bold",
        y=0.99,
    )
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.cutoff <= 0:
        raise ValueError("--cutoff must be positive.")
    values_by_dataset = {
        "CL-1 retained": molecular_weights(args.cl1),
        "CL-1 removed": molecular_weights(args.cl1_removed),
        "CL-2": molecular_weights(args.cl2),
        "CL-Test": molecular_weights(args.cl_test),
    }
    summary = summarize(values_by_dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_name("clearance_molecular_weight_summary.csv")
    summary.to_csv(summary_path, index=False)
    create_figure(values_by_dataset, args.output, args.cutoff)
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved figure: {args.output.resolve()}")
    print(f"Saved figure: {args.output.with_suffix('.pdf').resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
