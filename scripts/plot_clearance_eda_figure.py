#!/usr/bin/env python3
"""Create a publication-style multipanel EDA figure for clearance datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from plot_clearance_distributions import density_curve, finite_values


ENDPOINTS = {
    "HLM": "Log10_HLM_CLint_mL_min_kg",
    "RLM": "Log10_RLM_CLint_mL_min_kg",
    "MLM": "Log10_MLM_CLint_mL_min_kg",
}
DATASETS = (
    "CL-1",
    "CL-2",
    "CL-Test",
)
COLORS = {
    "CL-1": "#0072B2",
    "CL-2": "#E69F00",
    "CL-Test": "#009E73",
    "CL-Test → CL-Joint": "#7A5195",
}
LINESTYLES = {
    "CL-1": "-",
    "CL-2": "--",
    "CL-Test": "-.",
}
MARKERS = {
    "CL-1": "o",
    "CL-2": "^",
    "CL-Test": "D",
}


def morgan_fingerprints(smiles: pd.Series) -> list[DataStructs.ExplicitBitVect]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True,
    )
    fingerprints = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES: {value}")
        fingerprints.append(generator.GetFingerprint(molecule))
    return fingerprints


def maximum_similarities(
    queries: list[DataStructs.ExplicitBitVect],
    references: list[DataStructs.ExplicitBitVect],
) -> np.ndarray:
    return np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(query, references)) for query in queries],
        dtype=float,
    )


def calculate_similarity_table(
    combined: pd.DataFrame,
    expansion_train: pd.DataFrame,
    expansion_test: pd.DataFrame,
) -> pd.DataFrame:
    combined_fp = morgan_fingerprints(combined["SMILES"])
    train_fp = morgan_fingerprints(expansion_train["SMILES"])
    test_fp = morgan_fingerprints(expansion_test["SMILES"])
    train_to_combined = maximum_similarities(train_fp, combined_fp)
    test_to_combined = maximum_similarities(test_fp, combined_fp)
    test_to_train = maximum_similarities(test_fp, train_fp)
    test_to_joint = np.maximum(test_to_combined, test_to_train)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "Comparison": "CL-2 → CL-1",
                    "Maximum_Tanimoto": train_to_combined,
                }
            ),
            pd.DataFrame(
                {
                    "Comparison": "CL-Test → CL-1",
                    "Maximum_Tanimoto": test_to_combined,
                }
            ),
            pd.DataFrame(
                {
                    "Comparison": "CL-Test → CL-Joint",
                    "Maximum_Tanimoto": test_to_joint,
                }
            ),
        ],
        ignore_index=True,
    )


def plot_endpoint_distributions(
    axes: list[plt.Axes], frames: dict[str, pd.DataFrame]
) -> None:
    for axis, (endpoint, column) in zip(axes, ENDPOINTS.items()):
        values_by_dataset = {
            dataset: finite_values(frames[dataset], column) for dataset in DATASETS
        }
        pooled = np.concatenate(
            [values for values in values_by_dataset.values() if len(values)]
        )
        span = np.max(pooled) - np.min(pooled)
        padding = max(0.12, span * 0.05)
        grid = np.linspace(np.min(pooled) - padding, np.max(pooled) + padding, 450)
        for dataset in DATASETS:
            values = values_by_dataset[dataset]
            if not len(values):
                continue
            density = density_curve(values, grid)
            axis.plot(
                grid,
                density,
                color=COLORS[dataset],
                linestyle=LINESTYLES[dataset],
                linewidth=2.5,
            )
            axis.fill_between(grid, density, color=COLORS[dataset], alpha=0.08)
            axis.axvline(
                np.median(values),
                color=COLORS[dataset],
                linestyle=LINESTYLES[dataset],
                linewidth=1.4,
                alpha=0.9,
            )
        axis.set_title(f"{endpoint} CL$_{{int}}$", fontweight="bold")
        axis.set_xlabel(r"$\log_{10}$(mL min$^{-1}$ kg$^{-1}$)")
        axis.set_ylabel("Density")
        axis.grid(axis="y", linestyle=":", alpha=0.3)
        axis.spines[["top", "right"]].set_visible(False)


def plot_tsne(axis: plt.Axes, coordinates: pd.DataFrame) -> None:
    sizes = {
        "CL-1": 8,
        "CL-2": 13,
        "CL-Test": 20,
    }
    alphas = {
        "CL-1": 0.30,
        "CL-2": 0.50,
        "CL-Test": 0.75,
    }
    for dataset in DATASETS:
        selected = coordinates.loc[coordinates["Dataset"].eq(dataset)]
        axis.scatter(
            selected["tSNE_1"],
            selected["tSNE_2"],
            s=sizes[dataset],
            marker=MARKERS[dataset],
            color=COLORS[dataset],
            alpha=alphas[dataset],
            linewidths=0,
            label=f"{dataset} (n={len(selected):,})",
            rasterized=True,
        )
    axis.set_title("Chemical-space coverage", fontweight="bold")
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.grid(linestyle=":", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=19,
        markerscale=2.0,
        handletextpad=0.6,
        labelspacing=0.9,
    )


def plot_similarity(axis: plt.Axes, similarity: pd.DataFrame) -> None:
    comparisons = (
        "CL-2 → CL-1",
        "CL-Test → CL-1",
        "CL-Test → CL-Joint",
    )
    values = [
        similarity.loc[
            similarity["Comparison"].eq(comparison), "Maximum_Tanimoto"
        ].to_numpy()
        for comparison in comparisons
    ]
    violins = axis.violinplot(
        values,
        positions=np.arange(1, 4),
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.78,
    )
    violin_colors = (
        COLORS["CL-2"],
        COLORS["CL-Test"],
        COLORS["CL-Test → CL-Joint"],
    )
    for body, color in zip(violins["bodies"], violin_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)

    box = axis.boxplot(
        values,
        positions=np.arange(1, 4),
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"},
    )
    for patch, color in zip(box["boxes"], violin_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    labels = (
        "CL-2 → CL-1",
        "CL-Test → CL-1",
        "CL-Test → CL-Joint",
    )
    axis.set_xticks(np.arange(1, 4), labels)
    axis.tick_params(axis="x", labelsize=19, pad=11)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Maximum Morgan Tanimoto similarity")
    axis.set_title("Nearest-neighbor coverage", fontweight="bold")
    axis.grid(axis="y", linestyle=":", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    for index, group in enumerate(values, start=1):
        axis.text(
            index,
            0.035,
            f"median={np.median(group):.2f}",
            ha="center",
            va="bottom",
            fontsize=17,
            rotation=0,
        )


def create_figure(
    frames: dict[str, pd.DataFrame],
    coordinates: pd.DataFrame,
    similarity: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 20,
            "axes.linewidth": 1.8,
            "xtick.labelsize": 17,
            "xtick.major.size": 7,
            "xtick.major.width": 1.6,
            "ytick.labelsize": 17,
            "ytick.major.size": 7,
            "ytick.major.width": 1.6,
        }
    )
    figure = plt.figure(figsize=(20, 13.5))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 1.35),
        left=0.06,
        right=0.98,
        bottom=0.07,
        top=0.84,
        hspace=0.38,
    )
    top_grid = grid[0].subgridspec(1, 3, wspace=0.20)
    bottom_grid = grid[1].subgridspec(
        1,
        2,
        width_ratios=(1.25, 1.0),
        wspace=0.22,
    )
    distribution_axes = [
        figure.add_subplot(top_grid[0, index]) for index in range(3)
    ]
    tsne_axis = figure.add_subplot(bottom_grid[0, 0])
    similarity_axis = figure.add_subplot(bottom_grid[0, 1])

    plot_endpoint_distributions(distribution_axes, frames)
    plot_tsne(tsne_axis, coordinates)
    plot_similarity(similarity_axis, similarity)
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[dataset],
            linestyle=LINESTYLES[dataset],
            linewidth=3,
            marker=MARKERS[dataset],
            markersize=7,
            label=dataset,
        )
        for dataset in DATASETS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=3,
        frameon=False,
        fontsize=19,
    )
    figure.suptitle(
        "Exploratory analysis of the clearance modeling datasets",
        fontsize=27,
        fontweight="bold",
        y=0.975,
    )
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
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
        "--tsne-coordinates",
        type=Path,
        default=root / "clearance_chemical_space_tsne.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "clearance_publication_eda.png",
    )
    parser.add_argument(
        "--similarity-cache",
        type=Path,
        default=root / "clearance_nearest_neighbor_similarity.csv.gz",
    )
    parser.add_argument("--recalculate-similarity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = {
        "CL-1": pd.read_csv(args.combined),
        "CL-2": pd.read_csv(args.expansionrx_train),
        "CL-Test": pd.read_csv(args.expansionrx_test),
    }
    coordinates = pd.read_csv(args.tsne_coordinates)
    coordinates["Dataset"] = coordinates["Dataset"].replace(
        {
            "ChEMBL + Biogen": "CL-1",
            "ExpansionRX train": "CL-2",
            "ExpansionRX test": "CL-Test",
        }
    )
    if args.similarity_cache.exists() and not args.recalculate_similarity:
        similarity = pd.read_csv(args.similarity_cache)
        similarity["Comparison"] = similarity["Comparison"].replace(
            {
                "ExpansionRX train → combined": "CL-2 → CL-1",
                "ExpansionRX test → combined": "CL-Test → CL-1",
                "ExpansionRX test → joint": "CL-Test → CL-Joint",
            }
        )
        print(f"Loaded similarity cache: {args.similarity_cache}")
    else:
        print("Calculating nearest-neighbor Tanimoto similarities...")
        similarity = calculate_similarity_table(
            frames["CL-1"],
            frames["CL-2"],
            frames["CL-Test"],
        )
        args.similarity_cache.parent.mkdir(parents=True, exist_ok=True)
        similarity.to_csv(args.similarity_cache, index=False, compression="gzip")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_figure(frames, coordinates, similarity, args.output)
    summary = similarity.groupby("Comparison")["Maximum_Tanimoto"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    summary_path = args.output.with_name("clearance_similarity_summary.csv")
    summary.to_csv(summary_path)
    print(summary.round(4).to_string())
    print(f"\nSaved figure: {args.output.resolve()}")
    print(f"Saved figure: {args.output.with_suffix('.pdf').resolve()}")
    print(f"Saved similarity summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
