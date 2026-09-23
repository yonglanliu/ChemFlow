#!/usr/bin/env python3
"""Plot chemical-space t-SNE for combined and ExpansionRX clearance data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


DATASET_ORDER = (
    "ChEMBL + Biogen",
    "ExpansionRX train",
    "ExpansionRX test",
)
COLORS = {
    "ChEMBL + Biogen": "#0072B2",
    "ExpansionRX train": "#E69F00",
    "ExpansionRX test": "#009E73",
}
MARKERS = {
    "ChEMBL + Biogen": "o",
    "ExpansionRX train": "^",
    "ExpansionRX test": "D",
}


def sample_frame(
    frame: pd.DataFrame,
    dataset: str,
    max_molecules: int,
    seed: int,
) -> pd.DataFrame:
    frame = frame.dropna(subset=["SMILES", "InChIKey"]).drop_duplicates("InChIKey")
    if max_molecules > 0 and len(frame) > max_molecules:
        frame = frame.sample(n=max_molecules, random_state=seed)
    result = frame[["Ligand_ID", "SMILES", "InChIKey"]].copy()
    result["Dataset"] = dataset
    return result


def fingerprints(frame: pd.DataFrame, radius: int, n_bits: int) -> tuple[pd.DataFrame, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=True,
    )
    arrays: list[np.ndarray] = []
    valid_indices: list[int] = []
    for index, smiles in frame["SMILES"].items():
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            continue
        fingerprint = generator.GetFingerprint(molecule)
        array = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        arrays.append(array)
        valid_indices.append(index)
    if not arrays:
        raise ValueError("No valid molecular fingerprints were generated.")
    return frame.loc[valid_indices].reset_index(drop=True), np.vstack(arrays)


def calculate_tsne(
    matrix: np.ndarray,
    *,
    pca_components: int,
    perplexity: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    components = min(pca_components, matrix.shape[0] - 1, matrix.shape[1])
    pca = PCA(n_components=components, random_state=seed)
    reduced = pca.fit_transform(matrix)
    effective_perplexity = min(perplexity, max(5.0, (len(reduced) - 1) / 3))
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        max_iter=iterations,
        init="pca",
        random_state=seed,
        method="barnes_hut",
        angle=0.5,
        n_jobs=-1,
        verbose=1,
    ).fit_transform(reduced)
    return embedding, float(pca.explained_variance_ratio_.sum())


def plot_embedding(frame: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 23,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 17,
            "figure.titlesize": 25,
            "font.family": "DejaVu Sans",
        }
    )
    figure, axis = plt.subplots(figsize=(12.5, 10))
    sizes = {
        "ChEMBL + Biogen": 15,
        "ExpansionRX train": 24,
        "ExpansionRX test": 34,
    }
    alphas = {
        "ChEMBL + Biogen": 0.32,
        "ExpansionRX train": 0.52,
        "ExpansionRX test": 0.78,
    }
    for dataset in DATASET_ORDER:
        selected = frame.loc[frame["Dataset"].eq(dataset)]
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

    axis.set_title("Molecular chemical-space coverage", fontweight="bold", pad=14)
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.grid(linestyle=":", linewidth=0.9, alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", frameon=False)
    figure.suptitle(
        "ChEMBL–Biogen and ExpansionRX clearance chemical space",
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.012,
        "Morgan fingerprints (radius 2, 2,048 bits) → PCA → t-SNE; fixed random seed.",
        ha="center",
        fontsize=14,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
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
        "--output",
        type=Path,
        default=root / "clearance_chemical_space_tsne.png",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=0,
        help="Maximum molecules per dataset; 0 includes every molecule.",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bits < 1 or args.radius < 0 or args.iterations < 250:
        raise ValueError("Fingerprint size, radius, and t-SNE iterations are invalid.")
    frames = {
        "ChEMBL + Biogen": pd.read_csv(args.combined),
        "ExpansionRX train": pd.read_csv(args.expansionrx_train),
        "ExpansionRX test": pd.read_csv(args.expansionrx_test),
    }
    sampled = pd.concat(
        [
            sample_frame(frame, dataset, args.max_per_dataset, args.seed)
            for dataset, frame in frames.items()
        ],
        ignore_index=True,
    )
    sampled, matrix = fingerprints(sampled, args.radius, args.n_bits)
    print(f"Calculating t-SNE for {len(sampled):,} molecules...")
    embedding, pca_variance = calculate_tsne(
        matrix,
        pca_components=args.pca_components,
        perplexity=args.perplexity,
        iterations=args.iterations,
        seed=args.seed,
    )
    sampled[["tSNE_1", "tSNE_2"]] = embedding

    args.output.parent.mkdir(parents=True, exist_ok=True)
    coordinates_path = args.output.with_name("clearance_chemical_space_tsne.csv.gz")
    sampled.to_csv(coordinates_path, index=False, compression="gzip")
    plot_embedding(sampled, args.output)

    print("\nMolecules shown by dataset:")
    print(sampled["Dataset"].value_counts().reindex(DATASET_ORDER).to_string())
    print(f"PCA variance retained: {pca_variance:.3f}")
    print(f"Saved figure: {args.output.resolve()}")
    print(f"Saved figure: {args.output.with_suffix('.pdf').resolve()}")
    print(f"Saved coordinates: {coordinates_path.resolve()}")


if __name__ == "__main__":
    main()
