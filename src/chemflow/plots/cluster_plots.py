# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


ClusterMethod = Literal["Butina", "KMeans", "Agglomerative", "DBSCAN"]
DistanceMetric = Literal["tanimoto", "cosine", "euclidean", "manhattan"]
ReductionMethod = Literal["PCA", "t-SNE", "UMAP"]


# ============================================================
# Fingerprints
# ============================================================

def smiles_to_mol(smiles: str) -> Chem.Mol | None:
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return None

    return mol


def mol_to_morgan_fp(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 2048,
):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
    )


def fp_to_array(fp, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def prepare_molecule_fingerprints(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    ligand_id_col: str | None = None,
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[pd.DataFrame, list, np.ndarray]:
    records = []
    fps = []
    fp_arrays = []

    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {smiles_col}")

    for idx, row in df.iterrows():
        smiles = row.get(smiles_col)
        mol = smiles_to_mol(smiles)

        if mol is None:
            continue

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        fp = mol_to_morgan_fp(
            mol=mol,
            radius=radius,
            n_bits=n_bits,
        )

        fp_arr = fp_to_array(
            fp=fp,
            n_bits=n_bits,
        )

        if ligand_id_col is not None and ligand_id_col in df.columns:
            ligand_id = row.get(ligand_id_col)
        else:
            ligand_id = f"Mol_{idx}"

        record = {
            "Original_Index": idx,
            "Ligand_ID": ligand_id,
            "SMILES": canonical_smiles,
        }

        for col in df.columns:
            if col not in record:
                record[col] = row[col]

        records.append(record)
        fps.append(fp)
        fp_arrays.append(fp_arr)

    if not records:
        return pd.DataFrame(), [], np.empty((0, n_bits), dtype=np.int8)

    valid_df = pd.DataFrame(records).reset_index(drop=True)
    X = np.vstack(fp_arrays)

    return valid_df, fps, X


# ============================================================
# Distance Matrix
# ============================================================

def tanimoto_distance_matrix(fps: list) -> np.ndarray:
    n = len(fps)

    if n == 0:
        return np.empty((0, 0), dtype=float)

    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        dist[i, :] = [1.0 - sim for sim in sims]

    return dist


# ============================================================
# Clustering
# ============================================================

def butina_cluster(
    fps: list,
    cutoff: float = 0.3,
) -> np.ndarray:
    n = len(fps)

    if n == 0:
        return np.array([], dtype=int)

    dists = []

    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - sim for sim in sims])

    clusters = Butina.ClusterData(
        dists,
        nPts=n,
        distThresh=cutoff,
        isDistData=True,
    )

    labels = np.full(n, -1, dtype=int)

    for cluster_id, cluster in enumerate(clusters):
        for mol_idx in cluster:
            labels[mol_idx] = cluster_id

    return labels


def cluster_molecules(
    X: np.ndarray,
    fps: list,
    method: ClusterMethod = "Butina",
    metric: DistanceMetric = "tanimoto",
    n_clusters: int = 5,
    eps: float = 0.5,
    min_samples: int = 5,
    butina_cutoff: float = 0.3,
    random_state: int = 42,
) -> np.ndarray:
    if X.shape[0] == 0:
        raise ValueError("No molecules available for clustering.")

    if method in ["KMeans", "Agglomerative"] and n_clusters > X.shape[0]:
        raise ValueError("n_clusters cannot be larger than number of molecules.")

    if method == "Butina":
        return butina_cluster(
            fps=fps,
            cutoff=butina_cutoff,
        )

    if method == "KMeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )

        return model.fit_predict(X)

    if method == "Agglomerative":
        if metric == "tanimoto":
            dist = tanimoto_distance_matrix(fps)

            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
            )

            return model.fit_predict(dist)

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            linkage="average",
        )

        return model.fit_predict(X)

    if method == "DBSCAN":
        if metric == "tanimoto":
            dist = tanimoto_distance_matrix(fps)

            model = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="precomputed",
            )

            return model.fit_predict(dist)

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
        )

        return model.fit_predict(X)

    raise ValueError(f"Unsupported clustering method: {method}")


# ============================================================
# Dimension Reduction
# ============================================================

def reduce_dimensions(
    X: np.ndarray,
    fps: list,
    method: ReductionMethod = "UMAP",
    metric: DistanceMetric = "tanimoto",
    random_state: int = 42,
) -> np.ndarray:
    if X.shape[0] == 0:
        raise ValueError("No molecules available for dimension reduction.")

    if X.shape[0] < 3:
        raise ValueError("At least 3 molecules are required for 2D visualization.")

    if method == "PCA":
        reducer = PCA(
            n_components=2,
            random_state=random_state,
        )

        return reducer.fit_transform(X)

    if method == "t-SNE":
        perplexity = min(30, max(2, X.shape[0] // 3))

        if metric == "tanimoto":
            dist = tanimoto_distance_matrix(fps)

            reducer = TSNE(
                n_components=2,
                metric="precomputed",
                perplexity=perplexity,
                random_state=random_state,
                init="random",
                learning_rate="auto",
            )

            return reducer.fit_transform(dist)

        reducer = TSNE(
            n_components=2,
            metric=metric,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )

        return reducer.fit_transform(X)

    if method == "UMAP":
        if umap is None:
            raise ImportError(
                "UMAP is not installed. Please install it with: pip install umap-learn"
            )

        if metric == "tanimoto":
            dist = tanimoto_distance_matrix(fps)

            reducer = umap.UMAP(
                n_components=2,
                metric="precomputed",
                random_state=random_state,
            )

            return reducer.fit_transform(dist)

        reducer = umap.UMAP(
            n_components=2,
            metric=metric,
            random_state=random_state,
        )

        return reducer.fit_transform(X)

    raise ValueError(f"Unsupported reduction method: {method}")


# ============================================================
# Nearest Neighbors
# ============================================================

def find_nearest_neighbors(
    query_idx: int,
    fps: list,
    valid_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    if query_idx < 0 or query_idx >= len(fps):
        raise IndexError("query_idx is out of range.")

    query_fp = fps[query_idx]
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, fps)

    nn_df = valid_df.copy()
    nn_df["Tanimoto"] = sims

    nn_df = (
        nn_df
        .drop(index=query_idx)
        .sort_values("Tanimoto", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return nn_df


# ============================================================
# Full Pipeline
# ============================================================

def run_chemical_space_analysis(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    ligand_id_col: str | None = None,
    radius: int = 2,
    n_bits: int = 2048,
    cluster_method: ClusterMethod = "Butina",
    similarity_metric: DistanceMetric = "tanimoto",
    reduction_method: ReductionMethod = "UMAP",
    n_clusters: int = 5,
    eps: float = 0.5,
    min_samples: int = 5,
    butina_cutoff: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list, np.ndarray]:
    valid_df, fps, X = prepare_molecule_fingerprints(
        df=df,
        smiles_col=smiles_col,
        ligand_id_col=ligand_id_col,
        radius=radius,
        n_bits=n_bits,
    )

    if valid_df.empty:
        raise ValueError("No valid molecules were found.")

    labels = cluster_molecules(
        X=X,
        fps=fps,
        method=cluster_method,
        metric=similarity_metric,
        n_clusters=n_clusters,
        eps=eps,
        min_samples=min_samples,
        butina_cutoff=butina_cutoff,
        random_state=random_state,
    )

    coords = reduce_dimensions(
        X=X,
        fps=fps,
        method=reduction_method,
        metric=similarity_metric,
        random_state=random_state,
    )

    result_df = valid_df.copy()
    result_df["Cluster"] = labels.astype(str)
    result_df["Dim1"] = coords[:, 0]
    result_df["Dim2"] = coords[:, 1]

    return result_df, fps, X


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    input_file = "data/combined/combined_pivot.csv"

    df = pd.read_csv(input_file)

    result_df, fps, X = run_chemical_space_analysis(
        df=df,
        smiles_col="SMILES",
        ligand_id_col="Name" if "Name" in df.columns else None,
        cluster_method="Butina",
        similarity_metric="tanimoto",
        reduction_method="UMAP",
        butina_cutoff=0.3,
    )

    result_df.to_csv(
        "chemical_space_results.csv",
        index=False,
    )

    print(result_df.head())