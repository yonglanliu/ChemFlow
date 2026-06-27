
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.ML.Cluster import Butina

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import PROJECT_ROOT
from src.utils.style import load_css as inject_css
from src.streamlit.utils.select_file import file_picker
from src.chemflow.io.load_database import load_molecule_database
from src.chemflow.chemistry.similarity_search import SimilarityCalculator
import pandas as pd
import numpy as np



def fp_to_array(fp, n_bits: int) -> np.ndarray | None:
    if fp is None:
        return None

    arr = np.zeros((n_bits,), dtype=np.int8)

    try:
        DataStructs.ConvertToNumpyArray(fp, arr)
    except Exception:
        return None

    return arr

def prepare_molecules(
    df: pd.DataFrame,
    smiles_col: str,
    ligand_id_col: str,
    radius: int,
    n_bits: int,
):
    similarity_calculator = SimilarityCalculator(
        mode="2d_fingerprint",
        metric="tanimoto",
        radius=radius,
        n_bits=n_bits,
    )

    records = []
    fps = []
    fp_arrays = []
    invalid_count = 0

    for idx, row in df.iterrows():
        smiles = row.get(smiles_col)

        if pd.isna(smiles) or not str(smiles).strip():
            invalid_count += 1
            continue

        smiles = str(smiles).strip()
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            invalid_count += 1
            continue

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        try:
            fp = similarity_calculator.calculate_fingerprint(canonical_smiles)
        except Exception:
            invalid_count += 1
            continue

        if fp is None:
            invalid_count += 1
            continue

        fp_arr = fp_to_array(fp, n_bits=n_bits)

        if fp_arr is None:
            invalid_count += 1
            continue

        ligand_id = row.get(ligand_id_col, f"Mol_{idx}")

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
        return pd.DataFrame(), [], np.empty((0, n_bits)), invalid_count

    valid_df = pd.DataFrame(records).reset_index(drop=True)
    X = np.vstack(fp_arrays)

    return valid_df, fps, X, invalid_count


def tanimoto_distance_matrix(fps: list) -> np.ndarray:
    n = len(fps)

    if n == 0:
        return np.empty((0, 0), dtype=float)

    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        dist[i, :] = [1.0 - sim for sim in sims]

    return dist


def butina_cluster(fps: list, cutoff: float = 0.3) -> np.ndarray:
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
    method: str,
    metric: str,
    n_clusters: int,
    eps: float,
    min_samples: int,
    butina_cutoff: float,
) -> np.ndarray:
    if method == "Butina":
        return butina_cluster(
            fps=fps,
            cutoff=butina_cutoff,
        )

    if method == "KMeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
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


def reduce_dimensions(
    X: np.ndarray,
    fps: list,
    method: str,
    metric: str,
) -> np.ndarray:
    n_samples = X.shape[0]

    if n_samples < 3:
        raise ValueError("At least 3 molecules are required for visualization.")

    if method == "PCA":
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(X)

    if method == "t-SNE":
        perplexity = min(30, n_samples - 1)
        perplexity = max(2, perplexity)

        if metric == "tanimoto":
            dist = tanimoto_distance_matrix(fps)

            reducer = TSNE(
                n_components=2,
                metric="precomputed",
                perplexity=perplexity,
                random_state=42,
                init="random",
                learning_rate="auto",
            )

            return reducer.fit_transform(dist)

        reducer = TSNE(
            n_components=2,
            metric=metric,
            perplexity=perplexity,
            random_state=42,
            init="pca",
            learning_rate="auto",
        )

        return reducer.fit_transform(X)

    if method == "UMAP":
        try:
            import umap
        except ImportError:
            st.error("UMAP is not installed. Please run: pip install umap-learn")
            st.stop()

        if metric == "tanimoto":
            dist = tanimoto_distance_matrix(fps)

            reducer = umap.UMAP(
                n_components=2,
                metric="precomputed",
                random_state=42,
            )

            return reducer.fit_transform(dist)

        reducer = umap.UMAP(
            n_components=2,
            metric=metric,
            random_state=42,
        )

        return reducer.fit_transform(X)

    raise ValueError(f"Unsupported visualization method: {method}")


def find_nearest_neighbors(
    query_position: int,
    fps: list,
    valid_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    if query_position < 0 or query_position >= len(fps):
        raise IndexError("Selected molecule index is out of range.")

    query_fp = fps[query_position]
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, fps)

    nn_df = valid_df.copy()
    nn_df["Tanimoto"] = sims

    nn_df = (
        nn_df
        .drop(index=query_position)
        .sort_values("Tanimoto", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return nn_df