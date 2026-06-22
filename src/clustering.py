from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def smiles_to_fps(smiles_list, radius=2, n_bits=2048):
    fps = []
    arrs = []
    valid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
        )

        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)

        fps.append(fp)
        arrs.append(arr)
        valid_smiles.append(smi)

    return fps, np.array(arrs), valid_smiles


def butina_cluster(fps, tanimoto_cutoff=0.6):
    dists = []

    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    distance_cutoff = 1 - tanimoto_cutoff

    clusters = Butina.ClusterData(
        dists,
        len(fps),
        distance_cutoff,
        isDistData=True,
    )

    labels = np.full(len(fps), -1)

    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cluster_id

    return labels


def plot_cluster_scatter(X, labels, title="Cluster visualization"):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        s=35,
        alpha=0.8,
    )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    return fig


def plot_cluster_sizes(labels):
    cluster_counts = pd.Series(labels).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    cluster_counts.plot(kind="bar", ax=ax)

    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Compounds")

    return fig

st.divider()
st.subheader("Clustering Analysis")

cluster_method = st.selectbox(
    "Choose clustering method",
    ["Butina", "KMeans", "Agglomerative", "DBSCAN"],
)

radius = st.slider("Morgan fingerprint radius", 1, 4, 2)
n_bits = st.selectbox("Fingerprint bits", [1024, 2048, 4096], index=1)

smiles_list = df[smiles_col].dropna().tolist()

fps, X, valid_smiles = smiles_to_fps(
    smiles_list,
    radius=radius,
    n_bits=n_bits,
)

if len(valid_smiles) < 2:
    st.warning("Not enough valid molecules for clustering.")
    st.stop()

if cluster_method == "Butina":
    tanimoto_cutoff = st.slider(
        "Tanimoto similarity cutoff",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
    )

elif cluster_method in ["KMeans", "Agglomerative"]:
    n_clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=min(20, len(valid_smiles)),
        value=5,
    )

elif cluster_method == "DBSCAN":
    eps = st.slider(
        "DBSCAN eps",
        min_value=0.1,
        max_value=10.0,
        value=3.0,
        step=0.1,
    )

    min_samples = st.slider(
        "DBSCAN min_samples",
        min_value=2,
        max_value=20,
        value=5,
    )

if st.button("Run Clustering"):

    if cluster_method == "Butina":
        labels = butina_cluster(
            fps,
            tanimoto_cutoff=tanimoto_cutoff,
        )

    elif cluster_method == "KMeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init="auto",
        )
        labels = model.fit_predict(X)

    elif cluster_method == "Agglomerative":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
        )
        labels = model.fit_predict(X)

    elif cluster_method == "DBSCAN":
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
        )
        labels = model.fit_predict(X)

    cluster_df = pd.DataFrame({
        "smiles": valid_smiles,
        "cluster": labels,
    })

    st.subheader("Clustered Compounds")
    st.dataframe(cluster_df, use_container_width=True)

    st.subheader("Cluster Size Distribution")
    fig1 = plot_cluster_sizes(labels)
    st.pyplot(fig1)

    st.subheader("2D PCA Cluster Plot")
    fig2 = plot_cluster_scatter(
        X,
        labels,
        title=f"{cluster_method} Clustering of Compounds",
    )
    st.pyplot(fig2)

    csv = cluster_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download clustering results",
        data=csv,
        file_name="clustering_results.csv",
        mime="text/csv",
    )