# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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

from src.chemflow.chemistry.clustering import (
    prepare_molecules,
    cluster_molecules,
    reduce_dimensions,
    find_nearest_neighbors,
)

# ============================================================
# Helpers
# ============================================================

def divider() -> None:
    st.markdown(
        """
        <div style="
            height:3px;
            width:100%;
            background:linear-gradient(90deg,#3b82f6,#06b6d4,#10b981);
            border-radius:10px;
            margin:20px 0;
        "></div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_default_index(cols: list[str], preferred: list[str]) -> int:
    for col in preferred:
        if col in cols:
            return cols.index(col)
    return 0


def load_database(file_path: str | Path) -> pd.DataFrame:
    try:
        return load_molecule_database(file_path)
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()






# ============================================================
# Page Setup
# ============================================================

inject_css()

st.markdown(
    """
    <div class="page-title">
        Chemical Space Visualization
    </div>
    """,
    unsafe_allow_html=True,
)

divider()


# ============================================================
# Load Database
# ============================================================

section_title("Load Database")

database_file = file_picker(
    start_dir=PROJECT_ROOT,
    key_prefix="database_file_for_clustering",
    allowed_extensions=[".csv", ".tsv", ".sdf", ".sd", ".smi", ".db"],
)

database_df = load_database(database_file) if database_file else pd.DataFrame()

if database_df.empty:
    st.warning("Please select a valid molecule database file.")
    st.stop()

st.success(
    f"Loaded database: {database_df.shape[0]} rows × {database_df.shape[1]} columns"
)

st.dataframe(
    database_df.head(20),
    use_container_width=True,
)

divider()


# ============================================================
# Column Selection
# ============================================================

cols = database_df.columns.tolist()

if not cols:
    st.error("The database has no columns.")
    st.stop()

c1, c2 = st.columns(2, gap="large", vertical_alignment="bottom")

with c1:
    section_title("SMILES Column")
    structure_col = st.selectbox(
        "",
        options=cols,
        index=get_default_index(
            cols,
            [
                "SMILES",
                "smiles",
                "Canonical_SMILES",
                "canonical_smiles",
                "Structure",
            ],
        ),
        label_visibility="collapsed",
    )

with c2:
    section_title("Ligand ID Column")
    ligand_id_col = st.selectbox(
        "",
        options=cols,
        index=get_default_index(
            cols,
            [
                "Name",
                "name",
                "Ligand_ID",
                "ligand_id",
                "molecule_chembl_id",
                "Molecule ChEMBL ID",
            ],
        ),
        label_visibility="collapsed",
    )


# ============================================================
# Clustering Settings
# ============================================================

divider()
section_title("Clustering Settings")

c3, c4, c5, c6 = st.columns(4, gap="large", vertical_alignment="bottom")

with c3:
    cluster_method = st.selectbox(
        "Clustering Method",
        options=["Butina", "KMeans", "Agglomerative", "DBSCAN"],
        index=0,
        label_visibility="collapsed",
    )

with c4:
    similarity_metric = st.selectbox(
        "Similarity Metric",
        options=["tanimoto", "cosine", "euclidean"],
        index=0,
        label_visibility="collapsed",
    )

with c5:
    visualization_type = st.selectbox(
        "Visualization Type",
        options=["UMAP", "t-SNE", "PCA"],
        index=0,
        label_visibility="collapsed",
    )

with c6:
    if cluster_method == "Butina":
        butina_cutoff = st.number_input(
            "Butina distance cutoff",
            min_value=0.05,
            max_value=1.00,
            value=0.30,
            step=0.05,
        )
        n_clusters = 5
        eps = 0.5
        min_samples = 5

    elif cluster_method == "DBSCAN":
        eps = st.number_input(
            "DBSCAN eps",
            min_value=0.01,
            max_value=5.00,
            value=0.50,
            step=0.01,
        )

        min_samples = st.number_input(
            "Minimum samples",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )

        n_clusters = 5
        butina_cutoff = 0.30

    else:
        n_clusters = st.number_input(
            "Number of clusters",
            min_value=2,
            max_value=50,
            value=5,
            step=1,
        )
        eps = 0.5
        min_samples = 5
        butina_cutoff = 0.30


# ============================================================
# Fingerprint Settings
# ============================================================

divider()
section_title("Fingerprint Parameters")

c8, c9 = st.columns(2, gap="large", vertical_alignment="bottom")

with c8:
    radius = st.number_input(
        "Morgan fingerprint radius",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
    )

with c9:
    n_bits = st.selectbox(
        "Fingerprint bits",
        options=[1024, 2048, 4096],
        index=1,
    )


# ============================================================
# Run Clustering
# ============================================================

divider()

_, center, _ = st.columns([2, 2, 2])

with center:
    run = st.button(
        "Run Clustering",
        type="primary",
        use_container_width=True,
    )

if run:
    with st.spinner("Preparing fingerprints..."):
        valid_df, fps, X, invalid_count = prepare_molecules(
            df=database_df,
            smiles_col=structure_col,
            ligand_id_col=ligand_id_col,
            radius=int(radius),
            n_bits=int(n_bits),
        )

    if invalid_count > 0:
        st.warning(f"Skipped {invalid_count} invalid or unreadable molecule(s).")

    if valid_df.empty:
        st.error("No valid molecules found. Please check your SMILES column.")
        st.stop()

    if len(valid_df) < 3:
        st.error("At least 3 valid molecules are required for visualization.")
        st.stop()

    if cluster_method in ["KMeans", "Agglomerative"] and len(valid_df) < int(n_clusters):
        st.error("Number of clusters cannot be larger than number of valid molecules.")
        st.stop()

    st.success(f"Prepared {len(valid_df)} valid molecules.")

    with st.spinner("Running clustering..."):
        labels = cluster_molecules(
            X=X,
            fps=fps,
            method=cluster_method,
            metric=similarity_metric,
            n_clusters=int(n_clusters),
            eps=float(eps),
            min_samples=int(min_samples),
            butina_cutoff=float(butina_cutoff),
        )

    valid_df["Cluster"] = labels.astype(str)

    with st.spinner("Reducing dimensions..."):
        coords = reduce_dimensions(
            X=X,
            fps=fps,
            method=visualization_type,
            metric=similarity_metric,
        )

    valid_df["Dim1"] = coords[:, 0]
    valid_df["Dim2"] = coords[:, 1]

    st.session_state["chemical_space_df"] = valid_df
    st.session_state["chemical_space_fps"] = fps
    st.session_state["chemical_space_params"] = {
        "cluster_method": cluster_method,
        "similarity_metric": similarity_metric,
        "visualization_type": visualization_type,
    }

    st.success("Clustering and visualization completed.")


# ============================================================
# Explorer
# ============================================================

if (
    "chemical_space_df" in st.session_state
    and "chemical_space_fps" in st.session_state
):
    valid_df = st.session_state["chemical_space_df"].reset_index(drop=True)
    fps = st.session_state["chemical_space_fps"]
    params = st.session_state.get("chemical_space_params", {})

    cluster_method = params.get("cluster_method", "Clustering")
    visualization_type = params.get("visualization_type", "Projection")

    divider()
    section_title("Chemical Space Explorer")

    color_options = ["Cluster"]

    for possible_col in [
        "Ligand_ID",
        "Original_Index",
    ]:
        if possible_col in valid_df.columns:
            color_options.append(possible_col)

    color_by = st.selectbox(
        "Color by",
        options=color_options,
        index=0,
    )

    fig = px.scatter(
        valid_df,
        x="Dim1",
        y="Dim2",
        color=color_by,
        hover_data=["Ligand_ID", "SMILES", "Cluster"],
        title=f"{cluster_method} clustering visualized by {visualization_type}",
        labels={
            "Dim1": f"{visualization_type} 1",
            "Dim2": f"{visualization_type} 2",
        },
    )

    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.8,
            line=dict(width=0.4, color="DarkSlateGrey"),
        )
    )

    fig.update_layout(
        height=650,
        legend_title_text=color_by,
    )

    st.plotly_chart(fig, use_container_width=True)

    molecular_inspector, true_nearest_neighbors = st.columns([2, 1], gap="large")

    with molecular_inspector:
        st.markdown("### Molecule Inspector")

        selected_ligand = st.selectbox(
            "Select molecule",
            options=valid_df["Ligand_ID"].astype(str).tolist(),
        )

        selected_position = valid_df.index[
            valid_df["Ligand_ID"].astype(str) == selected_ligand
        ].tolist()[0]

        selected_row = valid_df.iloc[selected_position]

        st.markdown("**Selected Molecule**")
        st.write(f"**Ligand ID:** {selected_row['Ligand_ID']}")
        st.write(f"**Cluster:** {selected_row['Cluster']}")
        st.code(selected_row["SMILES"])

        mol = Chem.MolFromSmiles(selected_row["SMILES"])

        if mol is not None:
            st.image(
                Draw.MolToImage(mol, size=(300, 220)),
                caption="2D Structure",
            )

    with true_nearest_neighbors:
        top_nn = st.number_input(
            "Top N nearest neighbors",
            min_value=3,
            max_value=50,
            value=10,
            step=1,
        )

        nn_df = find_nearest_neighbors(
            query_position=int(selected_position),
            fps=fps,
            valid_df=valid_df,
            top_n=int(top_nn),
        )

        st.markdown("### True nearest neighbors")
        st.caption("Calculated by real Tanimoto similarity, not 2D plot distance.")

        st.dataframe(
            nn_df[
                [
                    "Ligand_ID",
                    "Cluster",
                    "Tanimoto",
                    "SMILES",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    divider()

    st.subheader("Cluster Summary")

    summary_df = (
        valid_df["Cluster"]
        .value_counts()
        .rename_axis("Cluster")
        .reset_index(name="Count")
        .sort_values("Cluster")
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Clustered Molecules")

    st.dataframe(
        valid_df,
        use_container_width=True,
        hide_index=True,
    )

    csv = valid_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Clustered Molecules",
        data=csv,
        file_name="clustered_molecules.csv",
        mime="text/csv",
        use_container_width=True,
    )