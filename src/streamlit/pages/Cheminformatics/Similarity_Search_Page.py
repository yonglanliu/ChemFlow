# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher

from src.config import PROJECT_ROOT
from src.utils.style import load_css as inject_css
from src.streamlit.utils.select_dir import directory_picker
from src.chemflow.chemistry.similarity_search import SimilarityCalculator
from src.chemflow.io.load_database import load_molecule_database


# ============================================================
# Constants
# ============================================================

SUPPORTED_MOL_FILES = ["sdf", "mol", "mol2", "pdb"]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "combined" / "combined_pivot.csv"


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


def mol_to_clean_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)


def read_uploaded_molecule(uploaded_file) -> Chem.Mol | None:
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        if suffix == ".sdf":
            suppl = Chem.SDMolSupplier(tmp_path, removeHs=False)
            mols = [m for m in suppl if m is not None]
            return mols[0] if mols else None

        if suffix == ".mol":
            return Chem.MolFromMolFile(tmp_path, removeHs=False)

        if suffix == ".mol2":
            return Chem.MolFromMol2File(tmp_path, removeHs=False)

        if suffix == ".pdb":
            return Chem.MolFromPDBFile(tmp_path, removeHs=False)

    except Exception as exc:
        st.error(f"Failed to read molecule file: {exc}")

    return None


def parse_pasted_smiles(text: str) -> list[dict]:
    query_molecules = []

    for i, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        if "," in line:
            name, smiles = line.split(",", 1)
            name = name.strip() or f"Query_{i}"
        else:
            name = f"Query_{i}"
            smiles = line

        smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            st.warning(f"Invalid SMILES skipped: `{smiles}`")
            continue

        query_molecules.append(
            {
                "Name": name,
                "SMILES": mol_to_clean_smiles(mol),
                "Mol": mol,
            }
        )

    return query_molecules


def preview_molecules(query_molecules: list[dict], max_cols: int = 4) -> None:
    if not query_molecules:
        return

    st.success(f"{len(query_molecules)} valid query molecule(s) detected.")

    cols = st.columns(max_cols)

    for i, item in enumerate(query_molecules):
        with cols[i % max_cols]:
            st.markdown(f"**{item['Name']}**")
            st.image(Draw.MolToImage(item["Mol"], size=(240, 180)))
            st.code(item["SMILES"])


def save_query_molecules(query_molecules: list[dict]) -> None:
    st.session_state["query_molecules"] = query_molecules

    if query_molecules:
        st.session_state["query_smiles"] = query_molecules[0]["SMILES"]
        st.session_state["query_mol"] = query_molecules[0]["Mol"]


def get_query_molecules() -> list[dict]:
    return st.session_state.get("query_molecules", [])


def load_database(db_path: Path, smiles_col: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    df = load_molecule_database(db_path, smiles_col=smiles_col)

    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column `{smiles_col}` not found in database.")

    df = df.dropna(subset=[smiles_col]).copy()
    df[smiles_col] = df[smiles_col].astype(str)

    return df


def run_similarity_for_queries(
    query_molecules: list[dict],
    db_df: pd.DataFrame,
    smiles_col: str,
    similarity_mode: str,
    similarity_metric: str,
    radius: int,
    n_bits: int,
    use_features: bool,
    top_n: int,
) -> pd.DataFrame:
    sim_calc = SimilarityCalculator(
        mode=similarity_mode,
        metric=similarity_metric,
        radius=radius,
        n_bits=n_bits,
        use_features=use_features,
    )

    all_results = []

    for query in query_molecules:
        result = sim_calc.search_dataframe(
            query["SMILES"],
            db_df,
            smiles_col=smiles_col,
        )

        result = result.head(top_n).copy()
        result.insert(0, "Query_Name", query["Name"])
        result.insert(1, "Query_SMILES", query["SMILES"])

        all_results.append(result)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


# ============================================================
# Page Setup
# ============================================================

inject_css()

st.markdown(
    """
    <div class="page-title">
        Similarity Search
    </div>
    """,
    unsafe_allow_html=True,
)

divider()


# ============================================================
# Working Directory
# ============================================================

with st.expander("Working Directory", expanded=True):
    selected_workdir = directory_picker(
        label="Select Working Directory",
        start_dir=PROJECT_ROOT,
        key_prefix="similarity_search",
        key="similarity_search_workdir",
    )

workdir = Path(selected_workdir) if selected_workdir else PROJECT_ROOT
divider()


# ============================================================
# Query Molecule Input
# ============================================================

section_title("Choose Molecule Input Method")

query_source = st.radio(
    label="",
    options=["Draw Molecule", "Use SMILES", "Upload Molecule File"],
    horizontal=True,
)

query_molecules: list[dict] = []

if query_source == "Draw Molecule":
    drawn_smiles = st_ketcher(
        "",
        height=500,
        key="query_ketcher",
    )

    if drawn_smiles:
        mol = Chem.MolFromSmiles(drawn_smiles)

        if mol is None:
            st.error("Invalid drawn molecule.")
        else:
            query_molecules = [
                {
                    "Name": "Drawn_Query",
                    "SMILES": mol_to_clean_smiles(mol),
                    "Mol": mol,
                }
            ]

            preview_molecules(query_molecules)
            save_query_molecules(query_molecules)

elif query_source == "Use SMILES":
    pasted_text = st.text_area(
        "Paste SMILES Here",
        height=250,
        placeholder=(
            "One molecule per line.\n\n"
            "Accepted formats:\n"
            "CCO\n"
            "CC(=O)Oc1ccccc1C(=O)O\n\n"
            "Or with names:\n"
            "Mol1,CCO\n"
            "Mol2,CCCO\n"
            "Aspirin,CC(=O)Oc1ccccc1C(=O)O"
        ),
    )

    if pasted_text.strip():
        query_molecules = parse_pasted_smiles(pasted_text)
        preview_molecules(query_molecules)
        save_query_molecules(query_molecules)

elif query_source == "Upload Molecule File":
    uploaded_query_file = st.file_uploader(
        "Upload query molecule file",
        type=SUPPORTED_MOL_FILES,
        accept_multiple_files=False,
    )

    if uploaded_query_file is not None:
        mol = read_uploaded_molecule(uploaded_query_file)

        if mol is None:
            st.error("Could not read molecule from uploaded file.")
        else:
            query_molecules = [
                {
                    "Name": Path(uploaded_query_file.name).stem,
                    "SMILES": mol_to_clean_smiles(mol),
                    "Mol": mol,
                }
            ]

            preview_molecules(query_molecules)
            save_query_molecules(query_molecules)

            if mol.GetNumConformers() > 0:
                st.info("3D coordinates detected. You can use 3D shape similarity.")
            else:
                st.warning("No 3D coordinates detected. Use 2D similarity or generate conformers first.")


# ============================================================
# Similarity Settings
# ============================================================

divider()
section_title("Similarity Settings")

c1, c2, c3 = st.columns([1, 1, 1], gap="small", vertical_alignment="bottom")

with c1:
    similarity_mode = st.selectbox(
        "Similarity Type",
        options=[
            "2d_fingerprint",
            "2d_descriptor",
            "topological_descriptor",
            "3d_shape",
        ],
        index=0,
    )

with c2:
    if similarity_mode == "2d_fingerprint":
        metric_options = ["tanimoto", "dice", "cosine", "mcconnaughey"]
    elif similarity_mode in ["2d_descriptor", "topological_descriptor"]:
        metric_options = ["cosine", "euclidean", "manhattan"]
    else:
        metric_options = ["shape_tanimoto", "shape_protrude"]

    similarity_metric = st.selectbox(
        "Similarity Metric",
        options=metric_options,
        index=0,
    )

with c3:
    top_n = st.number_input(
        "Top N Results",
        min_value=1,
        max_value=500,
        value=20,
        step=1,
    )

if similarity_mode == "2d_fingerprint":
    c1, c2, c3 = st.columns([1, 1, 1], gap="small", vertical_alignment="bottom")

    with c1:
        radius = st.number_input(
            "Morgan Radius",
            min_value=1,
            max_value=4,
            value=2,
            step=1,
        )

    with c2:
        n_bits = st.selectbox(
            "Fingerprint Bits",
            options=[1024, 2048, 4096],
            index=1,
        )

    with c3:
        use_features = st.checkbox(
            "Use FCFP-like Features",
            value=False,
        )
else:
    radius = 2
    n_bits = 2048
    use_features = False


# ============================================================
# Database Settings
# ============================================================

divider()
section_title("Database Settings")

db_path = st.text_input(
    "Database CSV Path",
    value=str(DEFAULT_DB_PATH),
)

smiles_col = st.text_input(
    "Database SMILES Column",
    value="SMILES",
)


# ============================================================
# Run Similarity Search
# ============================================================

divider()

query_molecules = get_query_molecules()

if query_molecules:
    st.info("Query molecule(s) are ready for similarity search.")
else:
    st.warning("Please provide at least one valid query molecule.")

_, center, _ = st.columns([2, 2, 2])

with center:
    run = st.button(
        "🔍 Run Similarity Search",
        type="primary",
        use_container_width=True,
        disabled=not bool(query_molecules),
    )

if run:
    try:
        db_df = load_database(Path(db_path), smiles_col=smiles_col)

        results = run_similarity_for_queries(
            query_molecules=query_molecules,
            db_df=db_df,
            smiles_col=smiles_col,
            similarity_mode=similarity_mode,
            similarity_metric=similarity_metric,
            radius=radius,
            n_bits=n_bits,
            use_features=use_features,
            top_n=int(top_n),
        )

        if results.empty:
            st.warning("No similarity results found.")
        else:
            st.subheader(f"Top {top_n} Similar Compounds")

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )

            csv = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="similarity_search_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as exc:
        st.error(f"Similarity search failed: {exc}")