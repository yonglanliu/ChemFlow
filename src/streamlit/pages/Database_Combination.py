from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

from src.streamlit.utils.select_file import file_picker
from src.utils.style import load_css as inject_css


# ============================================================
# Page setup
# ============================================================

inject_css()

st.markdown(
    """
    <div class="page-title">
        Database Combination
    </div>
    """,
    unsafe_allow_html=True,
)

tab_1, tab_2 = st.tabs(
    [
        "DB Combination",
        "Add Data",
    ]
)


# ============================================================
# Session state
# ============================================================

def init_file_picker_state():
    if "file_picker_blocks" not in st.session_state:
        st.session_state["file_picker_blocks"] = []

    if "database_df" not in st.session_state:
        st.session_state["database_df"] = {}

    if "raw_database_df" not in st.session_state:
        st.session_state["raw_database_df"] = {}


def add_file_picker_block():
    existing_ids = [p["id"] for p in st.session_state["file_picker_blocks"]]
    new_id = max(existing_ids) + 1 if existing_ids else 0
    st.session_state["file_picker_blocks"].append({"id": new_id})


def remove_file_picker_block(picker_id):
    st.session_state["file_picker_blocks"] = [
        p
        for p in st.session_state["file_picker_blocks"]
        if p["id"] != picker_id
    ]

    st.session_state["database_df"].pop(picker_id, None)
    st.session_state["raw_database_df"].pop(picker_id, None)


# ============================================================
# File loading
# ============================================================

def load_database_file(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(file_path)

        if suffix == ".tsv":
            return pd.read_csv(file_path, sep="\t")

        if suffix == ".txt":
            return pd.read_csv(file_path, sep=None, engine="python")

        if suffix in [".pkl", ".pickle"]:
            return pd.read_pickle(file_path)

        if suffix in [".db", ".sqlite", ".sqlite3"]:
            with sqlite3.connect(file_path) as conn:
                tables = pd.read_sql_query(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table'
                    ORDER BY name
                    """,
                    conn,
                )

            return {
                "type": "sqlite",
                "tables": tables["name"].tolist(),
                "path": str(file_path),
            }

        st.warning(f"Unsupported file type: {suffix}")
        return None

    except Exception as e:
        st.error(f"Failed to load file:\n{e}")
        return None


def read_sqlite_table(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            f'SELECT * FROM "{table_name}"',
            conn,
        )

    return df


# ============================================================
# UI blocks
# ============================================================

def render_file_picker_block(picker_id, workdir):
    with st.container(border=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            st.subheader(f"Database {picker_id + 1}")

        with col2:
            if st.button("Remove", key=f"remove_{picker_id}"):
                remove_file_picker_block(picker_id)
                st.rerun()

        file = file_picker(
            start_dir=workdir,
            allowed_extensions=(
                ".csv",
                ".tsv",
                ".txt",
                ".pkl",
                ".pickle",
                ".db",
                ".sqlite",
                ".sqlite3",
            ),
            key_prefix=f"{picker_id}_file",
        )

        if not file:
            return

        st.write(f"Selected file: `{file}`")

        result = load_database_file(file)

        if result is None:
            st.warning("Could not load this file.")
            return

        if isinstance(result, dict) and result.get("type") == "sqlite":
            tables = result["tables"]

            if not tables:
                st.warning("No tables found in this SQLite database.")
                return

            st.success(f"SQLite database found: {len(tables)} table(s)")

            table_name = st.selectbox(
                "Select table",
                tables,
                key=f"sqlite_table_{picker_id}",
            )

            try:
                df = read_sqlite_table(result["path"], table_name)
            except Exception as e:
                st.error(f"Failed to read SQLite table:\n{e}")
                return

        else:
            df = result

        if df is None or df.empty:
            st.warning("Loaded table is empty.")
            return

        st.session_state["raw_database_df"][picker_id] = df.copy()
        st.session_state["database_df"][picker_id] = df.copy()

        st.markdown("#### Preview")

        st.dataframe(
            df.head(10),
            width="stretch",
        )


def database_picker_builder(workdir):
    init_file_picker_state()

    if st.button(
        "+ Add Database",
        type="primary",
        width="stretch",
    ):
        add_file_picker_block()
        st.rerun()

    for block in st.session_state["file_picker_blocks"]:
        render_file_picker_block(
            picker_id=block["id"],
            workdir=workdir,
        )


# ============================================================
# Main design
# ============================================================

def design():
    workdir = Path.cwd()

    database_picker_builder(workdir)

    if st.session_state.get("database_df"):
        st.markdown("### Loaded Databases")

        for picker_id, df in st.session_state["database_df"].items():
            st.write(
                f"**Database {picker_id + 1}:** "
                f"{df.shape[0]} rows × {df.shape[1]} columns"
            )


# ============================================================
# Tabs
# ============================================================

with tab_1:
    design()

with tab_2:
    st.info("Add Data function will be added here.")