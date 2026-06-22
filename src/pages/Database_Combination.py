import streamlit as st
from src.utils.select_file import file_picker
from pathlib import Path
import pandas as pd
import sqlite3


st.markdown(
    """
    <h1 style="
        text-align:center;
        background: linear-gradient(90deg,#005388,#00A6D6);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        font-weight:800;
    ">
        Database Combination
    </h1>
    """,
    unsafe_allow_html=True,
)

tab_1, tab_2 = st.tabs([
    "DB Combination",
    "Add Data",
])


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
        p for p in st.session_state["file_picker_blocks"]
        if p["id"] != picker_id
    ]

    st.session_state["database_df"].pop(picker_id, None)
    st.session_state["raw_database_df"].pop(picker_id, None)


def load_database_file(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(file_path)

        elif suffix == ".tsv":
            return pd.read_csv(file_path, sep="\t")

        elif suffix == ".txt":
            return pd.read_csv(file_path, sep=None, engine="python")

        elif suffix in [".pkl", ".pickle"]:
            return pd.read_pickle(file_path)

        elif suffix in [".db", ".sqlite", ".sqlite3"]:
            conn = sqlite3.connect(file_path)

            tables = pd.read_sql_query(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                ORDER BY name
                """,
                conn,
            )

            conn.close()

            return {
                "type": "sqlite",
                "tables": tables["name"].tolist(),
                "path": str(file_path),
            }

    except Exception as e:
        st.error(f"Failed to load file:\n{e}")

    return None


def read_sqlite_table(db_path, table_name):
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        f'SELECT * FROM "{table_name}"',
        conn,
    )

    conn.close()

    return df


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

        working_df = df.copy()

        st.markdown("#### Filter Activity Type")

        cols = working_df.columns.tolist()

        st.dataframe(
            working_df.head(10),
            use_container_width=True,
        )
        


def database_picker_builder(workdir):
    init_file_picker_state()

    if st.button("+ Add Database", type="primary", use_container_width=True):
        add_file_picker_block()
        st.rerun()

    for block in st.session_state["file_picker_blocks"]:
        render_file_picker_block(
            picker_id=block["id"],
            workdir=workdir,
        )


def design():
    workdir = Path.cwd()

    st.markdown(
        """
        <h3 style="
            text-align:left;
            background: linear-gradient(90deg,#005388,#00A6D6);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            font-weight:800;
        ">
            Select Databases
        </h3>
        """,
        unsafe_allow_html=True,
    )

    database_picker_builder(workdir)

    if st.session_state.get("database_df"):
        st.markdown("### Loaded Databases")

        for picker_id, df in st.session_state["database_df"].items():
            st.write(
                f"**Database {picker_id + 1}:** "
                f"{df.shape[0]} rows × {df.shape[1]} columns"
            )


with tab_1:
    design()

with tab_2:
    st.info("Add Data function will be added here.")