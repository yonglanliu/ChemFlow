# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

from src.config import CONFIG
from src.utils.chem import safe_mol_wt
from src.streamlit.utils.data_extraction import add_quality_flags, show_bioactivity_table
from src.chemflow.curation.extract_bindingdb_data import (
    fetch_bindingdb_by_uniprot,
    BDBConfig,
    get_default_index,
    clean_numeric_activity,
    convert_to_nM,
)


# ============================================================
# Style
# ============================================================
from src.config import PROJECT_ROOT

def inject_css():
    css = Path(PROJECT_ROOT) / "assets" / "css" / "styles.css"
    css = css.read_text()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )

# ============================================================
# Session state
# ============================================================

def init_state():
    defaults = {
        "tab4_targets_df": None,
        "tab4_bioactivity_df": None,
        "tab4_original_bioactivity_df": None,
        "tab4_display_df": None,
        "tab4_deleted_rows_df": pd.DataFrame(),
        "tab4_standard_activity_col": "standard_value",
        "tab4_ligand_id_for_display": "monomerid",
        "tab4_structure_col_for_display": "smile",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "tab4_uniprot_id" not in st.session_state:
        selected_uniprot_ids = st.session_state.get("selected_uniprot_id", [])

        if isinstance(selected_uniprot_ids, str):
            selected_uniprot_ids = [selected_uniprot_ids]

        st.session_state["tab4_uniprot_id"] = "\n".join(selected_uniprot_ids)


def clear_tab4_uniprot_query():
    """
    Clear input and all downstream BindingDB results.

    Important:
    This function is used as an on_click callback.
    Do not modify st.session_state["tab4_uniprot_id"] after the text_area
    has already been rendered in the same run.
    """
    st.session_state["tab4_uniprot_id"] = ""
    st.session_state["tab4_targets_df"] = None
    st.session_state["tab4_bioactivity_df"] = None
    st.session_state["tab4_original_bioactivity_df"] = None
    st.session_state["tab4_display_df"] = None
    st.session_state["tab4_deleted_rows_df"] = pd.DataFrame()
    st.session_state["tab4_ligand_id_for_display"] = "monomerid"
    st.session_state["tab4_structure_col_for_display"] = "smile"
    st.session_state["tab4_standard_activity_col"] = "standard_value"


# ============================================================
# Query BindingDB by UniProt
# ============================================================

def target_query_by_uniprot(cfg):
    st.header("Search BindingDB by UniProt ID")

    uniprot_input = st.text_area(
        "UniProt ID",
        key="tab4_uniprot_id",
        placeholder="Example:\nP42336\nP42338",
    )

    cutoff = st.number_input(
        "BindingDB cutoff",
        min_value=1,
        max_value=100000,
        value=cfg.cutoff,
        step=100,
        key="tab4_cutoff",
    )

    cfg.cutoff = int(cutoff)

    c1, c2 = st.columns([3, 1])

    with c1:
        query_clicked = st.button(
            "QUERY",
            type="primary",
            key="tab4_bindingdb_query",
            use_container_width=True,
        )

    with c2:
        st.button(
            "Clear",
            key="tab4_clear_uniprot_query",
            use_container_width=True,
            on_click=clear_tab4_uniprot_query,
        )

    if query_clicked:
        id_list = [
            item.strip()
            for item in uniprot_input.splitlines()
            if item.strip()
        ]

        if not id_list:
            st.warning("Please enter or select at least one UniProt ID.")
            return

        dfs = []

        with st.spinner("Fetching BindingDB assay data..."):
            for uniprot_id in id_list:
                try:
                    temp_df = fetch_bindingdb_by_uniprot(uniprot_id, cfg)

                    if not temp_df.empty:
                        temp_df["query_uniprot"] = uniprot_id
                        dfs.append(temp_df)

                except Exception as e:
                    st.error(f"Failed to fetch BindingDB data for {uniprot_id}: {e}")

        if dfs:
            targets_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
            st.session_state["tab4_targets_df"] = targets_df

            st.success(f"BindingDB query finished: {len(targets_df)} rows.")
            st.dataframe(targets_df, use_container_width=True)
        else:
            st.session_state["tab4_targets_df"] = None
            st.warning("No BindingDB assay data found.")


# ============================================================
# Query settings
# ============================================================

def set_bioactivity_query():
    df = st.session_state.get("tab4_targets_df")

    if df is None or df.empty:
        return None, None, None, None, None, None

    query_list = df.columns.tolist()

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        ligand_id = st.selectbox(
            "Ligand ID",
            options=query_list,
            index=get_default_index(query_list, "monomerid"),
            key="tab4_ligand_id",
        )

    with c2:
        activity_col = st.selectbox(
            "Activity Column",
            options=query_list,
            index=get_default_index(query_list, "affinity"),
            key="tab4_activity_col",
        )

    with c3:
        activity_type = st.selectbox(
            "Activity Type",
            options=["IC50", "EC50", "Ki", "Kd", "All"],
            key="tab4_activity_type",
        )

    with c4:
        activity_unit = st.selectbox(
            "Activity Unit",
            options=["nM", "uM", "M", "pM"],
            key="tab4_activity_unit",
        )

    with c5:
        structure_col = st.selectbox(
            "Structure Column",
            options=query_list,
            index=get_default_index(query_list, "smile"),
            key="tab4_structure_col",
        )

    with c6:
        drop_na_col = st.multiselect(
            "Remove NaN From",
            options=query_list,
            default=[
                col
                for col in [ligand_id, activity_col, structure_col]
                if col in query_list
            ],
            key="tab4_drop_na_col",
        )

    return ligand_id, activity_col, activity_type, activity_unit, structure_col, drop_na_col


# ============================================================
# Build bioactivity table
# ============================================================

def get_bioactivity_data(
    ligand_id,
    activity_col,
    activity_type,
    activity_unit,
    structure_col,
    drop_na_col,
):
    if ligand_id is None:
        return

    if st.button(
        "GET DATA",
        use_container_width=True,
        type="primary",
        key="tab4_query_bio",
    ):
        targets_df = st.session_state.get("tab4_targets_df")

        if targets_df is None or targets_df.empty:
            st.warning("No BindingDB table available.")
            return

        bioactivity_df = targets_df.copy()

        with st.status("Curating BindingDB data...", expanded=True) as status:
            status.write("Filtering activity type...")

            if activity_type != "All" and "affinity_type" in bioactivity_df.columns:
                bioactivity_df = bioactivity_df[
                    bioactivity_df["affinity_type"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    == activity_type.upper()
                ].copy()

            if bioactivity_df.empty:
                st.warning("No data found after activity-type filtering.")
                st.session_state["tab4_bioactivity_df"] = None
                st.session_state["tab4_original_bioactivity_df"] = None
                st.session_state["tab4_display_df"] = None
                status.update(label="No data found", state="error")
                return

            status.write("Cleaning activity values...")

            if activity_col not in bioactivity_df.columns:
                st.error(f"Missing activity column: {activity_col}")
                status.update(label="Missing activity column", state="error")
                return

            bioactivity_df[activity_col] = clean_numeric_activity(
                bioactivity_df[activity_col]
            )

            bioactivity_df["standard_value"] = convert_to_nM(
                bioactivity_df[activity_col],
                activity_unit,
            )
            bioactivity_df["standard_units"] = "nM"
            bioactivity_df["standard_type"] = activity_type

            if drop_na_col:
                bioactivity_df = bioactivity_df.dropna(
                    subset=drop_na_col,
                ).copy()

            if ligand_id not in bioactivity_df.columns:
                st.error(f"Missing ligand ID column: {ligand_id}")
                status.update(label="Missing ligand ID", state="error")
                return

            if structure_col not in bioactivity_df.columns:
                st.error(f"Missing structure column: {structure_col}")
                status.update(label="Missing structure", state="error")
                return

            status.write("Calculating molecular properties...")

            bioactivity_df["SMILES"] = bioactivity_df[structure_col]
            bioactivity_df["mw"] = bioactivity_df["SMILES"].apply(safe_mol_wt)

            output_df = add_quality_flags(
                bioactivity_df,
                mol_id=ligand_id,
                activity_col="standard_value",
            )

            output_df = output_df.reset_index(drop=True)
            output_df["_row_id"] = range(len(output_df))

            st.session_state["tab4_bioactivity_df"] = output_df.copy()
            st.session_state["tab4_original_bioactivity_df"] = output_df.copy()
            st.session_state["tab4_display_df"] = output_df.copy()
            st.session_state["tab4_deleted_rows_df"] = pd.DataFrame()

            st.session_state["tab4_ligand_id_for_display"] = ligand_id
            st.session_state["tab4_structure_col_for_display"] = structure_col
            st.session_state["tab4_standard_activity_col"] = "standard_value"

            status.update(
                label=f"Finished ({len(output_df):,} records)",
                state="complete",
            )


# ============================================================
# Main
# ============================================================

def design():
    inject_css()
    init_state()

    cfg = BDBConfig()

    target_query_by_uniprot(cfg)

    (
        ligand_id,
        activity_col,
        activity_type,
        activity_unit,
        structure_col,
        drop_na_col,
    ) = set_bioactivity_query()

    if ligand_id is None:
        return

    get_bioactivity_data(
        ligand_id=ligand_id,
        activity_col=activity_col,
        activity_type=activity_type,
        activity_unit=activity_unit,
        structure_col=structure_col,
        drop_na_col=drop_na_col,
    )

    display_ligand_id_col = st.session_state.get(
        "tab4_ligand_id_for_display",
        ligand_id,
    )

    show_bioactivity_table(
        ligand_id_col=display_ligand_id_col,
        state_original_bioactivity_df_name="tab4_original_bioactivity_df",
        state_bioactivity_df_name="tab4_bioactivity_df",
        state_activity_col_name="tab4_standard_activity_col",
        state_display_df_name="tab4_display_df",
        state_deleted_rows_df_name="tab4_deleted_rows_df",
        display_key="tab4_display_mode",
        deleted_selected_row_button_key="tab4_delete_selected_row",
        deleted_checked_rows_button_key="tab4_delete_checked_rows",
        clear_history_button_key="tab4_clear_deleted_history",
        restore_button_key="tab4_restore_original",
        save_button_key="tab4_save_sql",
        bioactivity_table_key="tab4_bioactivity_table",
        editor_key="tab4_bioactivity_editor",
    )