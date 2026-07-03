# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from pathlib import Path

import pandas as pd
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from src.config import CONFIG
from src.chemflow.curation.extract_chembl_data import (
    fetch_bioactivity_data, 
    add_doi_data, 
    add_compounds,
    target_query_by_uniprot,
    ChEMBLConfig,
)
from src.streamlit.utils.data_extraction import add_quality_flags, show_bioactivity_table


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
        "tab2_targets_df": None,
        "tab2_bioactivity_df": None,
        "tab2_original_bioactivity_df": None,
        "tab2_display_df": None,
        "tab2_deleted_rows_df": pd.DataFrame(),
        "tab2_activity_col": None,
        "tab2_activity_name_col": "Activity Name",
        "tab2_target_chembl_id_col": "Target ChEMBL ID",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "tab2_uniprot_id" not in st.session_state:
        selected_uniprot_ids = st.session_state.get("selected_uniprot_id", [])

        if isinstance(selected_uniprot_ids, str):
            selected_uniprot_ids = [selected_uniprot_ids]

        st.session_state["tab2_uniprot_id"] = "\n".join(selected_uniprot_ids)


# ============================================================
# Target query
# ============================================================
def target_query_by_uniprot_streamlit():
    st.header("Search ChEMBL Target ID")

    selected_uniprot_ids = st.session_state.get("selected_uniprot_id", [])

    if isinstance(selected_uniprot_ids, str):
        selected_uniprot_ids = [selected_uniprot_ids]

    uniprot_ids_text = "\n".join(selected_uniprot_ids)

    if (
        uniprot_ids_text
        and st.session_state.get("tab2_loaded_uniprot_text") != uniprot_ids_text
    ):
        st.session_state["tab2_uniprot_id"] = uniprot_ids_text
        st.session_state["tab2_loaded_uniprot_text"] = uniprot_ids_text

    uniprot_input = st.text_area(
        "UniProt ID",
        key="tab2_uniprot_id",
        placeholder="Example:\nP42336\nP42338",
    )

    if st.button("QUERY", type="primary", key="tab2_chembl_query"):
        id_list = [
            item.strip()
            for item in uniprot_input.splitlines()
            if item.strip()
        ]

        dfs = []

        for uniprot_id in id_list:
            temp_df = target_query_by_uniprot(uniprot_id)

            if not temp_df.empty:
                temp_df["query_uniprot"] = uniprot_id
                dfs.append(temp_df)

        if dfs:
            targets_df = pd.concat(dfs, ignore_index=True)

            if "target_chembl_id" in targets_df.columns:
                targets_df = targets_df.drop_duplicates(
                    subset=["target_chembl_id"],
                    keep="first",
                )

            st.session_state["tab2_targets_df"] = targets_df.reset_index(
                drop=True
            )
            st.success("Target query finished.")
        else:
            st.session_state["tab2_targets_df"] = None
            st.warning("No target found.")


def select_chembl_id():
    targets = st.session_state.get("tab2_targets_df")

    if targets is None or targets.empty:
        return

    event = st.dataframe(
        targets,
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        key="tab2_targets_table",
    )

    selected_rows = event.selection.rows

    if selected_rows:
        row_idx = selected_rows[0]
        selected_target = targets.iloc[row_idx]

        chembl_id = selected_target["target_chembl_id"]
        st.session_state["tab2_sel_chembl_id"] = chembl_id

        st.success(f"Selected target: {chembl_id}")
        st.write(selected_target)



# ============================================================
# Bioactivity query settings
# ============================================================
def set_bioactivity_query():
    st.header("Fetch Bioactivity Data From ChEMBL")
    
    query_list = CONFIG["chembl"]["bioactivity_query_fields"]

    default_query_list = CONFIG["chembl"]["default_bioactivity_query_fields"]

    query_fields = st.multiselect(
        "Data to Query",
        options=query_list,
        default=default_query_list,
        key="tab2_query_fields",
    )
    c1, c2 = st.columns(2)
    with c1:
        query_type = st.selectbox(
            "Query Type",
            options=CONFIG["chembl"]["bioactivity_query_types"],
            key="tab2_query_type",
        )
    with c2:
        assay_fields = st.multiselect(
            "Assay Type",
            options=CONFIG["chembl"]["assay_types"],
            default=CONFIG["chembl"]["default_assay_types"],
            key="tab2_assay_fields",
        )


    return query_fields, query_type, assay_fields


# ============================================================
# Bioactivity download
# ============================================================
def get_bioactivity_data_streamlit(query_fields, query_type, assay_fields, cfg):
    if st.button(
        "FETCH DATA",
        use_container_width=True,
        type="primary",
        key="tab2_query_bio",
    ):
        chembl_id = st.session_state.get("tab2_sel_chembl_id")

        if chembl_id is None:
            st.error("Please select target.")
            return

        with st.status("Connecting to ChEMBL...", expanded=True) as status:
            status.write("Sending bioactivity query...")

            bioactivity_df = fetch_bioactivity_data(target_chembl_id=chembl_id,
                                                    query_fields=query_fields,
                                                    query_type=query_type,
                                                    assay_fields=assay_fields)

            if bioactivity_df.empty:
                st.warning("No bioactivity data found.")
                st.session_state["tab2_bioactivity_df"] = None
                st.session_state["tab2_display_df"] = None
                status.update(label="No data found", state="error")
                return

            status.write("Downloading document DOI information...")

            if "document_chembl_id" in bioactivity_df.columns:
                bioactivity_df = add_doi_data(bioactivity_df, cfg=cfg)

            if "standard_value" in bioactivity_df.columns:
                bioactivity_df["standard_value"] = pd.to_numeric(
                    bioactivity_df["standard_value"],
                    errors="coerce",
                )

            status.write("Downloading molecule structures...")

            if "molecule_chembl_id" in bioactivity_df.columns:
                output_df = add_compounds(bioactivity_df, cfg=cfg)
            
            output_df = add_quality_flags(output_df)
            output_df["_row_id"] = range(len(output_df))

            st.session_state["tab2_bioactivity_df"] = output_df.copy()
            st.session_state["tab2_original_bioactivity_df"] = output_df.copy()
            st.session_state["tab2_display_df"] = output_df.copy()
            st.session_state["tab2_deleted_rows_df"] = pd.DataFrame()

            status.update(
                label=f"Finished ({len(output_df):,} records)",
                state="complete",
            )


def get_chembl_client():
    try:
        from chembl_webresource_client.new_client import new_client
        return new_client
    except Exception as e:
        st.error("Could not connect to ChEMBL webresource client.")
        st.code(str(e))
        return None

# ============================================================
# Main
# ============================================================
def design():
    inject_css()
    init_state()

    new_client = get_chembl_client()
    if new_client is None:
        st.stop()

    cfg= ChEMBLConfig()

    target_query_by_uniprot_streamlit()
    select_chembl_id()

    query_fields, query_type, assay_fields = set_bioactivity_query()

    get_bioactivity_data_streamlit(query_fields=query_fields, query_type=query_type, assay_fields=assay_fields, cfg=cfg)
    st.session_state["tab2_activity_col"] = "standard_value"

    show_bioactivity_table(ligand_id_col=cfg.ligand_id_col, 
                           state_original_bioactivity_df_name="tab2_original_bioactivity_df",
                           state_bioactivity_df_name="tab2_display_df",
                           state_activity_col_name="tab2_activity_col",
                           state_display_df_name="tab2_display_df",
                           state_deleted_rows_df_name="tab2_deleted_rows_df",
                           display_key="tab2_display_mode",
                           deleted_selected_row_button_key="tab2_delete_selected_row",
                           deleted_checked_rows_button_key="tab2_delete_checked_rows",
                           clear_history_button_key="tab2_clear_deleted_history",
                           restore_button_key="tab2_restore_original",
                           save_button_key="tab2_save_sql",
                           bioactivity_table_key="tab2_bioactivity_table",
                           editor_key="tab2_bioactivity_editor")