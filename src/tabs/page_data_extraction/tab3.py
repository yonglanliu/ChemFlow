# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import pandas as pd
import streamlit as st
from pathlib import Path
from src.config import CONFIG
from src.streamlit.data_extraction import add_quality_flags, show_bioactivity_table
from src.utils import safe_mol_wt
from src.data.extract_pubchem_data import (
    fetch_pubchem_target_assays,
    add_compounds,
    PipelineConfig,
)
from src.config import PROJECT_ROOT


def inject_css():
    css = Path(PROJECT_ROOT) / "assets" / "css" / "styles.css"
    css = css.read_text()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def init_state():
    defaults = {
        "tab3_targets_df": None,
        "tab3_bioactivity_df": None,
        "tab3_original_bioactivity_df": None,
        "tab3_display_df": None,
        "tab3_deleted_rows_df": pd.DataFrame(),
        "tab3_activity_col": None,
        "tab3_activity_name_col": "Activity Name",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "tab3_uniprot_id" not in st.session_state:
        selected_uniprot_ids = st.session_state.get("selected_uniprot_id", [])

        if isinstance(selected_uniprot_ids, str):
            selected_uniprot_ids = [selected_uniprot_ids]

        st.session_state["tab3_uniprot_id"] = "\n".join(selected_uniprot_ids)


def clear_pubchem_query():
    st.session_state["tab3_uniprot_id"] = ""
    st.session_state["tab3_targets_df"] = None
    st.session_state["tab3_bioactivity_df"] = None
    st.session_state["tab3_original_bioactivity_df"] = None
    st.session_state["tab3_display_df"] = None
    st.session_state["tab3_deleted_rows_df"] = pd.DataFrame()
    st.session_state["tab3_activity_col"] = None


def target_query_by_uniprot(cfg: PipelineConfig):
    st.header("Search PubChem by UniProt ID")

    uniprot_input = st.text_area(
        "UniProt ID",
        key="tab3_uniprot_id",
        placeholder="Example:\nP42336\nP42338",
    )

    c1, c2 = st.columns([3, 1])

    with c1:
        query_clicked = st.button(
            "QUERY",
            type="primary",
            key="tab3_pubchem_query",
            use_container_width=True,
        )

    with c2:
        st.button(
            "Clear",
            key="tab3_clear_uniprot_query",
            use_container_width=True,
            on_click=clear_pubchem_query,
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

        with st.spinner("Fetching PubChem assay data..."):
            for uniprot_id in id_list:
                try:
                    temp_df = fetch_pubchem_target_assays(uniprot_id, cfg)

                    if not temp_df.empty:
                        temp_df["query_uniprot"] = uniprot_id
                        dfs.append(temp_df)

                except Exception as e:
                    st.error(f"Failed to fetch PubChem data for {uniprot_id}: {e}")

        if dfs:
            targets_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
            st.session_state["tab3_targets_df"] = targets_df

            st.success(f"PubChem query finished: {len(targets_df)} rows.")
            st.caption("AID = Assay ID; CID = Compound ID; SID = Substance ID.")
            st.dataframe(targets_df, use_container_width=True)
        else:
            st.session_state["tab3_targets_df"] = None
            st.warning("No PubChem assay data found.")


def set_bioactivity_query():
    df = st.session_state.get("tab3_targets_df")

    if df is None or df.empty:
        return None, None, None, None

    query_list = df.columns.tolist()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        default_fields = [
            col
            for col in CONFIG["pubchem"]["default_query_fields"]
            if col in query_list
        ]

        query_fields = st.multiselect(
            "Columns to Keep",
            options=query_list,
            default=default_fields if default_fields else query_list,
            key="tab3_query_fields",
        )

    with c2:
        query_type = st.selectbox(
            "Activity Type",
            options=CONFIG["pubchem"]["activity_types"],
            key="tab3_query_type",
        )

    with c3:
        numeric_candidates = [
            col
            for col in query_list
            if "value" in col.lower()
            or "[nm]" in col.lower()
            or "[um]" in col.lower()
        ]

        activity_col = st.selectbox(
            "Activity Value Column",
            options=query_list,
            index=(
                query_list.index(numeric_candidates[0])
                if numeric_candidates and numeric_candidates[0] in query_list
                else 0
            ),
            key="tab3_activity_col_select",
        )

    with c4:
        drop_na_col = st.multiselect(
            "Remove NaN From",
            options=query_fields,
            default=[col for col in ["CID", activity_col] if col in query_fields],
            key="tab3_drop_na_col",
        )

    st.session_state["tab3_activity_col"] = activity_col

    if "Activity Name" in query_list:
        st.session_state["tab3_activity_name_col"] = "Activity Name"

    return query_fields, query_type, drop_na_col, activity_col


def get_bioactivity_data(
    query_fields,
    query_type,
    drop_na_col,
    activity_col,
    cfg: PipelineConfig,
):
    if query_fields is None:
        return

    if st.button(
        "GET DATA",
        use_container_width=True,
        type="primary",
        key="tab3_query_bio",
    ):
        targets_df = st.session_state.get("tab3_targets_df")

        if targets_df is None or targets_df.empty:
            st.warning("No PubChem table available.")
            return

        if not query_fields:
            st.warning("Please select at least one column.")
            return

        if activity_col not in query_fields:
            st.warning("Activity value column must be included in Columns to Keep.")
            return

        bioactivity_df = targets_df[query_fields].copy()

        with st.status("Curating PubChem data...", expanded=True) as status:
            status.write("Filtering activity type...")

            activity_name_col = st.session_state.get("tab3_activity_name_col")

            if (
                query_type != "All"
                and activity_name_col in bioactivity_df.columns
            ):
                bioactivity_df = bioactivity_df[
                    bioactivity_df[activity_name_col]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    == query_type.upper()
                ].copy()

            if bioactivity_df.empty:
                st.warning("No bioactivity data found after filtering.")
                st.session_state["tab3_bioactivity_df"] = None
                st.session_state["tab3_display_df"] = None
                status.update(label="No data found", state="error")
                return

            status.write("Cleaning numeric activity values...")

            bioactivity_df[activity_col] = pd.to_numeric(
                bioactivity_df[activity_col],
                errors="coerce",
            )

            bioactivity_df["standard_value"] = bioactivity_df[activity_col] * 1000
            bioactivity_df["standard_units"] = "nM"

            if drop_na_col:
                bioactivity_df = bioactivity_df.dropna(subset=drop_na_col).copy()

            if cfg.ligand_id_col not in bioactivity_df.columns:
                st.error(f"Missing required compound ID column: {cfg.ligand_id_col}")
                status.update(label="Missing CID", state="error")
                return

            bioactivity_df[cfg.ligand_id_col] = pd.to_numeric(
                bioactivity_df[cfg.ligand_id_col],
                errors="coerce",
            )

            bioactivity_df = bioactivity_df.dropna(subset=[cfg.ligand_id_col]).copy()
            bioactivity_df[cfg.ligand_id_col] = bioactivity_df[cfg.ligand_id_col].astype(int)

            status.write("Downloading PubChem structures...")

            compound_ids = bioactivity_df[cfg.ligand_id_col].tolist()

            compound_df = add_compounds(
                cfg=cfg,
                cids=compound_ids,
            )

            output_df = bioactivity_df.merge(
                compound_df,
                on=cfg.ligand_id_col,
                how="left",
            )

            smiles_col = None

            if "IsomericSMILES" in output_df.columns:
                smiles_col = "IsomericSMILES"
            elif "CanonicalSMILES" in output_df.columns:
                smiles_col = "CanonicalSMILES"

            if smiles_col:
                output_df["SMILES"] = output_df[smiles_col]
                output_df["mw"] = output_df["SMILES"].apply(safe_mol_wt)

            output_df = add_quality_flags(
                output_df,
                mol_id=cfg.ligand_id_col,
                activity_col=activity_col,
            )

            output_df = output_df.reset_index(drop=True)
            output_df["_row_id"] = range(len(output_df))

            st.session_state["tab3_bioactivity_df"] = output_df.copy()
            st.session_state["tab3_original_bioactivity_df"] = output_df.copy()
            st.session_state["tab3_display_df"] = output_df.copy()
            st.session_state["tab3_deleted_rows_df"] = pd.DataFrame()

            status.update(
                label=f"Finished ({len(output_df):,} records)",
                state="complete",
            )


def design():
    inject_css()
    init_state()

    cfg = PipelineConfig()

    target_query_by_uniprot(cfg)

    query_fields, query_type, drop_na_col, activity_col = set_bioactivity_query()

    get_bioactivity_data(
        query_fields=query_fields,
        query_type=query_type,
        drop_na_col=drop_na_col,
        activity_col=activity_col,
        cfg=cfg,
    )

    show_bioactivity_table(
        ligand_id_col=cfg.ligand_id_col,
        state_original_bioactivity_df_name="tab3_original_bioactivity_df",
        state_bioactivity_df_name="tab3_display_df",
        state_activity_col_name="tab3_activity_col",
        state_display_df_name="tab3_display_df",
        state_deleted_rows_df_name="tab3_deleted_rows_df",
        display_key="tab3_display_mode",
        deleted_selected_row_button_key="tab3_delete_selected_row",
        deleted_checked_rows_button_key="tab3_delete_checked_rows",
        clear_history_button_key="tab3_clear_deleted_history",
        restore_button_key="tab3_restore_original",
        save_button_key="tab3_save_sql",
        bioactivity_table_key="tab3_bioactivity_table",
        editor_key="tab3_bioactivity_editor",
    )