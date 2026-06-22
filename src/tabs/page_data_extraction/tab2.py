# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors


# ============================================================
# Style
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
        /* thicker divider */
        hr {
            margin-top: 1.2rem;
            margin-bottom: 1.2rem;
            border: none;
            height: 4px;
            background-color: #c9c9c9;
        }

        /* custom button size */
        div.stButton > button {
            height: 3rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 10px;
        }

        /* make primary buttons slightly stronger */
        div.stButton > button[kind="primary"] {
            height: 3.2rem;
            font-size: 1.05rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Session state
# ============================================================
def init_state():
    defaults = {
        "tab2_targets_df": None,
        "tab2_sel_chembl_id": None,
        "tab2_bioactivity_df": None,
        "tab2_original_bioactivity_df": None,
        "tab2_display_df": None,
        "tab2_deleted_rows_df": pd.DataFrame(),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# Helpers
# ============================================================
def thick_divider():
    st.markdown("<hr>", unsafe_allow_html=True)


def safe_mol_wt(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Descriptors.MolWt(mol)


def save_df_to_sql(df, table_name, db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


def add_quality_flags(df):
    df = df.copy()

    if "standard_value" in df.columns:
        df["standard_value"] = pd.to_numeric(
            df["standard_value"],
            errors="coerce",
        )
        df["flag_missing_value"] = df["standard_value"].isna()
    else:
        df["flag_missing_value"] = True

    if "standard_units" in df.columns:
        df["flag_non_nM"] = (
            df["standard_units"].isna()
            | (df["standard_units"].astype(str).str.strip() != "nM")
        )
    else:
        df["flag_non_nM"] = True

    if "molecule_chembl_id" in df.columns:
        df["flag_duplicate_molecule"] = df.duplicated(
            subset=["molecule_chembl_id"],
            keep=False,
        )
    else:
        df["flag_duplicate_molecule"] = False

    return df


# ============================================================
# Target query
# ============================================================
def target_query_by_uniprot(targets_api):
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

    fields_list = [
        "target_chembl_id",
        "organism",
        "pref_name",
        "target_type",
    ]

    if st.button("QUERY", type="primary", key="tab2_chembl_query"):
        id_list = [
            item.strip()
            for item in uniprot_input.splitlines()
            if item.strip()
        ]

        dfs = []

        for uniprot_id in id_list:
            records = targets_api.get(
                target_components__accession=uniprot_id
            ).only(*fields_list)

            temp_df = pd.DataFrame.from_records(records)

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

    thick_divider()


# ============================================================
# Bioactivity query settings
# ============================================================
def set_bioactivity_query():
    st.header("Get Bioactivity Data")

    c1, c2 = st.columns(2)

    with c1:
        query_list = [
            "activity_id",
            "assay_chembl_id",
            "assay_description",
            "assay_type",
            "molecule_chembl_id",
            "type",
            "standard_units",
            "relation",
            "standard_value",
            "target_chembl_id",
            "target_organism",
            "document_chembl_id",
        ]

        default_query_list = [
            "activity_id",
            "assay_description",
            "assay_type",
            "molecule_chembl_id",
            "type",
            "standard_units",
            "standard_value",
            "target_chembl_id",
            "target_organism",
            "document_chembl_id",
        ]

        query_fields = st.multiselect(
            "Data to Query",
            options=query_list,
            default=default_query_list,
            key="tab2_query_fields",
        )

    with c2:
        query_type = st.selectbox(
            "Query Type",
            options=["IC50", "EC50", "Ki"],
            key="tab2_query_type",
        )

        assay_fields = st.multiselect(
            "Assay Type",
            options=["B", "F"],
            default=["B"],
            key="tab2_assay_fields",
        )

    thick_divider()

    return query_fields, query_type, assay_fields


# ============================================================
# Bioactivity download
# ============================================================
def get_bioactivity_data(
    query_fields,
    query_type,
    assay_fields,
    bioactivities_api,
    compounds_api,
    documents_api,
):
    if st.button(
        "GET BIOACTIVITY DATA",
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

            bioactivities = bioactivities_api.filter(
                target_chembl_id=chembl_id,
                type=query_type,
                relation="=",
                assay_type__in=assay_fields,
            ).only(*query_fields)

            bioactivity_df = pd.DataFrame.from_records(bioactivities)

            if bioactivity_df.empty:
                st.warning("No bioactivity data found.")
                st.session_state["tab2_bioactivity_df"] = None
                st.session_state["tab2_display_df"] = None
                status.update(label="No data found", state="error")
                return

            status.write("Downloading document DOI information...")

            if "document_chembl_id" in bioactivity_df.columns:
                document_ids = (
                    bioactivity_df["document_chembl_id"]
                    .dropna()
                    .unique()
                    .tolist()
                )

                if document_ids:
                    documents = documents_api.filter(
                        document_chembl_id__in=document_ids
                    ).only(
                        "document_chembl_id",
                        "doi",
                    )

                    documents_df = pd.DataFrame.from_records(documents)

                    if not documents_df.empty:
                        bioactivity_df = pd.merge(
                            bioactivity_df,
                            documents_df,
                            how="left",
                            on="document_chembl_id",
                        )

            if "standard_value" in bioactivity_df.columns:
                bioactivity_df["standard_value"] = pd.to_numeric(
                    bioactivity_df["standard_value"],
                    errors="coerce",
                )

            status.write("Downloading molecule structures...")

            if "molecule_chembl_id" in bioactivity_df.columns:
                molecule_ids = (
                    bioactivity_df["molecule_chembl_id"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            else:
                molecule_ids = []

            if molecule_ids:
                compounds_provider = compounds_api.filter(
                    molecule_chembl_id__in=molecule_ids
                ).only(
                    "molecule_chembl_id",
                    "molecule_structures",
                )

                compounds_df = pd.DataFrame.from_records(
                    list(compounds_provider)
                )

                if not compounds_df.empty:

                    def get_smiles(x):
                        try:
                            return x["canonical_smiles"]
                        except Exception:
                            return None

                    compounds_df["SMILES"] = compounds_df[
                        "molecule_structures"
                    ].apply(get_smiles)

                    compounds_df = compounds_df[
                        ["molecule_chembl_id", "SMILES"]
                    ].dropna()

                    compounds_df = compounds_df.drop_duplicates(
                        subset=["molecule_chembl_id"],
                        keep="first",
                    )

                    output_df = pd.merge(
                        bioactivity_df,
                        compounds_df,
                        how="left",
                        on="molecule_chembl_id",
                    )
                else:
                    output_df = bioactivity_df.copy()
            else:
                output_df = bioactivity_df.copy()

            output_df.reset_index(drop=True, inplace=True)

            if "SMILES" in output_df.columns:
                output_df["mw"] = output_df["SMILES"].apply(safe_mol_wt)

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


# ============================================================
# Display mode
# ============================================================
def aggregate_duplicate_values(df, method="mean"):
    df = df.copy()

    if "molecule_chembl_id" not in df.columns:
        return df

    if "standard_value" not in df.columns:
        return df.drop_duplicates(
            subset=["molecule_chembl_id"],
            keep="first",
        )

    df["standard_value"] = pd.to_numeric(
        df["standard_value"],
        errors="coerce",
    )

    if method == "mean":
        agg_values = (
            df.groupby("molecule_chembl_id", as_index=False)["standard_value"]
            .mean()
            .rename(columns={"standard_value": "standard_value_mean"})
        )

    elif method == "median":
        agg_values = (
            df.groupby("molecule_chembl_id", as_index=False)["standard_value"]
            .median()
            .rename(columns={"standard_value": "standard_value_median"})
        )

    else:
        return df

    base_df = df.drop_duplicates(
        subset=["molecule_chembl_id"],
        keep="first",
    ).copy()

    base_df = base_df.drop(columns=["standard_value"], errors="ignore")

    output_df = pd.merge(
        base_df,
        agg_values,
        on="molecule_chembl_id",
        how="left",
    )

    if method == "mean":
        output_df["standard_value"] = output_df["standard_value_mean"]
        output_df = output_df.drop(
            columns=["standard_value_mean"],
            errors="ignore",
        )

    elif method == "median":
        output_df["standard_value"] = output_df["standard_value_median"]
        output_df = output_df.drop(
            columns=["standard_value_median"],
            errors="ignore",
        )

    output_df["aggregation_method"] = method
    output_df = add_quality_flags(output_df)

    return output_df

def display_mode(bioactivity_df):
    duplicate_mode = st.radio(
        "Duplicate molecule handling",
        [
            "Show all",
            "Show duplicates first",
            "Keep first only",
            "Keep best potency only",
            "Use mean value",
            "Use median value",
            "Show duplicates only",
        ],
        horizontal=True,
        key="tab2_duplicate_mode",
    )

    display_df = bioactivity_df.copy()

    if duplicate_mode == "Show duplicates first":
        display_df = display_df.sort_values(
            by=["flag_duplicate_molecule", "molecule_chembl_id"],
            ascending=[False, True],
        )

    elif duplicate_mode == "Keep first only":
        display_df = display_df.drop_duplicates(
            subset=["molecule_chembl_id"],
            keep="first",
        )

    elif duplicate_mode == "Keep best potency only":
        display_df = (
            display_df.sort_values("standard_value", ascending=True)
            .drop_duplicates(
                subset=["molecule_chembl_id"],
                keep="first",
            )
        )

    elif duplicate_mode == "Use mean value":
        display_df = aggregate_duplicate_values(
            display_df,
            method="mean",
        )

    elif duplicate_mode == "Use median value":
        display_df = aggregate_duplicate_values(
            display_df,
            method="median",
        )

    elif duplicate_mode == "Show duplicates only":
        display_df = display_df[
            display_df["flag_duplicate_molecule"] == True
        ]

    return display_df


# ============================================================
# Save current display
# ============================================================
def save_data():
    display_df = st.session_state.get("tab2_display_df")
    chembl_id = st.session_state.get("tab2_sel_chembl_id")

    if display_df is None or display_df.empty:
        st.warning("No displayed table to save.")
        return

    save_df = display_df.drop(
        columns=["_row_id", "_delete"],
        errors="ignore",
    )

    db_name = f"{chembl_id}_bioactivity.db" if chembl_id else "bioactivity.db"
    db_path = Path("data") / db_name

    if st.button(
        "Save SQL",
        use_container_width=True,
        key="tab2_save_sql",
    ):
        save_df_to_sql(
            df=save_df,
            table_name="bioactivity",
            db_path=db_path,
        )

        st.success(f"Saved current displayed table to: {db_path}")


# ============================================================
# Bioactivity table
# ============================================================
def show_bioactivity_table():
    bioactivity_df = st.session_state.get("tab2_bioactivity_df")

    if bioactivity_df is None or bioactivity_df.empty:
        return

    st.header("Bioactivity Data")

    bioactivity_df = add_quality_flags(bioactivity_df)

    if "_row_id" not in bioactivity_df.columns:
        bioactivity_df["_row_id"] = range(len(bioactivity_df))
        st.session_state["tab2_bioactivity_df"] = bioactivity_df.copy()

    display_df = display_mode(bioactivity_df)
    display_df = display_df.reset_index(drop=True)

    # Important: this is the dataframe that Save SQL will save.
    st.session_state["tab2_display_df"] = display_df.copy()

    st.info(f"Current display: {len(display_df)} row(s)")

    if "_delete" not in display_df.columns:
        display_df.insert(0, "_delete", False)

    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="dynamic",
        key="tab2_bioactivity_editor",
        column_config={
            "_delete": st.column_config.CheckboxColumn(
                "Delete",
                help="Check rows to delete",
                default=False,
            ),
            "_row_id": st.column_config.NumberColumn(
                "_row_id",
                disabled=True,
            ),
        },
        disabled=["_row_id"],
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button(
            "Delete Checked",
            use_container_width=True,
            key="tab2_delete_checked_rows",
        ):
            deleted = edited_df[edited_df["_delete"] == True].copy()

            if deleted.empty:
                st.warning("No rows checked for deletion.")
                return

            deleted_row_ids = deleted["_row_id"].tolist()

            # Delete from the master table, not the filtered display table.
            master_df = st.session_state["tab2_bioactivity_df"].copy()

            deleted_from_master = master_df[
                master_df["_row_id"].isin(deleted_row_ids)
            ].copy()

            kept_master = master_df[
                ~master_df["_row_id"].isin(deleted_row_ids)
            ].copy()

            st.session_state["tab2_deleted_rows_df"] = pd.concat(
                [
                    st.session_state["tab2_deleted_rows_df"],
                    deleted_from_master,
                ],
                ignore_index=True,
            )

            kept_master = add_quality_flags(kept_master)

            st.session_state["tab2_bioactivity_df"] = kept_master.copy()

            st.success(f"Deleted {len(deleted_row_ids)} row(s).")
            st.rerun()

    with c2:
        if st.button(
            "Clear History",
            use_container_width=True,
            key="tab2_clear_deleted_history",
        ):
            st.session_state["tab2_deleted_rows_df"] = pd.DataFrame()
            st.success("Deleted-row history cleared.")
            st.rerun()

    with c3:
        if st.button(
            "Restore Original",
            use_container_width=True,
            key="tab2_restore_original",
        ):
            original_df = st.session_state.get("tab2_original_bioactivity_df")

            if original_df is None or original_df.empty:
                st.warning("No original table to restore.")
            else:
                st.session_state["tab2_bioactivity_df"] = original_df.copy()
                st.session_state["tab2_display_df"] = original_df.copy()
                st.session_state["tab2_deleted_rows_df"] = pd.DataFrame()
                st.success("Original table restored.")
                st.rerun()

    with c4:
        save_data()

    thick_divider()

    st.caption(
        "Show all means all remaining rows after deletion. "
        "Deleted rows will not reappear unless you restore the original table. "
        "Save SQL saves the current displayed table."
    )

    event = st.dataframe(
        display_df,
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        key="tab2_bioactivity_table",
    )

    selected_rows = event.selection.rows

    if selected_rows:
        row_idx = selected_rows[0]
        selected_row = display_df.iloc[row_idx]

        molecule_chembl_id = selected_row.get(
            "molecule_chembl_id",
            "Unknown",
        )
        smiles = selected_row.get("SMILES")

        st.subheader("Selected Molecule")
        st.write("Selected molecule:", molecule_chembl_id)

        if pd.notna(smiles):
            mol = Chem.MolFromSmiles(str(smiles))

            if mol is not None:
                st.image(
                    Draw.MolToImage(mol, size=(350, 300)),
                    caption=molecule_chembl_id,
                )
            else:
                st.warning("Invalid SMILES.")
        else:
            st.warning("No SMILES found for this molecule.")

        doi = selected_row.get("doi")
        if pd.notna(doi):
            st.markdown(f"**DOI:** https://doi.org/{doi}")

        st.write(selected_row)

        if st.button(
            "Delete Selected Row",
            use_container_width=True,
            key="tab2_delete_selected_row",
        ):
            row_id = selected_row["_row_id"]

            master_df = st.session_state["tab2_bioactivity_df"].copy()

            deleted_row = master_df[
                master_df["_row_id"] == row_id
            ].copy()

            updated_df = master_df[
                master_df["_row_id"] != row_id
            ].copy()

            st.session_state["tab2_deleted_rows_df"] = pd.concat(
                [
                    st.session_state["tab2_deleted_rows_df"],
                    deleted_row,
                ],
                ignore_index=True,
            )

            updated_df = add_quality_flags(updated_df)
            st.session_state["tab2_bioactivity_df"] = updated_df.copy()

            st.success("Selected row deleted.")
            st.rerun()

    deleted_rows_df = st.session_state.get("tab2_deleted_rows_df")

    if deleted_rows_df is not None and not deleted_rows_df.empty:
        with st.expander("Deleted rows history", expanded=False):
            st.dataframe(
                deleted_rows_df.drop(
                    columns=["_delete"],
                    errors="ignore",
                ),
                use_container_width=True,
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

    compounds_api = new_client.molecule
    bioactivities_api = new_client.activity
    targets_api = new_client.target
    documents_api = new_client.document

    target_query_by_uniprot(targets_api)
    select_chembl_id()

    query_fields, query_type, assay_fields = set_bioactivity_query()

    get_bioactivity_data(
        query_fields=query_fields,
        query_type=query_type,
        assay_fields=assay_fields,
        bioactivities_api=bioactivities_api,
        compounds_api=compounds_api,
        documents_api=documents_api,
    )

    show_bioactivity_table()