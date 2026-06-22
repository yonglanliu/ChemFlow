# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors


# ============================================================
# Style
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
        hr {
            margin-top: 1.2rem;
            margin-bottom: 1.2rem;
            border: none;
            height: 4px;
            background-color: #c9c9c9;
        }

        div.stButton > button {
            height: 3rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 10px;
        }

        div.stButton > button[kind="primary"] {
            height: 3.2rem;
            font-size: 1.05rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def thick_divider():
    st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# Config
# ============================================================
@dataclass
class BDBConfig:
    pause_sec: float = 0.25
    retries: int = 4
    connect_timeout_sec: int = 30
    read_timeout_sec: int = 300
    cutoff: int = 1000

    ligand_id_col: str = "monomerid"
    structure_col: str = "smile"
    activity_col: str = "affinity"
    activity_type_col: str = "affinity_type"

    standard_activity_col: str = "standard_value"
    standard_unit: str = "nM"


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

    # Only initialize once.
    # Do NOT overwrite this on every rerun.
    if "tab4_uniprot_id" not in st.session_state:
        selected_uniprot_ids = st.session_state.get("selected_uniprot_id", [])

        if isinstance(selected_uniprot_ids, str):
            selected_uniprot_ids = [selected_uniprot_ids]

        st.session_state["tab4_uniprot_id"] = "\n".join(selected_uniprot_ids)


# ============================================================
# Helpers
# ============================================================
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


def get_default_index(options, target):
    return options.index(target) if target in options else 0


def clean_numeric_activity(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(">", "", regex=False)
        .str.replace("<", "", regex=False)
        .str.replace("=", "", regex=False)
        .str.replace("~", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def convert_to_nM(values, unit):
    unit_factor = {
        "nM": 1.0,
        "uM": 1000.0,
        "µM": 1000.0,
        "μM": 1000.0,
        "M": 1_000_000_000.0,
        "pM": 0.001,
    }

    factor = unit_factor.get(unit, 1.0)
    return values * factor


def add_quality_flags(df, mol_id, activity_col):
    df = df.copy()

    if activity_col and activity_col in df.columns:
        df[activity_col] = pd.to_numeric(df[activity_col], errors="coerce")
        df["flag_missing_value"] = df[activity_col].isna()
    else:
        df["flag_missing_value"] = False

    if mol_id in df.columns:
        df["flag_duplicate_molecule"] = df.duplicated(
            subset=[mol_id],
            keep=False,
        )
    else:
        df["flag_duplicate_molecule"] = False

    return df


# ============================================================
# BindingDB request and parser
# ============================================================
def fetch_bindingdb_by_uniprot(uniprot_id, cfg):
    url = "https://www.bindingdb.org/rest/getLigandsByUniprots"

    params = {
        "uniprot": uniprot_id,
        "cutoff": cfg.cutoff,
        "response": "application/json",
    }

    headers = {
        "User-Agent": "BindingDBDashboard/1.0",
    }

    for attempt in range(1, cfg.retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=(cfg.connect_timeout_sec, cfg.read_timeout_sec),
            )

            response.raise_for_status()

            data = response.json()

            root_key = next(iter(data.keys()))
            resp = data[root_key]

            possible_keys = [
                "bdb.affinities",
                "affinities",
                "affinity",
            ]

            rows = None

            for key in possible_keys:
                if key in resp:
                    rows = resp[key]
                    break

            if rows is None:
                raise KeyError(
                    f"Cannot find affinities in response. Keys={list(resp.keys())}"
                )

            if isinstance(rows, dict):
                if "affinity" in rows:
                    rows = rows["affinity"]
                elif "bdb.affinity" in rows:
                    rows = rows["bdb.affinity"]

            if not isinstance(rows, list):
                rows = [rows]

            df = pd.DataFrame(rows)

            time.sleep(cfg.pause_sec)

            return df

        except Exception as e:
            wait = min(8.0, 0.5 * (2 ** (attempt - 1)))
            st.info(
                f"BindingDB request failed for {uniprot_id} "
                f"({attempt}/{cfg.retries}): {e}. Retrying in {wait:.1f}s."
            )
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch BindingDB data for {uniprot_id}.")


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
        if st.button(
            "Clear",
            key="tab4_clear_uniprot_query",
            use_container_width=True,
        ):
            st.session_state["tab4_uniprot_id"] = ""
            st.rerun()

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

    thick_divider()


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

    thick_divider()

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
# Display modes
# ============================================================
def aggregate_duplicate_values(df, mol_id, activity_col, method="mean"):
    df = df.copy()

    if mol_id not in df.columns:
        return df

    if activity_col not in df.columns:
        return df.drop_duplicates(subset=[mol_id], keep="first")

    df[activity_col] = pd.to_numeric(df[activity_col], errors="coerce")

    if method == "mean":
        agg_values = (
            df.groupby(mol_id, as_index=False)[activity_col]
            .mean()
            .rename(columns={activity_col: f"{activity_col}_mean"})
        )
        agg_col = f"{activity_col}_mean"

    elif method == "median":
        agg_values = (
            df.groupby(mol_id, as_index=False)[activity_col]
            .median()
            .rename(columns={activity_col: f"{activity_col}_median"})
        )
        agg_col = f"{activity_col}_median"

    else:
        return df

    base_df = df.drop_duplicates(subset=[mol_id], keep="first").copy()
    base_df = base_df.drop(columns=[activity_col], errors="ignore")

    output_df = pd.merge(base_df, agg_values, on=mol_id, how="left")

    output_df[activity_col] = output_df[agg_col]
    output_df = output_df.drop(columns=[agg_col], errors="ignore")
    output_df["aggregation_method"] = method

    output_df = add_quality_flags(
        output_df,
        mol_id=mol_id,
        activity_col=activity_col,
    )

    return output_df


def display_mode(bioactivity_df):
    ligand_id = st.session_state.get("tab4_ligand_id_for_display", "monomerid")
    activity_col = st.session_state.get("tab4_standard_activity_col", "standard_value")

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
        key="tab4_duplicate_mode",
    )

    display_df = bioactivity_df.copy()

    if duplicate_mode == "Show duplicates first":
        display_df = display_df.sort_values(
            by=["flag_duplicate_molecule", ligand_id],
            ascending=[False, True],
        )

    elif duplicate_mode == "Keep first only":
        display_df = display_df.drop_duplicates(
            subset=[ligand_id],
            keep="first",
        )

    elif duplicate_mode == "Keep best potency only":
        if activity_col in display_df.columns:
            display_df = (
                display_df.sort_values(activity_col, ascending=True)
                .drop_duplicates(subset=[ligand_id], keep="first")
            )

    elif duplicate_mode == "Use mean value":
        display_df = aggregate_duplicate_values(
            display_df,
            mol_id=ligand_id,
            activity_col=activity_col,
            method="mean",
        )

    elif duplicate_mode == "Use median value":
        display_df = aggregate_duplicate_values(
            display_df,
            mol_id=ligand_id,
            activity_col=activity_col,
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
    display_df = st.session_state.get("tab4_display_df")

    if display_df is None or display_df.empty:
        st.warning("No displayed table to save.")
        return

    save_df = display_df.drop(
        columns=["_row_id", "_delete"],
        errors="ignore",
    )

    db_path = Path("data") / "bindingdb_bioactivity.db"

    if st.button(
        "Save SQL",
        use_container_width=True,
        key="tab4_save_sql",
    ):
        save_df_to_sql(
            df=save_df,
            table_name="bindingdb_bioactivity",
            db_path=db_path,
        )

        st.success(f"Saved current displayed table to: {db_path}")


# ============================================================
# Bioactivity table
# ============================================================
def show_bioactivity_table():
    bioactivity_df = st.session_state.get("tab4_bioactivity_df")

    if bioactivity_df is None or bioactivity_df.empty:
        return

    st.header("Bioactivity Data")

    ligand_id = st.session_state.get("tab4_ligand_id_for_display", "monomerid")
    activity_col = st.session_state.get("tab4_standard_activity_col", "standard_value")

    bioactivity_df = add_quality_flags(
        bioactivity_df,
        mol_id=ligand_id,
        activity_col=activity_col,
    )

    if "_row_id" not in bioactivity_df.columns:
        bioactivity_df["_row_id"] = range(len(bioactivity_df))
        st.session_state["tab4_bioactivity_df"] = bioactivity_df.copy()

    display_df = display_mode(bioactivity_df)
    display_df = display_df.reset_index(drop=True)

    st.session_state["tab4_display_df"] = display_df.copy()

    st.info(f"Current display: {len(display_df)} row(s)")

    if "_delete" not in display_df.columns:
        display_df.insert(0, "_delete", False)

    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="dynamic",
        key="tab4_bioactivity_editor",
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
            key="tab4_delete_checked_rows",
        ):
            deleted = edited_df[edited_df["_delete"] == True].copy()

            if deleted.empty:
                st.warning("No rows checked for deletion.")
                return

            deleted_row_ids = deleted["_row_id"].tolist()

            master_df = st.session_state["tab4_bioactivity_df"].copy()

            deleted_from_master = master_df[
                master_df["_row_id"].isin(deleted_row_ids)
            ].copy()

            kept_master = master_df[
                ~master_df["_row_id"].isin(deleted_row_ids)
            ].copy()

            st.session_state["tab4_deleted_rows_df"] = pd.concat(
                [
                    st.session_state["tab4_deleted_rows_df"],
                    deleted_from_master,
                ],
                ignore_index=True,
            )

            kept_master = add_quality_flags(
                kept_master,
                mol_id=ligand_id,
                activity_col=activity_col,
            )

            st.session_state["tab4_bioactivity_df"] = kept_master.copy()

            st.success(f"Deleted {len(deleted_row_ids)} row(s).")
            st.rerun()

    with c2:
        if st.button(
            "Clear History",
            use_container_width=True,
            key="tab4_clear_deleted_history",
        ):
            st.session_state["tab4_deleted_rows_df"] = pd.DataFrame()
            st.success("Deleted-row history cleared.")
            st.rerun()

    with c3:
        if st.button(
            "Restore Original",
            use_container_width=True,
            key="tab4_restore_original",
        ):
            original_df = st.session_state.get("tab4_original_bioactivity_df")

            if original_df is None or original_df.empty:
                st.warning("No original table to restore.")
            else:
                st.session_state["tab4_bioactivity_df"] = original_df.copy()
                st.session_state["tab4_display_df"] = original_df.copy()
                st.session_state["tab4_deleted_rows_df"] = pd.DataFrame()
                st.success("Original table restored.")
                st.rerun()

    with c4:
        save_data()

    thick_divider()

    event = st.dataframe(
        display_df,
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        key="tab4_bioactivity_table",
    )

    selected_rows = event.selection.rows

    if selected_rows:
        row_idx = selected_rows[0]
        selected_row = display_df.iloc[row_idx]

        selected_ligand = selected_row.get(ligand_id, "Unknown")
        smiles = selected_row.get("SMILES")

        st.subheader("Selected Molecule")
        st.write("Selected ligand:", selected_ligand)

        if pd.notna(smiles):
            mol = Chem.MolFromSmiles(str(smiles))

            if mol is not None:
                st.image(
                    Draw.MolToImage(mol, size=(350, 300)),
                    caption=str(selected_ligand),
                )
            else:
                st.warning("Invalid SMILES.")
        else:
            st.warning("No SMILES found for this molecule.")

        st.write(selected_row)

        if st.button(
            "Delete Selected Row",
            use_container_width=True,
            key="tab4_delete_selected_row",
        ):
            row_id = selected_row["_row_id"]

            master_df = st.session_state["tab4_bioactivity_df"].copy()

            deleted_row = master_df[master_df["_row_id"] == row_id].copy()
            updated_df = master_df[master_df["_row_id"] != row_id].copy()

            st.session_state["tab4_deleted_rows_df"] = pd.concat(
                [
                    st.session_state["tab4_deleted_rows_df"],
                    deleted_row,
                ],
                ignore_index=True,
            )

            updated_df = add_quality_flags(
                updated_df,
                mol_id=ligand_id,
                activity_col=activity_col,
            )

            st.session_state["tab4_bioactivity_df"] = updated_df.copy()

            st.success("Selected row deleted.")
            st.rerun()

    deleted_rows_df = st.session_state.get("tab4_deleted_rows_df")

    if deleted_rows_df is not None and not deleted_rows_df.empty:
        with st.expander("Deleted rows history", expanded=False):
            st.dataframe(
                deleted_rows_df.drop(columns=["_delete"], errors="ignore"),
                use_container_width=True,
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

    show_bioactivity_table()