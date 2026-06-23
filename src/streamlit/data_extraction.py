# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw


# ============================================================
# SQL helper
# ============================================================

def save_df_to_sql(df: pd.DataFrame, table_name: str, db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


# ============================================================
# Data quality helpers
# ============================================================

def add_quality_flags(
    df: pd.DataFrame,
    mol_id: str = "CID",
    activity_col: str | None = None,
) -> pd.DataFrame:
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


def aggregate_duplicate_values(
    df: pd.DataFrame,
    mol_id: str,
    activity_col: str | None = None,
    method: str = "mean",
) -> pd.DataFrame:
    df = df.copy()

    if mol_id not in df.columns:
        return df

    if activity_col is None or activity_col not in df.columns:
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

    output_df = pd.merge(
        base_df,
        agg_values,
        on=mol_id,
        how="left",
    )

    output_df[activity_col] = output_df[agg_col]
    output_df = output_df.drop(columns=[agg_col], errors="ignore")
    output_df["aggregation_method"] = method

    output_df = add_quality_flags(
        output_df,
        mol_id=mol_id,
        activity_col=activity_col,
    )

    return output_df


# ============================================================
# Display mode
# ============================================================

def display_mode(
    bioactivity_df: pd.DataFrame,
    ligand_id_col: str,
    activity_col: str | None = None,
    key: str = "bioactivity_display_mode",
) -> pd.DataFrame:
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
        key=key,
    )

    display_df = bioactivity_df.copy()

    if duplicate_mode == "Show duplicates first":
        if "flag_duplicate_molecule" in display_df.columns:
            display_df = display_df.sort_values(
                by=["flag_duplicate_molecule", ligand_id_col],
                ascending=[False, True],
            )

    elif duplicate_mode == "Keep first only":
        display_df = display_df.drop_duplicates(
            subset=[ligand_id_col],
            keep="first",
        )

    elif duplicate_mode == "Keep best potency only":
        if activity_col and activity_col in display_df.columns:
            display_df[activity_col] = pd.to_numeric(
                display_df[activity_col],
                errors="coerce",
            )
            display_df = (
                display_df.sort_values(activity_col, ascending=True)
                .drop_duplicates(subset=[ligand_id_col], keep="first")
            )
        else:
            st.warning("No activity column selected for potency sorting.")

    elif duplicate_mode == "Use mean value":
        display_df = aggregate_duplicate_values(
            display_df,
            mol_id=ligand_id_col,
            activity_col=activity_col,
            method="mean",
        )

    elif duplicate_mode == "Use median value":
        display_df = aggregate_duplicate_values(
            display_df,
            mol_id=ligand_id_col,
            activity_col=activity_col,
            method="median",
        )

    elif duplicate_mode == "Show duplicates only":
        if "flag_duplicate_molecule" in display_df.columns:
            display_df = display_df[
                display_df["flag_duplicate_molecule"] == True
            ]

    return display_df


# ============================================================
# Save current display
# ============================================================

def save_data(
    state_df_name: str,
    button_key: str,
    table_name: str = "bioactivity",
    db_path: str | Path = Path("data") / "bioactivity.db",
) -> None:
    display_df = st.session_state.get(state_df_name)

    if display_df is None or display_df.empty:
        st.warning("No displayed table to save.")
        return

    save_df = display_df.drop(
        columns=["_row_id", "_delete"],
        errors="ignore",
    )

    if st.button(
        "Save SQL",
        use_container_width=True,
        key=button_key,
    ):
        save_df_to_sql(
            df=save_df,
            table_name=table_name,
            db_path=db_path,
        )

        st.success(f"Saved current displayed table to: {db_path}")


# ============================================================
# Main bioactivity table
# ============================================================

def show_bioactivity_table(
    ligand_id_col: str,
    state_original_bioactivity_df_name: str,
    state_bioactivity_df_name: str,
    state_activity_col_name: str,
    state_display_df_name: str,
    state_deleted_rows_df_name: str,
    display_key: str,
    deleted_selected_row_button_key: str,
    deleted_checked_rows_button_key: str,
    clear_history_button_key: str,
    restore_button_key: str,
    save_button_key: str,
    bioactivity_table_key: str,
    editor_key: str,
) -> None:
    bioactivity_df = st.session_state.get(state_bioactivity_df_name)

    if bioactivity_df is None or bioactivity_df.empty:
        return

    st.header("Bioactivity Data")

    activity_col = st.session_state.get(state_activity_col_name)

    bioactivity_df = add_quality_flags(
        bioactivity_df,
        mol_id=ligand_id_col,
        activity_col=activity_col,
    )

    if "_row_id" not in bioactivity_df.columns:
        bioactivity_df["_row_id"] = range(len(bioactivity_df))

    st.session_state[state_bioactivity_df_name] = bioactivity_df.copy()

    display_df = display_mode(
        bioactivity_df=bioactivity_df,
        ligand_id_col=ligand_id_col,
        activity_col=activity_col,
        key=display_key,
    )

    display_df = display_df.reset_index(drop=True)
    st.session_state[state_display_df_name] = display_df.copy()

    st.info(f"Current display: {len(display_df)} row(s)")

    editor_df = display_df.copy()

    if "_delete" not in editor_df.columns:
        editor_df.insert(0, "_delete", False)

    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        num_rows="dynamic",
        key=editor_key,
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
            key=deleted_checked_rows_button_key,
        ):
            deleted = edited_df[edited_df["_delete"] == True].copy()

            if deleted.empty:
                st.warning("No rows checked for deletion.")
                return

            deleted_row_ids = deleted["_row_id"].tolist()

            master_df = st.session_state[state_bioactivity_df_name].copy()

            deleted_from_master = master_df[
                master_df["_row_id"].isin(deleted_row_ids)
            ].copy()

            kept_master = master_df[
                ~master_df["_row_id"].isin(deleted_row_ids)
            ].copy()

            old_deleted = st.session_state.get(
                state_deleted_rows_df_name,
                pd.DataFrame(),
            )

            st.session_state[state_deleted_rows_df_name] = pd.concat(
                [old_deleted, deleted_from_master],
                ignore_index=True,
            )

            kept_master = add_quality_flags(
                kept_master,
                mol_id=ligand_id_col,
                activity_col=activity_col,
            )

            st.session_state[state_bioactivity_df_name] = kept_master.copy()

            st.success(f"Deleted {len(deleted_row_ids)} row(s).")
            st.rerun()

    with c2:
        if st.button(
            "Clear History",
            use_container_width=True,
            key=clear_history_button_key,
        ):
            st.session_state[state_deleted_rows_df_name] = pd.DataFrame()
            st.success("Deleted-row history cleared.")
            st.rerun()

    with c3:
        if st.button(
            "Restore Original",
            use_container_width=True,
            key=restore_button_key,
        ):
            original_df = st.session_state.get(state_original_bioactivity_df_name)

            if original_df is None or original_df.empty:
                st.warning("No original table to restore.")
            else:
                st.session_state[state_bioactivity_df_name] = original_df.copy()
                st.session_state[state_display_df_name] = original_df.copy()
                st.session_state[state_deleted_rows_df_name] = pd.DataFrame()
                st.success("Original table restored.")
                st.rerun()

    with c4:
        save_data(
            state_df_name=state_display_df_name,
            button_key=save_button_key,
        )

    event = st.dataframe(
        display_df,
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        key=bioactivity_table_key,
    )

    selected_rows = event.selection.rows

    if selected_rows:
        row_idx = selected_rows[0]
        selected_row = display_df.iloc[row_idx]

        ligand_id = selected_row.get(ligand_id_col, "Unknown")
        smiles = selected_row.get("SMILES")

        st.subheader("Selected Molecule")
        st.write("Selected Ligand ID:", ligand_id)

        if pd.notna(smiles):
            mol = Chem.MolFromSmiles(str(smiles))

            if mol is not None:
                st.image(
                    Draw.MolToImage(mol, size=(350, 300)),
                    caption=f"Ligand ID {ligand_id}",
                )
            else:
                st.warning("Invalid SMILES.")
        else:
            st.warning("No SMILES found for this molecule.")

        st.write(selected_row)

        if st.button(
            "Delete Selected Row",
            use_container_width=True,
            key=deleted_selected_row_button_key,
        ):
            row_id = selected_row["_row_id"]

            master_df = st.session_state[state_bioactivity_df_name].copy()

            deleted_row = master_df[
                master_df["_row_id"] == row_id
            ].copy()

            updated_df = master_df[
                master_df["_row_id"] != row_id
            ].copy()

            old_deleted = st.session_state.get(
                state_deleted_rows_df_name,
                pd.DataFrame(),
            )

            st.session_state[state_deleted_rows_df_name] = pd.concat(
                [old_deleted, deleted_row],
                ignore_index=True,
            )

            updated_df = add_quality_flags(
                updated_df,
                mol_id=ligand_id_col,
                activity_col=activity_col,
            )

            st.session_state[state_bioactivity_df_name] = updated_df.copy()

            st.success("Selected row deleted.")
            st.rerun()

    deleted_rows_df = st.session_state.get(state_deleted_rows_df_name)

    if deleted_rows_df is not None and not deleted_rows_df.empty:
        with st.expander("Deleted rows history", expanded=False):
            st.dataframe(
                deleted_rows_df.drop(columns=["_delete"], errors="ignore"),
                use_container_width=True,
            )