# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import io
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors


BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


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


@dataclass
class PipelineConfig:
    pause_sec: float = 0.25
    batch_size: int = 200
    properties: str = "CanonicalSMILES,IsomericSMILES,InChI,InChIKey"
    retries: int = 4
    timeout_sec: int = 180
    mol_id: str = "CID"


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


def run_request(url, retries=4, timeout_sec=180, pause_sec=0.25):
    headers = {"User-Agent": "PubChemPipeline/1.0"}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout_sec,
            )
            response.raise_for_status()

            if not response.text.strip():
                raise RuntimeError("Empty PubChem response.")

            time.sleep(pause_sec)
            return response.text

        except Exception as e:
            wait = min(8.0, 0.5 * (2 ** (attempt - 1)))
            st.info(
                f"Request failed ({attempt}/{retries}): {e}. "
                f"Retrying in {wait:.1f}s."
            )
            time.sleep(wait)

    raise RuntimeError(f"Request failed after {retries} attempts: {url}")


def chunked(xs: List[int], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


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


def fetch_pubchem_target_assays(uniprot_id, cfg):
    url = f"{BASE}/assay/target/accession/{uniprot_id}/concise/CSV"

    text = run_request(
        url,
        retries=cfg.retries,
        timeout_sec=cfg.timeout_sec,
        pause_sec=cfg.pause_sec,
    )

    df = pd.read_csv(io.StringIO(text))

    if "Target Accession" in df.columns:
        df = df[df["Target Accession"].astype(str).str.strip() == uniprot_id].copy()

    return df


def fetch_structure_on_cid(cfg: PipelineConfig, cids: Iterable[int]) -> pd.DataFrame:
    cid_list = sorted({int(c) for c in cids if pd.notna(c)})

    if not cid_list:
        return pd.DataFrame(columns=["CID"] + cfg.properties.split(","))

    frames = []

    for batch in chunked(cid_list, cfg.batch_size):
        cid_str = ",".join(map(str, batch))

        url = (
            f"{BASE}/compound/cid/"
            f"{cid_str}/property/"
            f"{cfg.properties}/CSV"
        )

        text = run_request(
            url,
            retries=cfg.retries,
            timeout_sec=cfg.timeout_sec,
            pause_sec=cfg.pause_sec,
        )

        df_props = pd.read_csv(io.StringIO(text))

        df_props["CID"] = pd.to_numeric(
            df_props["CID"],
            errors="coerce",
        ).astype("Int64")

        frames.append(df_props)

    out = pd.concat(frames, ignore_index=True).dropna(subset=["CID"])
    out["CID"] = out["CID"].astype(int)
    out = out.drop_duplicates(subset=["CID"])

    return out


def add_quality_flags(df, mol_id="CID", activity_col=None):
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


def target_query_by_uniprot(cfg):
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
        if st.button(
            "Clear",
            key="tab3_clear_uniprot_query",
            use_container_width=True,
        ):
            st.session_state["tab3_uniprot_id"] = ""
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

    thick_divider()


def set_bioactivity_query():
    df = st.session_state.get("tab3_targets_df")

    if df is None or df.empty:
        return None, None, None, None

    query_list = df.columns.tolist()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        default_fields = [
            col for col in [
                "CID",
                "Activity Name",
                "Activity Value [nM]",
                "Activity Value",
                "Activity Unit",
                "Target Accession",
                "Target Name",
                "query_uniprot",
            ]
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
            options=["IC50", "EC50", "Ki", "Kd", "All"],
            key="tab3_query_type",
        )

    with c3:
        numeric_candidates = [
            c for c in query_list
            if "value" in c.lower() or "[nm]" in c.lower() or "[um]" in c.lower()
        ]

        activity_col = st.selectbox(
            "Activity Value Column",
            options=query_list,
            index=query_list.index(numeric_candidates[0])
            if numeric_candidates and numeric_candidates[0] in query_list
            else 0,
            key="tab3_activity_col_select",
        )

    with c4:
        drop_na_col = st.multiselect(
            "Remove NaN From",
            options=query_fields,
            default=[c for c in ["CID", activity_col] if c in query_fields],
            key="tab3_drop_na_col",
        )

    st.session_state["tab3_activity_col"] = activity_col

    if "Activity Name" in query_list:
        st.session_state["tab3_activity_name_col"] = "Activity Name"

    thick_divider()

    return query_fields, query_type, drop_na_col, activity_col


def get_bioactivity_data(
    query_fields,
    query_type,
    drop_na_col,
    activity_col,
    cfg,
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

            if activity_col in bioactivity_df.columns:
                bioactivity_df[activity_col] = pd.to_numeric(
                    bioactivity_df[activity_col],
                    errors="coerce",
                )

            bioactivity_df["standard_value"] = bioactivity_df[activity_col] * 1000
            bioactivity_df["standard_units"] = "nM"

            if drop_na_col:
                bioactivity_df = bioactivity_df.dropna(subset=drop_na_col).copy()

            if cfg.mol_id not in bioactivity_df.columns:
                st.error(f"Missing required compound ID column: {cfg.mol_id}")
                status.update(label="Missing CID", state="error")
                return

            bioactivity_df[cfg.mol_id] = pd.to_numeric(
                bioactivity_df[cfg.mol_id],
                errors="coerce",
            )

            bioactivity_df = bioactivity_df.dropna(subset=[cfg.mol_id]).copy()
            bioactivity_df[cfg.mol_id] = bioactivity_df[cfg.mol_id].astype(int)

            status.write("Downloading PubChem structures...")

            compound_ids = bioactivity_df[cfg.mol_id].tolist()

            compound_df = fetch_structure_on_cid(
                cfg=cfg,
                cids=compound_ids,
            )

            output_df = bioactivity_df.merge(
                compound_df,
                on=cfg.mol_id,
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
                mol_id=cfg.mol_id,
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


def aggregate_duplicate_values(df, mol_id="CID", activity_col=None, method="mean"):
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


def display_mode(bioactivity_df, cfg):
    activity_col = st.session_state.get("tab3_activity_col")

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
        key="tab3_duplicate_mode",
    )

    display_df = bioactivity_df.copy()

    if duplicate_mode == "Show duplicates first":
        display_df = display_df.sort_values(
            by=["flag_duplicate_molecule", cfg.mol_id],
            ascending=[False, True],
        )

    elif duplicate_mode == "Keep first only":
        display_df = display_df.drop_duplicates(
            subset=[cfg.mol_id],
            keep="first",
        )

    elif duplicate_mode == "Keep best potency only":
        if activity_col in display_df.columns:
            display_df = (
                display_df.sort_values(activity_col, ascending=True)
                .drop_duplicates(subset=[cfg.mol_id], keep="first")
            )

    elif duplicate_mode == "Use mean value":
        display_df = aggregate_duplicate_values(
            display_df,
            mol_id=cfg.mol_id,
            activity_col=activity_col,
            method="mean",
        )

    elif duplicate_mode == "Use median value":
        display_df = aggregate_duplicate_values(
            display_df,
            mol_id=cfg.mol_id,
            activity_col=activity_col,
            method="median",
        )

    elif duplicate_mode == "Show duplicates only":
        display_df = display_df[
            display_df["flag_duplicate_molecule"] == True
        ]

    return display_df


def save_data():
    display_df = st.session_state.get("tab3_display_df")

    if display_df is None or display_df.empty:
        st.warning("No displayed table to save.")
        return

    save_df = display_df.drop(
        columns=["_row_id", "_delete"],
        errors="ignore",
    )

    db_path = Path("data") / "pubchem_bioactivity.db"

    if st.button(
        "Save SQL",
        use_container_width=True,
        key="tab3_save_sql",
    ):
        save_df_to_sql(
            df=save_df,
            table_name="pubchem_bioactivity",
            db_path=db_path,
        )

        st.success(f"Saved current displayed table to: {db_path}")


def show_bioactivity_table(cfg):
    bioactivity_df = st.session_state.get("tab3_bioactivity_df")

    if bioactivity_df is None or bioactivity_df.empty:
        return

    st.header("Bioactivity Data")

    activity_col = st.session_state.get("tab3_activity_col")

    bioactivity_df = add_quality_flags(
        bioactivity_df,
        mol_id=cfg.mol_id,
        activity_col=activity_col,
    )

    if "_row_id" not in bioactivity_df.columns:
        bioactivity_df["_row_id"] = range(len(bioactivity_df))
        st.session_state["tab3_bioactivity_df"] = bioactivity_df.copy()

    display_df = display_mode(bioactivity_df, cfg)
    display_df = display_df.reset_index(drop=True)

    st.session_state["tab3_display_df"] = display_df.copy()

    st.info(f"Current display: {len(display_df)} row(s)")

    if "_delete" not in display_df.columns:
        display_df.insert(0, "_delete", False)

    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="dynamic",
        key="tab3_bioactivity_editor",
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
            key="tab3_delete_checked_rows",
        ):
            deleted = edited_df[edited_df["_delete"] == True].copy()

            if deleted.empty:
                st.warning("No rows checked for deletion.")
                return

            deleted_row_ids = deleted["_row_id"].tolist()

            master_df = st.session_state["tab3_bioactivity_df"].copy()

            deleted_from_master = master_df[
                master_df["_row_id"].isin(deleted_row_ids)
            ].copy()

            kept_master = master_df[
                ~master_df["_row_id"].isin(deleted_row_ids)
            ].copy()

            st.session_state["tab3_deleted_rows_df"] = pd.concat(
                [
                    st.session_state["tab3_deleted_rows_df"],
                    deleted_from_master,
                ],
                ignore_index=True,
            )

            kept_master = add_quality_flags(
                kept_master,
                mol_id=cfg.mol_id,
                activity_col=activity_col,
            )

            st.session_state["tab3_bioactivity_df"] = kept_master.copy()

            st.success(f"Deleted {len(deleted_row_ids)} row(s).")
            st.rerun()

    with c2:
        if st.button(
            "Clear History",
            use_container_width=True,
            key="tab3_clear_deleted_history",
        ):
            st.session_state["tab3_deleted_rows_df"] = pd.DataFrame()
            st.success("Deleted-row history cleared.")
            st.rerun()

    with c3:
        if st.button(
            "Restore Original",
            use_container_width=True,
            key="tab3_restore_original",
        ):
            original_df = st.session_state.get("tab3_original_bioactivity_df")

            if original_df is None or original_df.empty:
                st.warning("No original table to restore.")
            else:
                st.session_state["tab3_bioactivity_df"] = original_df.copy()
                st.session_state["tab3_display_df"] = original_df.copy()
                st.session_state["tab3_deleted_rows_df"] = pd.DataFrame()
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
        key="tab3_bioactivity_table",
    )

    selected_rows = event.selection.rows

    if selected_rows:
        row_idx = selected_rows[0]
        selected_row = display_df.iloc[row_idx]

        cid = selected_row.get("CID", "Unknown")
        smiles = selected_row.get("SMILES")

        st.subheader("Selected Molecule")
        st.write("Selected CID:", cid)

        if pd.notna(smiles):
            mol = Chem.MolFromSmiles(str(smiles))

            if mol is not None:
                st.image(
                    Draw.MolToImage(mol, size=(350, 300)),
                    caption=f"CID {cid}",
                )
            else:
                st.warning("Invalid SMILES.")
        else:
            st.warning("No SMILES found for this molecule.")

        st.write(selected_row)

        if st.button(
            "Delete Selected Row",
            use_container_width=True,
            key="tab3_delete_selected_row",
        ):
            row_id = selected_row["_row_id"]

            master_df = st.session_state["tab3_bioactivity_df"].copy()

            deleted_row = master_df[
                master_df["_row_id"] == row_id
            ].copy()

            updated_df = master_df[
                master_df["_row_id"] != row_id
            ].copy()

            st.session_state["tab3_deleted_rows_df"] = pd.concat(
                [
                    st.session_state["tab3_deleted_rows_df"],
                    deleted_row,
                ],
                ignore_index=True,
            )

            updated_df = add_quality_flags(
                updated_df,
                mol_id=cfg.mol_id,
                activity_col=activity_col,
            )

            st.session_state["tab3_bioactivity_df"] = updated_df.copy()

            st.success("Selected row deleted.")
            st.rerun()

    deleted_rows_df = st.session_state.get("tab3_deleted_rows_df")

    if deleted_rows_df is not None and not deleted_rows_df.empty:
        with st.expander("Deleted rows history", expanded=False):
            st.dataframe(
                deleted_rows_df.drop(columns=["_delete"], errors="ignore"),
                use_container_width=True,
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

    show_bioactivity_table(cfg)