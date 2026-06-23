# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.select_dir import directory_picker
from src.streamlit.plot_widgets import (
    st_radar_chart,
    st_roc_auc_plot,
    st_distribution_with_boxplot,
    st_metric_bar_plot,
)


PLOT_OPTIONS = [
    "bar",
    "radar",
    "distribution",
    "ROC-AUC",
]


# ============================================================
# Session state
# ============================================================

def init_plot_state() -> None:
    if "plot_blocks" not in st.session_state:
        st.session_state["plot_blocks"] = []


def add_plot_block() -> None:
    existing_ids = [p["id"] for p in st.session_state["plot_blocks"]]
    new_id = max(existing_ids) + 1 if existing_ids else 0

    st.session_state["plot_blocks"].append(
        {
            "id": new_id,
        }
    )


def remove_plot_block(plot_id: int) -> None:
    st.session_state["plot_blocks"] = [
        p
        for p in st.session_state["plot_blocks"]
        if p["id"] != plot_id
    ]


# ============================================================
# Helpers
# ============================================================

def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = []

    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors="raise")
            numeric_cols.append(col)
        except Exception:
            continue

    return numeric_cols


def has_roc_data(df: pd.DataFrame) -> bool:
    cols = [str(c) for c in df.columns]

    if "roc_curve" in cols:
        return True

    if "roc_curves_ovr" in cols:
        return True

    if any(c.startswith("roc_curves_ovr.") and c.endswith(".fpr") for c in cols):
        return True

    if any(c.startswith("roc_curve.") and c.endswith(".fpr") for c in cols):
        return True

    return False


def extract_model_summaries(
    json_dir: str | Path,
    output_tsv: str | Path,
) -> pd.DataFrame:
    json_dir = Path(json_dir)
    output_tsv = Path(output_tsv)

    dfs = []

    for json_file in json_dir.rglob("*summary.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            df = pd.json_normalize(data)
            df["file"] = str(json_file)
            df["model_file"] = json_file.name

            if "confusion_matrix" in data:
                df["confusion_matrix"] = [data["confusion_matrix"]]

            if "confusion_matrix_labels" in data:
                df["confusion_matrix_labels"] = [data["confusion_matrix_labels"]]

            dfs.append(df)

        except Exception as e:
            st.warning(f"Failed to read {json_file}: {e}")

    if not dfs:
        return pd.DataFrame()

    results = pd.concat(dfs, ignore_index=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_tsv, sep="\t", index=False)

    return results


# ============================================================
# Plot block
# ============================================================

def render_plot_block(
    results: pd.DataFrame,
    model_col: str,
    plot_id: int,
) -> None:
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")

        with c1:
            plot_type = st.selectbox(
                "Plot Type",
                PLOT_OPTIONS,
                key=f"plot_type_{plot_id}",
            )

        with c2:
            selected_model = st.selectbox(
                "Model",
                results[model_col].astype(str).unique(),
                key=f"plot_model_{plot_id}",
            )

        with c3:
            if st.button("Remove", key=f"remove_plot_{plot_id}"):
                remove_plot_block(plot_id)
                st.rerun()

        row_df = results[
            results[model_col].astype(str) == str(selected_model)
        ]

        if row_df.empty:
            st.warning("No matching model found.")
            return

        row = row_df.iloc[0]
        if plot_type == "bar":

            numeric_cols = get_numeric_columns(results)

            metric_cols = st.multiselect(
                "Metrics",
                numeric_cols,
                key=f"bar_metrics_{plot_id}",
            )

            bar_data = {}

            for col in metric_cols:
                try:
                    bar_data[col] = float(row[col])
                except Exception:
                    pass

            if len(bar_data) > 0:
                st_metric_bar_plot(
                    metric_dict=bar_data,
                    key=f"bar_{plot_id}",
                )
            else:
                st.info("Select at least one numeric metric.")
        elif plot_type == "radar":
            numeric_cols = get_numeric_columns(results)

            metric_cols = st.multiselect(
                "Metrics",
                numeric_cols,
                key=f"radar_metrics_{plot_id}",
            )

            radar_data = {}

            for col in metric_cols:
                try:
                    value = float(row[col])
                    radar_data[col] = value
                except Exception:
                    continue

            if len(radar_data) >= 3:
                st_radar_chart(
                    data=radar_data,
                    name=str(selected_model),
                    key=f"radar_{plot_id}",
                )
            else:
                st.info("Select at least 3 numeric metrics.")

        elif plot_type == "distribution":
            numeric_cols = get_numeric_columns(results)

            if not numeric_cols:
                st.warning("No numeric columns found for distribution plot.")
                return

            metric_col = st.selectbox(
                "Metric",
                numeric_cols,
                key=f"dist_metric_{plot_id}",
            )

            st_distribution_with_boxplot(
                df=results,
                x_col=metric_col,
                key=f"dist_{plot_id}",
            )

        elif plot_type == "ROC-AUC":
            if not has_roc_data(results):
                st.warning("No ROC curve data found.")
                return

            st_roc_auc_plot(
                results=row.to_dict(),
                key=f"roc_{plot_id}",
            )


def plot_builder(
    results: pd.DataFrame,
    model_col: str,
) -> None:
    init_plot_state()

    if st.button(
        "+ Add Plot",
        type="primary",
        width="stretch",
    ):
        add_plot_block()
        st.rerun()

    for plot_block in st.session_state["plot_blocks"]:
        render_plot_block(
            results=results,
            model_col=model_col,
            plot_id=plot_block["id"],
        )


# ============================================================
# Main page
# ============================================================

def design(workdir: str | Path | None = None) -> None:
    if workdir is None:
        workdir = Path.cwd()
    else:
        workdir = Path(workdir).expanduser().resolve()

    result_dir = directory_picker(
        label="Select result directory",
        start_dir=workdir,
        key="result_dir_picker",
    )

    if not result_dir:
        st.info("Please select a result directory.")
        return

    outdir = Path(workdir) / "data"
    outdir.mkdir(parents=True, exist_ok=True)

    output_file = outdir / "summary_results.tsv"

    if st.button(
        "Extract summaries",
        type="primary",
        width="stretch",
    ):
        results = extract_model_summaries(
            json_dir=result_dir,
            output_tsv=output_file,
        )

        if results.empty:
            st.warning("No summary JSON files found.")
            return

        st.session_state["model_summary_results"] = results
        st.success(f"Saved summary results to: {output_file}")

    if "model_summary_results" not in st.session_state:
        st.info("Please extract summaries first.")
        return

    results = st.session_state["model_summary_results"]

    if results is None or results.empty:
        st.warning("Summary results are empty.")
        return

    st.dataframe(results, width="stretch")
    st.divider()

    c1, c2 = st.columns(2, vertical_alignment="bottom")

    with c1:
        st.selectbox(
            "Task Type",
            [
                "regression",
                "classification (binary)",
                "classification (multi-class)",
            ],
            index=0,
            key="analysis_task_type",
        )

    with c2:
        model_col = st.selectbox(
            "Model Name Column",
            results.columns,
            index=0,
            key="analysis_model_name_col",
        )

    st.divider()

    st.subheader("Analysis Settings")

    plot_builder(
        results=results,
        model_col=model_col,
    )