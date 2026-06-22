from src.utils.select_dir import directory_picker
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

import numpy as np

import streamlit as st


PLOT_OPTIONS = [
    "bar",
    "radar",
    "distribution",
    "confusion matrix",
    "ROC-AUC",
]


def init_plot_state():
    if "plot_blocks" not in st.session_state:
        st.session_state["plot_blocks"] = []


def add_plot_block():
    st.session_state["plot_blocks"].append({
        "id": len(st.session_state["plot_blocks"]),
    })


def remove_plot_block(plot_id):
    st.session_state["plot_blocks"] = [
        p for p in st.session_state["plot_blocks"]
        if p["id"] != plot_id
    ]

def select_column_pair(results, plot_id, x_label="X Column", y_label="Y Column"):
    c1, c2 = st.columns(2)

    with c1:
        x_col = st.selectbox(
            x_label,
            results.columns,
            key=f"x_col_{plot_id}",
        )

    with c2:
        y_col = st.selectbox(
            y_label,
            results.columns,
            key=f"y_col_{plot_id}",
        )

    return x_col, y_col

def has_roc_data(df):
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

def render_plot_block(results, model_col, plot_id):
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

        row = results[
            results[model_col].astype(str) == selected_model
        ]

        if row.empty:
            st.warning("No matching model found.")
            return

        row = row.iloc[0]

        if plot_type == "radar":
            metric_cols = st.multiselect(
                "Metrics",
                results.columns,
                key=f"radar_metrics_{plot_id}",
            )

            radar_data = {}

            for col in metric_cols:
                try:
                    radar_data[col] = float(row[col])
                except Exception:
                    pass

            if len(radar_data) >= 3:
                from src.plot import plot_radar_chart

                plot_radar_chart(
                    data=radar_data,
                    name=selected_model,
                    key=f"radar_{plot_id}",
                )
            else:
                st.info("Select at least 3 numeric metrics.")

        # elif plot_type == "bar":
        #     metric_cols = st.multiselect(
        #         "Metrics",
        #         results.columns,
        #         key=f"bar_metrics_{plot_id}",
        #     )

        #     if metric_cols:
        #         bar_data = {}

        #         for col in metric_cols:
        #             try:
        #                 bar_data[col] = float(row[col])
        #             except Exception:
        #                 pass

        #         st.bar_chart(bar_data)

        # elif plot_type == "distribution":
        #     metric_col = st.selectbox(
        #         "Metric",
        #         results.columns,
        #         key=f"dist_metric_{plot_id}",
        #     )

        #     try:
        #         values = results[metric_col].astype(float)
        #         st.line_chart(values)
        #     except Exception:
        #         st.warning(f"{metric_col} is not numeric.")

        # elif plot_type == "confusion matrix":
        #     if "confusion_matrix" not in results.columns:
        #         st.warning("No confusion_matrix column found.")
        #         return

        #     from src.plot import plot_confusion_matrix_from_row

        #     plot_confusion_matrix_from_row(
        #         row=row,
        #         key=f"cm_{plot_id}",
        #     )

        elif plot_type == "ROC-AUC":
            if not has_roc_data(results):
                st.warning("No ROC curve data found.")
                return

            selected_model = st.selectbox(
                "Model",
                results[model_col].astype(str).unique(),
                key=f"roc_model_{plot_id}",
            )

            row = results[
                results[model_col].astype(str) == selected_model
            ].iloc[0]

            from src.plot import plot_roc_auc

            plot_roc_auc(
                results=row.to_dict(),
                key=f"roc_{plot_id}",
            )


def plot_builder(results, model_col):
    init_plot_state()

    if st.button("+ Add Plot", type="primary", use_container_width=True):
        add_plot_block()
        st.rerun()

    for plot_block in st.session_state["plot_blocks"]:
        render_plot_block(
            results=results,
            model_col=model_col,
            plot_id=plot_block["id"],
        )

def extract_model_summaries(json_dir, output_tsv):
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

            # Keep nested confusion matrix as object if present
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


def design(workdir):
    if workdir is None:
        workdir = Path.cwd()
    else:
        workdir = Path(workdir).expanduser().resolve()

    result_dir = directory_picker(
        label="Select result directory",
        start_dir=workdir,
        key="result_dir_picker",
    )

    outdir = Path(workdir) / "data"
    outdir.mkdir(parents=True, exist_ok=True)

    output_file = outdir / "summary_results.tsv"

    if st.button("Extract summaries", type="primary", use_container_width=True):
        results = extract_model_summaries(
            json_dir=result_dir,
            output_tsv=output_file,
        )

        st.session_state["model_summary_results"] = results
        st.success(f"Saved summary results to: {output_file}")

    if "model_summary_results" not in st.session_state:
        st.info("Please extract summaries first.")
        return

    results = st.session_state["model_summary_results"]

    st.dataframe(results, use_container_width=True)
    st.divider()

    c1, c2 = st.columns(2, vertical_alignment="bottom")

    with c1:
        task_type = st.selectbox(
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


    # if task_type == "regression":
    #     pass
    # elif task_type == "classification (binary)":
    #     pass
    # elif task_type == "classification (multi-class)":
    #     pass