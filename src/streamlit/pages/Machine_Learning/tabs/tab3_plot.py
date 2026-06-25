# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from src.streamlit.utils.select_dir import directory_picker
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


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


def is_lower_better(metric):
    metric = metric.lower()
    return (
        metric.endswith(("rmse", "mae"))
        or metric == "runtime_seconds"
        or "error" in metric
        or "loss" in metric
    )


def get_default_metrics(results):
    numeric_cols = results.select_dtypes(include="number").columns.tolist()

    preferred = [
        "test_f1_macro",
        "test_roc_auc",
        "test_accuracy",
        "best_cv_f1",
        "best_cv_roc_auc",
        "best_cv_balanced_accuracy",
        "test_r2",
        "test_rmse",
        "test_mae",
        "best_cv_r2",
        "best_cv_root_mean_squared_error",
        "best_cv_mean_absolute_error",
        "runtime_seconds",
    ]

    return [c for c in preferred if c in numeric_cols]


def plot_one_metric(
    results,
    metric,
    color="#4C78A8",
    opacity=0.8,
    value_label_size=12,
    axis_label_size=12,
    title_size=16,
    tick_angle=-45,
    transparent_bg=True,
):
    plot_df = results.sort_values(
        metric,
        ascending=is_lower_better(metric),
    )

    fig = px.bar(
        plot_df,
        x="model_name",
        y=metric,
        text=metric,
        hover_data=plot_df.columns,
        title=metric,
    )

    fig.update_traces(
        marker_color=color,
        opacity=opacity,
        texttemplate="%{text:.3f}",
        textposition="outside",
        textfont_size=value_label_size,
    )

    paper_bgcolor = "rgba(0,0,0,0)" if transparent_bg else "white"
    plot_bgcolor = "rgba(0,0,0,0)" if transparent_bg else "white"

    fig.update_layout(
        template="simple_white",
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        title=dict(text=metric, x=0.5, font=dict(size=title_size)),
        showlegend=False,
        height=420,
        font=dict(size=axis_label_size),
        margin=dict(l=10, r=10, t=60, b=100),
        xaxis=dict(
            title="Model",
            tickangle=tick_angle,
            tickfont=dict(size=axis_label_size),
        ),
        yaxis=dict(
            title=metric,
            tickfont=dict(size=axis_label_size),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"plot_{metric}")


def plot_selected_metrics(
    results,
    selected_metrics,
    color="#4C78A8",
    opacity=0.8,
    value_label_size=12,
    axis_label_size=12,
    title_size=16,
    tick_angle=-45,
    transparent_bg=True,
):
    if results.empty:
        st.warning("No results to plot.")
        return

    if not selected_metrics:
        st.warning("Please select at least one metric to plot.")
        return

    for i in range(0, len(selected_metrics), 3):
        cols = st.columns(3)

        for j, metric in enumerate(selected_metrics[i:i + 3]):
            with cols[j]:
                plot_one_metric(
                    results=results,
                    metric=metric,
                    color=color,
                    opacity=opacity,
                    value_label_size=value_label_size,
                    axis_label_size=axis_label_size,
                    title_size=title_size,
                    tick_angle=tick_angle,
                    transparent_bg=transparent_bg,
                )


def plot_radar_chart(results):
    st.markdown("#### Radar chart")

    radar_metrics = [
        c for c in [
            "test_f1_macro",
            "test_roc_auc",
            "test_accuracy",
            "best_cv_f1",
            "best_cv_roc_auc",
            "best_cv_balanced_accuracy",
        ]
        if c in results.columns
    ]

    if len(radar_metrics) < 3:
        st.info("Need at least 3 metrics for radar chart.")
        return

    selected_models = st.multiselect(
        "Select models for radar chart",
        results["model_name"].tolist(),
        default=results["model_name"].tolist()[:3],
        key="radar_models",
    )

    if not selected_models:
        return

    fill_alpha = st.slider(
        "Radar fill opacity",
        0.05,
        0.8,
        0.25,
        0.05,
        key="radar_fill_alpha",
    )

    fig = go.Figure()

    for _, row in results[results["model_name"].isin(selected_models)].iterrows():
        values = [row[m] for m in radar_metrics]
        values.append(values[0])

        theta = radar_metrics + [radar_metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                fill="toself",
                opacity=fill_alpha,
                name=row["model_name"],
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        showlegend=True,
        height=550,
        title="Model performance radar chart",
    )

    st.plotly_chart(fig, use_container_width=True)


def get_confusion_matrix_from_row(row):
    if "confusion_matrix" not in row.index:
        return None, None

    cm = row["confusion_matrix"]

    if cm is None or (isinstance(cm, float) and pd.isna(cm)):
        return None, None

    if isinstance(cm, str):
        try:
            cm = json.loads(cm)
        except Exception:
            return None, None

    labels = None

    if "confusion_matrix_labels" in row.index:
        labels = row["confusion_matrix_labels"]

        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except Exception:
                labels = None

    if labels is None:
        labels = [str(i) for i in range(len(cm))]

    labels = [str(x) for x in labels]

    return cm, labels


def plot_confusion_matrix(results):
    st.markdown("#### Confusion matrix")

    selected_model = st.selectbox(
        "Select model for confusion matrix",
        results["model_name"].tolist(),
        key="confusion_matrix_model",
    )

    row = results[results["model_name"] == selected_model].iloc[0]
    cm, labels = get_confusion_matrix_from_row(row)

    if cm is None:
        st.warning(
            "No real confusion_matrix found in this summary JSON. "
            "Save confusion_matrix and confusion_matrix_labels during training."
        )
        plot_recall_fallback(row, selected_model)
        return

    normalize = st.checkbox(
        "Normalize by true class",
        value=False,
        key="normalize_confusion_matrix",
    )

    color_scale = st.selectbox(
        "Confusion matrix color scale",
        [
            "Blues",
            "Viridis",
            "Plasma",
            "Inferno",
            "Magma",
            "Cividis",
            "Greens",
            "Reds",
            "Purples",
            "Turbo",
        ],
        index=0,
        key="cm_color_scale",
    )

    reverse_scale = st.checkbox(
        "Reverse color scale",
        value=False,
        key="cm_reverse_color",
    )

    text_size = st.slider(
        "Confusion matrix text size",
        8,
        28,
        14,
        key="cm_text_size",
    )

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    if normalize:
        cm_plot = cm_df.div(cm_df.sum(axis=1).replace(0, pd.NA), axis=0)
        text_template = ".2f"
        color_title = "Fraction"
    else:
        cm_plot = cm_df
        text_template = "d"
        color_title = "Count"

    fig = px.imshow(
        cm_plot,
        x=labels,
        y=labels,
        text_auto=text_template,
        color_continuous_scale=color_scale,
        aspect="auto",
        labels=dict(
            x="Predicted class",
            y="True class",
            color=color_title,
        ),
        title=f"Confusion matrix: {selected_model}",
    )

    if reverse_scale:
        fig.update_layout(coloraxis_reversescale=True)

    fig.update_traces(
        textfont=dict(size=text_size),
    )

    fig.update_layout(
        template="simple_white",
        title_x=0.5,
        height=550,
        xaxis=dict(side="bottom"),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show confusion matrix table"):
        st.dataframe(cm_df, use_container_width=True)


def plot_recall_fallback(row, selected_model):
    rows = []

    for col in row.index:
        if col.startswith("classification_report.") and col.endswith(".recall"):
            parts = col.split(".")
            if len(parts) >= 3:
                class_id = parts[1]
                if class_id not in ["accuracy", "macro avg", "weighted avg"]:
                    rows.append(
                        {
                            "class": class_id,
                            "recall": row[col],
                        }
                    )

    if not rows:
        st.info("No classification report recall found.")
        return

    df = pd.DataFrame(rows)

    fig = px.bar(
        df,
        x="class",
        y="recall",
        text="recall",
        title=f"Per-class recall fallback: {selected_model}",
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig.update_layout(
        template="simple_white",
        height=400,
        title_x=0.5,
        yaxis=dict(range=[0, 1]),
    )

    st.plotly_chart(fig, use_container_width=True)


def get_best_model(results, metric):
    if metric not in results.columns:
        return None

    ranked = results.dropna(subset=[metric]).sort_values(
        metric,
        ascending=is_lower_better(metric),
    )

    if ranked.empty:
        return None

    return ranked.iloc[0]


def show_analysis(results):
    st.markdown("### Analysis")

    candidate_metrics = [
        "test_f1_macro",
        "test_roc_auc",
        "test_accuracy",
        "test_r2",
        "test_rmse",
        "test_mae",
        "best_cv_f1",
        "best_cv_roc_auc",
        "best_cv_r2",
        "runtime_seconds",
    ]

    available = [m for m in candidate_metrics if m in results.columns]

    if not available:
        st.warning("No analysis metrics found.")
        return

    primary_metric = st.selectbox(
        "Primary metric for ranking",
        available,
        index=0,
        key="analysis_primary_metric",
    )

    best = get_best_model(results, primary_metric)

    if best is not None:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Best model", best.get("model_name", "NA"))

        with c2:
            st.metric(primary_metric, f"{best[primary_metric]:.4f}")

        with c3:
            if "runtime_seconds" in best.index and pd.notna(best["runtime_seconds"]):
                st.metric("Runtime seconds", f"{best['runtime_seconds']:.1f}")

    rank_cols = [
        c for c in [
            "model_name",
            "task_type",
            "best_cv_f1",
            "test_f1_macro",
            "best_cv_roc_auc",
            "test_roc_auc",
            "best_cv_balanced_accuracy",
            "test_accuracy",
            "best_cv_r2",
            "test_r2",
            "test_rmse",
            "test_mae",
            "runtime_seconds",
            "refit_metric",
            "model_file",
        ]
        if c in results.columns
    ]

    ranked = results.sort_values(
        primary_metric,
        ascending=is_lower_better(primary_metric),
    )

    st.markdown("#### Ranked model summary")
    st.dataframe(ranked[rank_cols], use_container_width=True, hide_index=True)

    plot_radar_chart(results)
    plot_confusion_matrix(results)


def design(workdir):
    st.subheader("Model Summary Results")

    # before calling directory_picker
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
        st.info("Please select a results directory.")
        return

    outdir = Path(workdir) / "data"
    output_file = outdir / "summary_results.tsv"

    if st.button("Extract summaries", type="primary"):
        results = extract_model_summaries(
            json_dir=result_dir,
            output_tsv=output_file,
        )

        st.session_state["model_summary_results"] = results
        st.session_state["show_model_plot"] = True
        st.session_state["show_model_analysis"] = False

        st.success(f"Saved summary results to: {output_file}")

    if "model_summary_results" not in st.session_state:
        return

    results = st.session_state["model_summary_results"]

    st.dataframe(results, use_container_width=True)

    c_plot, c_analysis = st.columns(2)

    with c_plot:
        show_plot = st.checkbox(
            "Show interactive plots",
            value=st.session_state.get("show_model_plot", True),
            key="show_model_plot_checkbox",
        )

    with c_analysis:
        if st.button("Analyze results", type="primary", use_container_width=True):
            st.session_state["show_model_analysis"] = True

    st.session_state["show_model_plot"] = show_plot

    if st.session_state.get("show_model_analysis", False):
        show_analysis(results)

    if not show_plot:
        return

    st.markdown("### Plot settings")

    numeric_cols = results.select_dtypes(include="number").columns.tolist()
    default_metrics = get_default_metrics(results)

    selected_metrics = st.multiselect(
        "Select columns to plot",
        numeric_cols,
        default=default_metrics,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        color = st.color_picker("Bar color", "#4C78A8")

    with c2:
        opacity = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05)

    with c3:
        value_label_size = st.slider("Value label size", 8, 30, 12)

    with c4:
        axis_label_size = st.slider("Axis label size", 8, 30, 12)

    c5, c6, c7 = st.columns(3)

    with c5:
        title_size = st.slider("Title size", 10, 40, 16)

    with c6:
        tick_angle = st.slider("X label angle", -90, 90, -45)

    with c7:
        transparent_bg = st.checkbox("Transparent background", value=True)

    plot_selected_metrics(
        results=results,
        selected_metrics=selected_metrics,
        color=color,
        opacity=opacity,
        value_label_size=value_label_size,
        axis_label_size=axis_label_size,
        title_size=title_size,
        tick_angle=tick_angle,
        transparent_bg=transparent_bg,
    )