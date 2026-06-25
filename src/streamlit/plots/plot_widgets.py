# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import streamlit as st

from src.chemflow.plots.model_plots import (
    make_radar_chart, 
    make_roc_auc_plot, 
    make_distribution_with_boxplot,
    make_metric_bar_plot
    )


def st_radar_chart(
    data: dict,
    name: str = "Model",
    key: str = "radar",
) -> None:
    values = list(data.values())
    names = list(data.keys())

    if len(names) < 3:
        st.info("Need at least 3 metrics for radar chart.")
        return

    with st.expander("Radar Chart Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            title_text = st.text_input(
                "Chart Title",
                "Model Performance Radar Chart",
                key=f"{key}_title_text",
            )

            trace_name = st.text_input(
                "Legend Name",
                value=name,
                key=f"{key}_trace_name",
            )

            line_color = st.color_picker(
                "Line Color",
                "#4C78A8",
                key=f"{key}_line_color",
            )

        with c2:
            fill_color = st.color_picker(
                "Fill Color",
                "#4C78A8",
                key=f"{key}_fill_color",
            )

            fill_alpha = st.slider(
                "Fill Opacity",
                0.0,
                1.0,
                0.25,
                0.05,
                key=f"{key}_fill_alpha",
            )

            line_width = st.slider(
                "Line Width",
                1,
                10,
                3,
                key=f"{key}_line_width",
            )

        with c3:
            marker_size = st.slider(
                "Marker Size",
                0,
                20,
                8,
                key=f"{key}_marker_size",
            )

            axis_max = st.slider(
                "Axis Max",
                0.5,
                2.0,
                1.0,
                0.1,
                key=f"{key}_axis_max",
            )

            chart_height = st.slider(
                "Chart Height",
                300,
                900,
                550,
                50,
                key=f"{key}_chart_height",
            )

        with c4:
            title_size = st.slider(
                "Title Size",
                10,
                40,
                22,
                key=f"{key}_title_size",
            )

            label_size = st.slider(
                "Axis Label Size",
                8,
                30,
                14,
                key=f"{key}_label_size",
            )

            legend_size = st.slider(
                "Legend Size",
                8,
                30,
                12,
                key=f"{key}_legend_size",
            )

        c5, c6, c7, c8 = st.columns(4)

        with c5:
            title_color = st.color_picker(
                "Title Color",
                "#000000",
                key=f"{key}_title_color",
            )

        with c6:
            label_color = st.color_picker(
                "Axis Label Color",
                "#000000",
                key=f"{key}_label_color",
            )

        with c7:
            radial_label_color = st.color_picker(
                "Radial Label Color",
                "#666666",
                key=f"{key}_radial_label_color",
            )

        with c8:
            show_values = st.checkbox(
                "Show Values",
                value=True,
                key=f"{key}_show_values",
            )

        st.markdown("#### Rename Metric Labels")

        label_map = {}
        for metric_name in names:
            new_label = st.text_input(
                f"Label for {metric_name}",
                value=metric_name,
                key=f"{key}_label_{metric_name}",
            )
            label_map[metric_name] = new_label

    fig = make_radar_chart(
        data=data,
        trace_name=trace_name,
        title_text=title_text,
        line_color=line_color,
        fill_color=fill_color,
        fill_alpha=fill_alpha,
        line_width=line_width,
        marker_size=marker_size,
        axis_max=axis_max,
        chart_height=chart_height,
        title_size=title_size,
        label_size=label_size,
        legend_size=legend_size,
        title_color=title_color,
        label_color=label_color,
        radial_label_color=radial_label_color,
        show_values=show_values,
        label_map=label_map,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key=f"{key}_plot",
    )


def st_roc_auc_plot(
    results: dict,
    key: str = "roc",
) -> None:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        line_width = st.slider(
            "Line Width",
            1,
            10,
            3,
            key=f"{key}_line_width",
        )

    with c2:
        font_size = st.slider(
            "Text Size",
            8,
            30,
            16,
            key=f"{key}_font_size",
        )

    with c3:
        chart_height = st.slider(
            "Height",
            300,
            1000,
            550,
            50,
            key=f"{key}_height",
        )

    with c4:
        show_markers = st.checkbox(
            "Show Markers",
            value=False,
            key=f"{key}_show_markers",
        )

    c5, c6 = st.columns(2)

    with c5:
        title_text = st.text_input(
            "Title",
            "ROC-AUC Curve",
            key=f"{key}_title",
        )

    with c6:
        baseline_color = st.color_picker(
            "Baseline Color",
            "#FF8C42",
            key=f"{key}_baseline_color",
        )

    binary_color = st.color_picker(
        "ROC Curve Color",
        "#636EFA",
        key=f"{key}_binary_color",
    )

    class_colors = {}

    for label in ["0.0", "1.0", "2.0", "3.0"]:
        class_colors[label] = st.color_picker(
            f"Class {label}",
            value={
                "0.0": "#636EFA",
                "1.0": "#EF553B",
                "2.0": "#00CC96",
                "3.0": "#AB63FA",
            }[label],
            key=f"{key}_color_{label}",
        )

    try:
        fig = make_roc_auc_plot(
            results=results,
            title_text=title_text,
            line_width=line_width,
            font_size=font_size,
            chart_height=chart_height,
            show_markers=show_markers,
            baseline_color=baseline_color,
            binary_color=binary_color,
            class_colors=class_colors,
        )

    except ValueError as e:
        st.info(str(e))
        return

    st.plotly_chart(
        fig,
        width="stretch",
        key=f"{key}_roc_auc",
    )



def st_distribution_with_boxplot(
    df,
    x_col: str,
    key: str = "distribution_boxplot",
):
    with st.expander("Distribution Plot Settings", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            title = st.text_input(
                "Title",
                value=f"{x_col} Distribution",
                key=f"{key}_title",
            )

            nbins = st.slider(
                "Number of bins",
                min_value=10,
                max_value=100,
                value=40,
                step=5,
                key=f"{key}_nbins",
            )

        with c2:
            x_label = st.text_input(
                "X-axis label",
                value=x_col,
                key=f"{key}_x_label",
            )

            height = st.slider(
                "Height",
                min_value=300,
                max_value=900,
                value=500,
                step=50,
                key=f"{key}_height",
            )

        with c3:
            color = st.color_picker(
                "Bar color",
                value="#1f77b4",
                key=f"{key}_color",
            )

            template = st.selectbox(
                "Theme",
                options=["plotly_white", "plotly_dark", "simple_white"],
                index=1,
                key=f"{key}_template",
            )

    try:
        fig = make_distribution_with_boxplot(
            df=df,
            x_col=x_col,
            title=title,
            x_label=x_label,
            nbins=nbins,
            color=color,
            template=template,
            height=height,
        )

    except ValueError as e:
        st.info(str(e))
        return

    st.plotly_chart(
        fig,
        width="stretch",
        key=f"{key}_plot",
    )



def st_metric_bar_plot(
    metric_dict: dict,
    key: str = "bar",
):
    c1, c2, c3 = st.columns(3)

    with c1:
        title = st.text_input(
            "Title",
            "Model Metrics",
            key=f"{key}_title",
        )

    with c2:
        height = st.slider(
            "Height",
            300,
            900,
            500,
            key=f"{key}_height",
        )

    with c3:
        color = st.color_picker(
            "Bar Color",
            "#4C78A8",
            key=f"{key}_color",
        )

    fig = make_metric_bar_plot(
        metric_dict=metric_dict,
        title=title,
        height=height,
        color=color,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key=f"{key}_plot",
    )