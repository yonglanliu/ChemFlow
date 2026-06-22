import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict
from src.utils.design import temp_error, temp_success, temp_info
import ast
import numpy as np
import math


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_radar_chart(data: Dict, name: str = "Model", key: str = "radar"):

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

        new_names = []
        for metric_name in names:
            new_label = st.text_input(
                f"Label for {metric_name}",
                value=metric_name,
                key=f"{key}_label_{metric_name}",
            )
            new_names.append(new_label)

    r = values + [values[0]]
    theta = new_names + [new_names[0]]

    text_values = [f"{v:.3f}" for v in r]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            mode="lines+markers+text" if show_values else "lines+markers",
            text=text_values if show_values else None,
            textposition="top center",
            fill="toself",
            fillcolor=hex_to_rgba(fill_color, fill_alpha),
            line=dict(
                color=line_color,
                width=line_width,
            ),
            marker=dict(
                size=marker_size,
                color=line_color,
            ),
            name=trace_name,
        )
    )

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(
                size=title_size,
                color=title_color,
            ),
        ),
        polar=dict(
            angularaxis=dict(
                tickfont=dict(
                    size=label_size,
                    color=label_color,
                )
            ),
            radialaxis=dict(
                visible=True,
                range=[0, axis_max],
                tickfont=dict(
                    size=max(label_size - 2, 8),
                    color=radial_label_color,
                ),
            ),
        ),
        legend=dict(
            font=dict(
                size=legend_size,
            )
        ),
        showlegend=True,
        height=chart_height,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{key}_plot",
    )

def is_empty_value(x):
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() in ["", "None", "nan", "NaN"]:
        return True
    return False


def parse_saved_object(x):
    if is_empty_value(x):
        return None

    if isinstance(x, str):
        try:
            return ast.literal_eval(x.strip())
        except Exception:
            try:
                return [float(v) for v in x.split(",")]
            except Exception:
                return None

    return x


def get_curve_array(x):
    x = parse_saved_object(x)

    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)

    return None


def extract_flattened_roc_curves(results: dict):
    curves = {}

    for col in results.keys():
        col = str(col)

        if not col.startswith("roc_curves_ovr."):
            continue

        if not col.endswith(".fpr"):
            continue

        prefix = col[:-4]
        tpr_col = prefix + ".tpr"

        if tpr_col not in results:
            continue

        fpr = get_curve_array(results[col])
        tpr = get_curve_array(results[tpr_col])

        if fpr is None or tpr is None:
            continue

        if len(fpr) != len(tpr):
            continue

        label = prefix.replace("roc_curves_ovr.", "")

        curves[label] = {
            "fpr": fpr,
            "tpr": tpr,
        }

    return curves


def plot_roc_auc(results: dict, key: str = "roc"):

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

    fig = go.Figure()
    found_curve = False

    auc_value = results.get("test_roc_auc", None)
    mode = "lines+markers" if show_markers else "lines"

    default_colors = {
        "0.0": "#636EFA",
        "1.0": "#EF553B",
        "2.0": "#00CC96",
        "3.0": "#AB63FA",
    }

    # Binary ROC curve
    roc = parse_saved_object(results.get("roc_curve", None))

    if isinstance(roc, dict) and "fpr" in roc and "tpr" in roc:
        fpr = get_curve_array(roc["fpr"])
        tpr = get_curve_array(roc["tpr"])

        if fpr is not None and tpr is not None and len(fpr) == len(tpr):
            curve_color = st.color_picker(
                "ROC Curve Color",
                "#636EFA",
                key=f"{key}_binary_color",
            )

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode=mode,
                    name="ROC curve",
                    line=dict(
                        color=curve_color,
                        width=line_width,
                    ),
                    marker=dict(size=6),
                )
            )

            found_curve = True

    # Nested multiclass ROC curves
    roc_curves = parse_saved_object(results.get("roc_curves_ovr", None))

    if isinstance(roc_curves, dict):
        st.markdown("#### Curve Colors")

        color_cols = st.columns(min(len(roc_curves), 4))

        for i, (label, curve) in enumerate(roc_curves.items()):
            if not isinstance(curve, dict):
                continue

            fpr = get_curve_array(curve.get("fpr"))
            tpr = get_curve_array(curve.get("tpr"))

            if fpr is None or tpr is None or len(fpr) != len(tpr):
                continue

            with color_cols[i % len(color_cols)]:
                curve_color = st.color_picker(
                    f"Class {label}",
                    default_colors.get(str(label), "#636EFA"),
                    key=f"{key}_color_{label}",
                )

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode=mode,
                    name=f"Class {label}",
                    line=dict(
                        color=curve_color,
                        width=line_width,
                    ),
                    marker=dict(size=6),
                )
            )

            found_curve = True

    # Flattened multiclass ROC curves
    flat_curves = extract_flattened_roc_curves(results)

    if flat_curves:
        st.markdown("#### Curve Colors")

        color_cols = st.columns(min(len(flat_curves), 4))

        for i, (label, curve) in enumerate(flat_curves.items()):
            with color_cols[i % len(color_cols)]:
                curve_color = st.color_picker(
                    f"Class {label}",
                    default_colors.get(str(label), "#636EFA"),
                    key=f"{key}_flat_color_{label}",
                )

            fig.add_trace(
                go.Scatter(
                    x=curve["fpr"],
                    y=curve["tpr"],
                    mode=mode,
                    name=f"Class {label}",
                    line=dict(
                        color=curve_color,
                        width=line_width,
                    ),
                    marker=dict(size=6),
                )
            )

            found_curve = True

    if not found_curve:
        st.info(
            "No ROC curve data found. Need roc_curve, roc_curves_ovr, "
            "or flattened columns like roc_curves_ovr.2.0.fpr / .tpr."
        )
        return

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(
                dash="dash",
                color=baseline_color,
                width=line_width,
            ),
        )
    )

    final_title = title_text

    if not is_empty_value(auc_value):
        try:
            final_title += f" | AUC = {float(auc_value):.3f}"
        except Exception:
            pass

    fig.update_layout(
        title=dict(
            text=final_title,
            font=dict(size=font_size + 6),
        ),
        xaxis=dict(
            title=dict(
                text="False Positive Rate",
                font=dict(size=font_size),
            ),
            tickfont=dict(size=max(font_size - 2, 8)),
        ),
        yaxis=dict(
            title=dict(
                text="True Positive Rate",
                font=dict(size=font_size),
            ),
            tickfont=dict(size=max(font_size - 2, 8)),
        ),
        legend=dict(
            font=dict(size=font_size),
        ),
        height=chart_height,
        template="plotly_white",
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{key}_roc_auc",
    )

