# src/plots/model_plots.py

from __future__ import annotations

import ast
import math
from typing import Dict, Any

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def is_empty_value(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() in ["", "None", "nan", "NaN"]:
        return True
    return False


def parse_saved_object(x: Any) -> Any:
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


def get_curve_array(x: Any) -> list | None:
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


def extract_flattened_roc_curves(results: dict) -> dict:
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


def make_radar_chart(
    data: Dict[str, float],
    trace_name: str = "Model",
    title_text: str = "Model Performance Radar Chart",
    line_color: str = "#4C78A8",
    fill_color: str = "#4C78A8",
    fill_alpha: float = 0.25,
    line_width: int = 3,
    marker_size: int = 8,
    axis_max: float = 1.0,
    chart_height: int = 550,
    title_size: int = 22,
    label_size: int = 14,
    legend_size: int = 12,
    title_color: str = "#000000",
    label_color: str = "#000000",
    radial_label_color: str = "#666666",
    show_values: bool = True,
    label_map: dict[str, str] | None = None,
) -> go.Figure:
    values = list(data.values())
    names = list(data.keys())

    if len(names) < 3:
        raise ValueError("Need at least 3 metrics for radar chart.")

    label_map = label_map or {}
    display_names = [label_map.get(name, name) for name in names]

    r = values + [values[0]]
    theta = display_names + [display_names[0]]
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
            line=dict(color=line_color, width=line_width),
            marker=dict(size=marker_size, color=line_color),
            name=trace_name,
        )
    )

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=title_size, color=title_color),
        ),
        polar=dict(
            angularaxis=dict(
                tickfont=dict(size=label_size, color=label_color),
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
        legend=dict(font=dict(size=legend_size)),
        showlegend=True,
        height=chart_height,
    )

    return fig


def make_roc_auc_plot(
    results: dict,
    title_text: str = "ROC-AUC Curve",
    line_width: int = 3,
    font_size: int = 16,
    chart_height: int = 550,
    show_markers: bool = False,
    baseline_color: str = "#FF8C42",
    binary_color: str = "#636EFA",
    class_colors: dict[str, str] | None = None,
) -> go.Figure:
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

    class_colors = class_colors or {}

    roc = parse_saved_object(results.get("roc_curve", None))

    if isinstance(roc, dict) and "fpr" in roc and "tpr" in roc:
        fpr = get_curve_array(roc["fpr"])
        tpr = get_curve_array(roc["tpr"])

        if fpr is not None and tpr is not None and len(fpr) == len(tpr):
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode=mode,
                    name="ROC curve",
                    line=dict(color=binary_color, width=line_width),
                    marker=dict(size=6),
                )
            )
            found_curve = True

    roc_curves = parse_saved_object(results.get("roc_curves_ovr", None))

    if isinstance(roc_curves, dict):
        for label, curve in roc_curves.items():
            if not isinstance(curve, dict):
                continue

            fpr = get_curve_array(curve.get("fpr"))
            tpr = get_curve_array(curve.get("tpr"))

            if fpr is None or tpr is None or len(fpr) != len(tpr):
                continue

            label = str(label)
            curve_color = class_colors.get(
                label,
                default_colors.get(label, "#636EFA"),
            )

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode=mode,
                    name=f"Class {label}",
                    line=dict(color=curve_color, width=line_width),
                    marker=dict(size=6),
                )
            )
            found_curve = True

    flat_curves = extract_flattened_roc_curves(results)

    for label, curve in flat_curves.items():
        label = str(label)
        curve_color = class_colors.get(
            label,
            default_colors.get(label, "#636EFA"),
        )

        fig.add_trace(
            go.Scatter(
                x=curve["fpr"],
                y=curve["tpr"],
                mode=mode,
                name=f"Class {label}",
                line=dict(color=curve_color, width=line_width),
                marker=dict(size=6),
            )
        )
        found_curve = True

    if not found_curve:
        raise ValueError(
            "No ROC curve data found. Need roc_curve, roc_curves_ovr, "
            "or flattened roc_curves_ovr.*.fpr / .tpr columns."
        )

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
            title=dict(text="False Positive Rate", font=dict(size=font_size)),
            tickfont=dict(size=max(font_size - 2, 8)),
        ),
        yaxis=dict(
            title=dict(text="True Positive Rate", font=dict(size=font_size)),
            tickfont=dict(size=max(font_size - 2, 8)),
        ),
        legend=dict(font=dict(size=font_size)),
        height=chart_height,
        template="plotly_white",
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    return fig

def make_distribution_with_boxplot(
    df: pd.DataFrame,
    x_col: str,
    title: str = "Distribution",
    x_label: str | None = None,
    y_label: str = "Frequency",
    nbins: int = 40,
    color: str | None = None,
    marginal: str = "box",
    template: str = "plotly_dark",
    height: int = 500,
) -> go.Figure:
    """
    Create a histogram with a marginal box plot.

    This is a pure Plotly function.
    It does not depend on Streamlit, so it can also be reused in PyQt6.
    """

    if x_col not in df.columns:
        raise ValueError(f"Column not found: {x_col}")

    plot_df = df[[x_col]].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col])

    if plot_df.empty:
        raise ValueError(f"No valid numeric values found in column: {x_col}")

    fig = px.histogram(
        plot_df,
        x=x_col,
        nbins=nbins,
        marginal=marginal,
        color_discrete_sequence=[color] if color else None,
        template=template,
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=x_label or x_col,
        yaxis_title=y_label,
        height=height,
        bargap=0.02,
    )

    fig.update_traces(
        marker_line_width=0,
        selector=dict(type="histogram"),
    )

    return fig

def make_metric_bar_plot(
    metric_dict: dict[str, float],
    title: str = "Model Metrics",
    x_label: str = "Metric",
    y_label: str = "Value",
    height: int = 500,
    color: str = "#4C78A8",
) -> go.Figure:

    df = pd.DataFrame(
        {
            "Metric": list(metric_dict.keys()),
            "Value": list(metric_dict.values()),
        }
    )

    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        text="Value",
    )

    fig.update_traces(
        marker_color=color,
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        template="plotly_white",
    )

    return fig