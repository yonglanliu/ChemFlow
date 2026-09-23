"""Publication-ready uncertainty reports shared by property-prediction models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import pandas as pd


# Keep all uncertainty figures readable when they are reduced to manuscript size.
_INDIVIDUAL_FIGSIZE = (8.2, 7.0)
_TICK_SIZE = 15
_LABEL_SIZE = 17
_TITLE_SIZE = 18
_LEGEND_SIZE = 14
_ANNOTATION_SIZE = 14


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return name or "target"


def _save_figure(figure, path: Path) -> list[str]:
    paths = []
    for suffix in (".png", ".pdf"):
        output = path.with_suffix(suffix)
        figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(str(output))
    return paths


def _finite_distribution(
    truth: np.ndarray, prediction: np.ndarray, uncertainty: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    valid = (
        np.isfinite(truth)
        & np.isfinite(prediction)
        & np.isfinite(uncertainty)
        & (uncertainty >= 0.0)
    )
    truth, prediction, uncertainty = (
        values[valid] for values in (truth, prediction, uncertainty)
    )
    scale = float(np.nanstd(truth)) if len(truth) else 0.0
    floor = max(scale * 1e-6, 1e-8)
    floor_count = int(np.count_nonzero(uncertainty < floor))
    return truth, prediction, np.maximum(uncertainty, floor), floor, floor_count


def _subset(*arrays: np.ndarray, maximum: int = 100) -> list[np.ndarray]:
    if not arrays or len(arrays[0]) <= maximum:
        return [np.asarray(values) for values in arrays]
    # Even spacing is deterministic and preserves the full response range after
    # the caller orders the arrays by observed value.
    indices = np.linspace(0, len(arrays[0]) - 1, maximum, dtype=int)
    return [np.asarray(values)[indices] for values in arrays]


def _style_axis(axis) -> None:
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=_TICK_SIZE,
        width=1.5,
        length=6,
    )
    axis.tick_params(axis="both", which="minor", width=1.2, length=3.5)
    for spine in axis.spines.values():
        spine.set_linewidth(1.5)
    axis.xaxis.label.set_size(_LABEL_SIZE)
    axis.yaxis.label.set_size(_LABEL_SIZE)
    axis.xaxis.labelpad = 8
    axis.yaxis.labelpad = 8
    axis.set_title(
        axis.get_title(),
        fontsize=_TITLE_SIZE,
        fontweight="bold",
        pad=12,
    )
    for annotation in axis.texts:
        annotation.set_fontsize(_ANNOTATION_SIZE)
    legend = axis.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_fontsize(_LEGEND_SIZE)


def _label_interval_coverage(axis, confidence: float) -> None:
    legend = axis.get_legend()
    if legend is None:
        return
    for text in legend.get_texts():
        if "Interval" in text.get_text():
            text.set_text(f"{confidence:.0%} interval")


def _calibration_metrics(uct_metrics, mean, std, truth, bins: int) -> dict[str, float]:
    values = uct_metrics.get_all_average_calibration(
        mean, std, truth, bins, verbose=False
    )
    return {key: float(value) for key, value in values.items()}


def generate_uncertainty_report(
    *,
    workdir: str | Path,
    predictions: pd.DataFrame,
    targets: Sequence[str],
    task_types: Sequence[str],
    confidence: float,
) -> dict[str, Any]:
    """Save UQ Toolbox plots and their source tables below ``uncertainty``."""
    import matplotlib.pyplot as plt
    from uncertainty_toolbox import metrics as uct_metrics
    from uncertainty_toolbox import metrics_calibration, viz

    root = Path(workdir) / "uncertainty"
    root.mkdir(parents=True, exist_ok=True)
    artifact_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    z_value = float(NormalDist().inv_cdf(0.5 + float(confidence) / 2.0))

    style = {
        "font.size": 15,
        "axes.labelsize": _LABEL_SIZE,
        "axes.titlesize": _TITLE_SIZE,
        "legend.fontsize": _LEGEND_SIZE,
        "xtick.labelsize": _TICK_SIZE,
        "ytick.labelsize": _TICK_SIZE,
        "axes.linewidth": 1.5,
    }
    with plt.rc_context(style):
        for target, task_type in zip(targets, task_types):
            if task_type != "regression":
                skipped[target] = (
                    "Uncertainty Toolbox 0.1.1 visualizations are regression-only."
                )
                continue
            target_dir = root / _safe_name(target)
            target_dir.mkdir(parents=True, exist_ok=True)
            required = [
                f"{target}_true",
                f"{target}_mc_prediction",
                f"{target}_mc_sd",
                f"{target}_calibrated_prediction",
                f"{target}_calibrated_uncertainty",
            ]
            missing = [column for column in required if column not in predictions]
            if missing:
                skipped[target] = f"Missing report columns: {missing}"
                continue

            truth, mc_mean, mc_sd, mc_floor, mc_floor_count = _finite_distribution(
                predictions[f"{target}_true"].to_numpy(float),
                predictions[f"{target}_mc_prediction"].to_numpy(float),
                predictions[f"{target}_mc_sd"].to_numpy(float),
            )
            if len(truth) < 5:
                skipped[target] = "Fewer than five finite test predictions."
                continue
            calibrated_truth, calibrated_mean, calibrated_radius = (
                predictions[column].to_numpy(float)
                for column in (
                    f"{target}_true",
                    f"{target}_calibrated_prediction",
                    f"{target}_calibrated_uncertainty",
                )
            )
            calibrated_std_raw = calibrated_radius / z_value
            (
                calibrated_truth,
                calibrated_mean,
                calibrated_std,
                calibrated_floor,
                calibrated_floor_count,
            ) = _finite_distribution(
                calibrated_truth, calibrated_mean, calibrated_std_raw
            )
            if len(calibrated_truth) < 5:
                skipped[target] = "Fewer than five locally calibrated predictions."
                continue

            bins = min(100, max(10, int(np.sqrt(len(truth)))))
            before = _calibration_metrics(
                uct_metrics, mc_mean, mc_sd, truth, bins
            )
            after = _calibration_metrics(
                uct_metrics,
                calibrated_mean,
                calibrated_std,
                calibrated_truth,
                bins,
            )
            for stage, values, floor, floor_count in (
                ("MC dropout", before, mc_floor, mc_floor_count),
                (
                    "Locally calibrated",
                    after,
                    calibrated_floor,
                    calibrated_floor_count,
                ),
            ):
                comparison_rows.append(
                    {
                        "Target": target,
                        "Stage": stage,
                        "N": len(truth) if stage == "MC dropout" else len(calibrated_truth),
                        "MACE": values["ma_cal"],
                        "RMSCE": values["rms_cal"],
                        "Miscalibration_area": values["miscal_area"],
                        "SD_floor": floor,
                        "SD_floor_count": floor_count,
                    }
                )

            expected_before, observed_before = (
                metrics_calibration.get_proportion_lists_vectorized(
                    mc_mean, mc_sd, truth, num_bins=bins
                )
            )
            expected_after, observed_after = (
                metrics_calibration.get_proportion_lists_vectorized(
                    calibrated_mean,
                    calibrated_std,
                    calibrated_truth,
                    num_bins=bins,
                )
            )
            pd.DataFrame(
                {
                    "expected_proportion": expected_before,
                    "observed_proportion": observed_before,
                }
            ).to_csv(target_dir / "calibration_curve_mc_dropout.csv", index=False)
            pd.DataFrame(
                {
                    "expected_proportion": expected_after,
                    "observed_proportion": observed_after,
                }
            ).to_csv(
                target_dir / "calibration_curve_locally_calibrated.csv", index=False
            )
            pd.DataFrame(
                [row for row in comparison_rows if row["Target"] == target]
            ).to_csv(target_dir / "calibration_metrics_comparison.csv", index=False)

            report_columns = [
                "SMILES",
                f"{target}_true",
                f"{target}_direct_prediction",
                f"{target}_mc_prediction",
                f"{target}_mc_sd",
                f"{target}_calibrated_prediction",
                f"{target}_calibrated_uncertainty",
                f"{target}_ood_score",
                f"{target}_ood_flag",
                f"{target}_uncertainty_flag",
                f"{target}_interval_covered",
            ]
            predictions[
                [column for column in report_columns if column in predictions]
            ].to_csv(target_dir / "prediction_uncertainty_data.csv", index=False)

            prediction_order = np.argsort(mc_mean, kind="stable")
            band_truth, band_mean, band_sd = _subset(
                truth[prediction_order],
                mc_mean[prediction_order],
                mc_sd[prediction_order],
                maximum=100,
            )
            band_index = np.arange(len(band_truth))
            pd.DataFrame(
                {
                    "plot_index": band_index,
                    "observed_value": band_truth,
                    "mc_prediction": band_mean,
                    "mc_sd": band_sd,
                    "interval_lower": band_mean - z_value * band_sd,
                    "interval_upper": band_mean + z_value * band_sd,
                }
            ).to_csv(
                target_dir / "confidence_band_sorted_by_prediction.csv",
                index=False,
            )

            observed_order = np.argsort(truth, kind="stable")
            ordered_truth, ordered_mean, ordered_sd = _subset(
                truth[observed_order],
                mc_mean[observed_order],
                mc_sd[observed_order],
                maximum=100,
            )
            pd.DataFrame(
                {
                    "plot_index": np.arange(len(ordered_truth)),
                    "observed_value": ordered_truth,
                    "mc_prediction": ordered_mean,
                    "mc_sd": ordered_sd,
                    "interval_lower": ordered_mean - z_value * ordered_sd,
                    "interval_upper": ordered_mean + z_value * ordered_sd,
                }
            ).to_csv(
                target_dir / "ordered_intervals_sorted_by_observed.csv",
                index=False,
            )

            figures: list[tuple[str, Any]] = []
            figure, axis = plt.subplots(figsize=_INDIVIDUAL_FIGSIZE)
            viz.plot_xy(
                band_mean,
                band_sd,
                band_truth,
                band_index,
                num_stds_confidence_bound=z_value,
                ax=axis,
            )
            axis.set_xlabel("Test compounds ordered by predicted value")
            axis.set_ylabel(target)
            axis.set_title("MC-dropout confidence band")
            _label_interval_coverage(axis, confidence)
            _style_axis(axis)
            figures.append(("01_confidence_band", figure))

            figure, axis = plt.subplots(figsize=_INDIVIDUAL_FIGSIZE)
            viz.plot_intervals(
                ordered_mean,
                ordered_sd,
                ordered_truth,
                num_stds_confidence_bound=z_value,
                ax=axis,
            )
            axis.set_title("Observed versus predicted intervals")
            _style_axis(axis)
            figures.append(("02_prediction_intervals", figure))

            figure, axis = plt.subplots(figsize=_INDIVIDUAL_FIGSIZE)
            viz.plot_intervals_ordered(
                ordered_mean,
                ordered_sd,
                ordered_truth,
                num_stds_confidence_bound=z_value,
                ax=axis,
            )
            _style_axis(axis)
            figures.append(("03_ordered_prediction_intervals", figure))

            figure, axis = plt.subplots(figsize=_INDIVIDUAL_FIGSIZE)
            viz.plot_calibration(
                mc_mean,
                mc_sd,
                truth,
                exp_props=expected_before,
                obs_props=observed_before,
                curve_label="MC dropout",
                ax=axis,
            )
            _style_axis(axis)
            figures.append(("04_average_calibration_mc_dropout", figure))

            figure, axis = plt.subplots(figsize=_INDIVIDUAL_FIGSIZE)
            viz.plot_calibration(
                calibrated_mean,
                calibrated_std,
                calibrated_truth,
                exp_props=expected_after,
                obs_props=observed_after,
                curve_label="Locally calibrated",
                ax=axis,
            )
            axis.set_title("Average calibration after local calibration")
            _style_axis(axis)
            figures.append(("05_average_calibration_locally_calibrated", figure))

            figure, axis = plt.subplots(figsize=_INDIVIDUAL_FIGSIZE)
            axis.plot([0, 1], [0, 1], "--", color="#4D4D4D", label="Ideal")
            axis.plot(
                expected_before, observed_before, color="#D55E00", linewidth=2,
                label=f"MC dropout (MA={before['miscal_area']:.3f})",
            )
            axis.plot(
                expected_after, observed_after, color="#0072B2", linewidth=2,
                label=f"Locally calibrated (MA={after['miscal_area']:.3f})",
            )
            axis.set(xlim=(0, 1), ylim=(0, 1),
                     xlabel="Predicted proportion in interval",
                     ylabel="Observed proportion in interval",
                     title="Calibration before and after local calibration")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            axis.legend()
            _style_axis(axis)
            figures.append(("06_calibration_before_after", figure))

            figure, axis = plt.subplots(figsize=_INDIVIDUAL_FIGSIZE)
            viz.plot_residuals_vs_stds(mc_mean, mc_sd, truth, ax=axis)
            _style_axis(axis)
            figures.append(("07_residuals_vs_mc_sd", figure))

            figure, axis = plt.subplots(figsize=_INDIVIDUAL_FIGSIZE)
            viz.plot_sharpness(mc_sd, ax=axis)
            _style_axis(axis)
            figures.append(("08_sharpness", figure))

            state = np.random.get_state()
            np.random.seed(42)
            try:
                adversarial = metrics_calibration.adversarial_group_calibration(
                    mc_mean,
                    mc_sd,
                    truth,
                    cali_type="mean_abs",
                    num_bins=bins,
                    num_group_bins=10,
                    num_trials=10,
                    num_group_draws=10,
                )
            finally:
                np.random.set_state(state)
            pd.DataFrame(
                {
                    "group_size": adversarial.group_size,
                    "score_mean": adversarial.score_mean,
                    "score_stderr": adversarial.score_stderr,
                }
            ).to_csv(target_dir / "adversarial_group_calibration.csv", index=False)
            figure, axis = plt.subplots(figsize=_INDIVIDUAL_FIGSIZE)
            viz.plot_adversarial_group_calibration(
                mc_mean,
                mc_sd,
                truth,
                group_size=adversarial.group_size,
                score_mean=adversarial.score_mean,
                score_stderr=adversarial.score_stderr,
                curve_label="MC dropout",
                ax=axis,
            )
            _style_axis(axis)
            figures.append(("09_adversarial_group_calibration", figure))

            overview, axes = plt.subplots(
                1,
                3,
                figsize=(18.0, 6.6),
                gridspec_kw={"wspace": 0.12},
            )
            viz.plot_xy(
                band_mean,
                band_sd,
                band_truth,
                band_index,
                num_stds_confidence_bound=z_value,
                ax=axes[0],
            )
            axes[0].set_xlabel("Test compounds ordered by predicted value")
            axes[0].set_ylabel(target)
            _label_interval_coverage(axes[0], confidence)
            viz.plot_intervals_ordered(
                ordered_mean,
                ordered_sd,
                ordered_truth,
                num_stds_confidence_bound=z_value,
                ax=axes[1],
            )
            viz.plot_calibration(
                mc_mean,
                mc_sd,
                truth,
                exp_props=expected_before,
                obs_props=observed_before,
                curve_label="MC dropout",
                ax=axes[2],
            )
            for axis in axes:
                _style_axis(axis)
            overview.suptitle(
                f"{target}: MC-dropout uncertainty",
                fontsize=20,
                fontweight="bold",
                y=0.975,
            )
            # Explicit margins avoid the very large inter-panel whitespace that
            # tight_layout adds around Uncertainty Toolbox's square axes.
            overview.subplots_adjust(
                left=0.055,
                right=0.985,
                bottom=0.16,
                top=0.86,
                wspace=0.12,
            )
            figures.append(("10_uncertainty_overview", overview))

            for stem, figure in figures:
                paths = _save_figure(figure, target_dir / stem)
                plt.close(figure)
                for path in paths:
                    artifact_rows.append(
                        {"Target": target, "Artifact": stem, "Path": path}
                    )

    comparison_path = root / "calibration_metrics_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    artifact_path = root / "uncertainty_artifacts.csv"
    pd.DataFrame(artifact_rows).to_csv(artifact_path, index=False)
    manifest = {
        "toolbox": "uncertainty-toolbox",
        "output_directory": str(root),
        "confidence": float(confidence),
        "calibrated_sd_note": (
            "For visualization only, local interval half-width was divided by the "
            "matching Gaussian z value. The saved local interval remains the "
            "authoritative uncertainty estimate."
        ),
        "targets_plotted": sorted({row["Target"] for row in artifact_rows}),
        "targets_skipped": skipped,
        "calibration_table": str(comparison_path),
        "artifact_index": str(artifact_path),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
