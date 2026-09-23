"""Consistent post-training test artifacts for predictive deep-learning models."""

from __future__ import annotations

import json
import importlib.metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from statistics import NormalDist
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    log_loss,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _finite_quantile(values: np.ndarray, confidence: float) -> float:
    values = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if not len(values):
        return float("nan")
    rank = int(np.ceil((len(values) + 1) * float(confidence))) - 1
    return float(values[min(max(rank, 0), len(values) - 1)])


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def fingerprint_local_predictions(
    *,
    train_smiles: Sequence[str],
    validation_smiles: Sequence[str],
    validation_truths: np.ndarray,
    validation_predictions: np.ndarray,
    test_smiles: Sequence[str],
    direct_predictions: np.ndarray,
    mc_predictions: np.ndarray,
    mc_standard_deviations: np.ndarray,
    targets: Sequence[str],
    confidence: float = 0.90,
    min_similarity: float = 0.30,
    max_neighbors: int = 100,
) -> dict[str, np.ndarray]:
    """Create validation-only local calibration and fingerprint AD diagnostics."""
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprints(values: Sequence[str]):
        result = []
        for value in values:
            molecule = Chem.MolFromSmiles(str(value))
            if molecule is None:
                raise ValueError(f"Invalid SMILES during post-training evaluation: {value}")
            result.append(generator.GetFingerprint(molecule))
        return result

    train_fps = fingerprints(train_smiles)
    val_fps = fingerprints(validation_smiles)
    test_fps = fingerprints(test_smiles)
    validation_truths = np.asarray(validation_truths, dtype=float)
    validation_predictions = np.asarray(validation_predictions, dtype=float)
    direct_predictions = np.asarray(direct_predictions, dtype=float)
    mc_predictions = np.asarray(mc_predictions, dtype=float)
    mc_standard_deviations = np.asarray(mc_standard_deviations, dtype=float)
    output: dict[str, np.ndarray] = {}

    val_train_similarity = np.asarray([
        max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in val_fps
    ])
    test_train_similarity = np.asarray([
        max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in test_fps
    ])
    val_novelty = 1.0 - val_train_similarity
    test_novelty = 1.0 - test_train_similarity
    ood_scores = np.asarray([
        1.0 - (1.0 + np.count_nonzero(val_novelty >= value))
        / (len(val_novelty) + 1.0)
        for value in test_novelty
    ])
    ood_flags = ood_scores >= 0.95
    z_value = float(NormalDist().inv_cdf(0.5 + float(confidence) / 2.0))
    coverage_label = int(round(float(confidence) * 100))

    for task_index, target in enumerate(targets):
        output_name = f"{target}_prediction"
        output[f"direct_{output_name}"] = direct_predictions[:, task_index]
        output[f"mc_mean_{output_name}"] = mc_predictions[:, task_index]
        output[output_name] = mc_predictions[:, task_index]
        output[f"mc_std_{output_name}"] = mc_standard_deviations[:, task_index]
        calibrated = np.full(len(test_fps), np.nan)
        lower = np.full(len(test_fps), np.nan)
        upper = np.full(len(test_fps), np.nan)
        local_bias = np.full(len(test_fps), np.nan)
        local_radius = np.full(len(test_fps), np.nan)
        local_count = np.zeros(len(test_fps), dtype=int)
        local_weak = np.zeros(len(test_fps), dtype=bool)
        uncertainty_flag = np.zeros(len(test_fps), dtype=bool)
        valid_val = np.isfinite(validation_truths[:, task_index]) & np.isfinite(
            validation_predictions[:, task_index]
        )
        eligible = np.flatnonzero(valid_val)
        for test_index, test_fp in enumerate(test_fps):
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(test_fp, val_fps), dtype=float
            )
            selected = eligible[similarities[eligible] >= float(min_similarity)]
            weak = len(selected) < min(20, len(eligible))
            if weak:
                # Expand only to a small nearest-neighbor neighborhood. This
                # must not become a whole-validation/global residual estimate.
                neighbor_count = min(max_neighbors, 20, len(eligible))
                selected = eligible[
                    np.argsort(similarities[eligible])[::-1][:neighbor_count]
                ]
            elif len(selected) > max_neighbors:
                selected = selected[np.argsort(similarities[selected])[::-1][:max_neighbors]]
            if not len(selected):
                continue
            local_count[test_index] = len(selected)
            local_weak[test_index] = weak
            signed_errors = (
                validation_truths[selected, task_index]
                - validation_predictions[selected, task_index]
            )
            bias = float(np.median(signed_errors))
            center = mc_predictions[test_index, task_index] + bias
            residual_radius = _finite_quantile(
                np.abs(signed_errors - bias), confidence
            )
            mc_radius = z_value * mc_standard_deviations[test_index, task_index]
            radius = max(residual_radius, mc_radius)
            uncertainty_flag[test_index] = mc_radius > residual_radius
            local_bias[test_index] = bias
            local_radius[test_index] = residual_radius
            calibrated[test_index] = center
            lower[test_index] = center - radius
            upper[test_index] = center + radius
        output[f"calibrated_{output_name}"] = calibrated
        output[f"{output_name}_lower_{coverage_label}"] = lower
        output[f"{output_name}_upper_{coverage_label}"] = upper
        output[f"{target}_train_max_tanimoto"] = test_train_similarity
        output[f"{target}_ood_score"] = ood_scores
        output[f"{target}_ood_flag"] = ood_flags
        output[f"{target}_local_calibration_count"] = local_count
        output[f"{target}_local_calibration_used"] = np.isfinite(calibrated)
        output[f"{target}_local_calibration_weak"] = local_weak
        output[f"{target}_local_calibration_bias"] = local_bias
        output[f"{target}_local_uncertainty_radius"] = local_radius
        output[f"{output_name}_uncertainty_flag"] = uncertainty_flag
    return output


def _finite_statistic(function, truth: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(truth) < 2 or np.unique(truth).size < 2 or np.unique(prediction).size < 2:
        return None
    value = float(function(truth, prediction).statistic)
    return value if np.isfinite(value) else None


def uncertainty_toolbox_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    standard_deviation: np.ndarray,
) -> dict[str, Any]:
    """Calculate distributional MC-dropout diagnostics with UQ Toolbox."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    standard_deviation = np.asarray(standard_deviation, dtype=float)
    valid = (
        np.isfinite(truth)
        & np.isfinite(prediction)
        & np.isfinite(standard_deviation)
        & (standard_deviation >= 0.0)
    )
    truth, prediction, standard_deviation = (
        values[valid] for values in (truth, prediction, standard_deviation)
    )
    metadata: dict[str, Any] = {
        "package": "uncertainty-toolbox",
        "version": importlib.metadata.version("uncertainty-toolbox"),
        "distribution_assumption": "Gaussian MC-dropout predictive distribution",
        "n": int(len(truth)),
    }
    if len(truth) < 2:
        return {**metadata, "available": False, "reason": "fewer_than_two_values"}

    # UQ Toolbox requires a positive predictive standard deviation. Preserve
    # zero-spread MC predictions as strongly overconfident by flooring rather
    # than dropping them from the evaluation.
    scale = float(np.nanstd(truth))
    sd_floor = max(scale * 1e-6, 1e-8)
    clipped = standard_deviation < sd_floor
    standard_deviation = np.maximum(standard_deviation, sd_floor)
    num_bins = min(100, max(10, int(np.sqrt(len(truth)))))
    resolution = min(99, max(20, int(np.sqrt(len(truth)) * 2)))
    try:
        from uncertainty_toolbox import metrics as uct_metrics

        calibration = uct_metrics.get_all_average_calibration(
            prediction, standard_deviation, truth, num_bins, verbose=False
        )
        sharpness = uct_metrics.get_all_sharpness_metrics(
            standard_deviation, verbose=False
        )
        scoring = uct_metrics.get_all_scoring_rule_metrics(
            prediction,
            standard_deviation,
            truth,
            resolution,
            True,
            verbose=False,
        )
    except Exception as error:
        return {
            **metadata,
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
        }
    return {
        **metadata,
        "available": True,
        "sd_floor": sd_floor,
        "sd_floor_count": int(clipped.sum()),
        "num_bins": num_bins,
        "resolution": resolution,
        "average_calibration": {
            key: float(value) for key, value in calibration.items()
        },
        "sharpness": {key: float(value) for key, value in sharpness.items()},
        "proper_scoring_rules": {
            key: float(value) for key, value in scoring.items()
        },
    }


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[valid], prediction[valid]
    if not len(truth):
        return {name: None for name in (
            "mse", "rmse", "mae", "median_ae", "r2", "pearson", "spearman", "kendall"
        )} | {"n": 0}
    return {
        "n": int(len(truth)),
        "mse": float(mean_squared_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "median_ae": float(median_absolute_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)) if len(truth) >= 2 else None,
        "pearson": _finite_statistic(pearsonr, truth, prediction),
        "spearman": _finite_statistic(spearmanr, truth, prediction),
        "kendall": _finite_statistic(kendalltau, truth, prediction),
    }


def classification_metrics(
    truth: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=float)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(probability)
    truth = truth[valid].astype(int)
    probability = np.clip(probability[valid], 1e-7, 1.0 - 1e-7)
    if not len(truth):
        return {"n": 0}
    predicted = (probability >= float(threshold)).astype(int)
    both = np.unique(truth).size == 2
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, predicted)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(truth, probability)),
        "roc_auc": float(roc_auc_score(truth, probability)) if both else None,
        "pr_auc": float(average_precision_score(truth, probability)) if both else None,
    }


def save_test_evaluation(
    *,
    workdir: str | Path,
    smiles: Sequence[str],
    truths: np.ndarray,
    targets: Sequence[str],
    task_types: Sequence[str],
    predictions: dict[str, np.ndarray],
    summary: dict[str, Any],
    confidence: float = 0.90,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write standardized direct, MC, local-calibrated, AD, and UQ results."""
    workdir = Path(workdir)
    truths = np.asarray(truths, dtype=float)
    if truths.ndim == 1:
        truths = truths.reshape(-1, 1)
    if truths.shape != (len(smiles), len(targets)):
        raise ValueError("Test truth shape does not match SMILES and target metadata.")
    coverage_label = int(round(float(confidence) * 100))
    frame = pd.DataFrame({"SMILES": list(smiles)})
    variants: dict[str, dict[str, Any]] = {
        "direct_prediction": {},
        "mc_dropout_prediction": {},
        "locally_calibrated_prediction": {},
    }
    diagnostics: dict[str, Any] = {}

    for index, (target, task_type) in enumerate(zip(targets, task_types)):
        output_name = f"{target}_prediction"
        truth = truths[:, index]
        direct = np.asarray(
            predictions.get(f"direct_{output_name}", np.full(len(frame), np.nan)),
            dtype=float,
        )
        mc = np.asarray(
            predictions.get(f"mc_mean_{output_name}", predictions.get(output_name, direct)),
            dtype=float,
        )
        mc_sd = np.asarray(
            predictions.get(f"mc_std_{output_name}", np.full(len(frame), np.nan)),
            dtype=float,
        )
        calibrated = np.asarray(
            predictions.get(f"calibrated_{output_name}", np.full(len(frame), np.nan)),
            dtype=float,
        )
        lower = np.asarray(
            predictions.get(
                f"{output_name}_lower_{coverage_label}", np.full(len(frame), np.nan)
            ),
            dtype=float,
        )
        upper = np.asarray(
            predictions.get(
                f"{output_name}_upper_{coverage_label}", np.full(len(frame), np.nan)
            ),
            dtype=float,
        )
        frame[f"{target}_true"] = truth
        # Keep the historical prediction name for downstream scripts while
        # exposing the explicit direct/MC/calibrated contract.
        frame[f"{target}_prediction"] = direct
        frame[f"{target}_direct_prediction"] = direct
        frame[f"{target}_mc_prediction"] = mc
        frame[f"{target}_mc_sd"] = mc_sd
        frame[f"{target}_calibrated_prediction"] = calibrated
        frame[f"{target}_calibrated_lower_{coverage_label}"] = lower
        frame[f"{target}_calibrated_upper_{coverage_label}"] = upper
        frame[f"{target}_calibrated_uncertainty"] = np.maximum(
            calibrated - lower, upper - calibrated
        )

        metric_function = (
            classification_metrics if task_type == "classification" else regression_metrics
        )
        if task_type == "classification":
            variants["direct_prediction"][target] = metric_function(
                truth, direct, threshold
            )
            variants["mc_dropout_prediction"][target] = metric_function(
                truth, mc, threshold
            )
            variants["locally_calibrated_prediction"][target] = metric_function(
                truth, calibrated, threshold
            )
        else:
            variants["direct_prediction"][target] = metric_function(truth, direct)
            variants["mc_dropout_prediction"][target] = metric_function(truth, mc)
            variants["locally_calibrated_prediction"][target] = metric_function(
                truth, calibrated
            )

        diagnostic_names = (
            "train_max_tanimoto",
            "train_embedding_cosine_distance",
            "ood_score",
            "ood_flag",
            "local_calibration_count",
            "local_calibration_used",
            "local_calibration_weak",
            "local_calibration_bias",
            "local_uncertainty_radius",
            "validation_max_tanimoto",
            "validation_max_embedding_similarity",
        )
        for diagnostic_name in diagnostic_names:
            key = f"{target}_{diagnostic_name}"
            if key in predictions:
                frame[key] = predictions[key]
        uncertainty_flag_key = f"{output_name}_uncertainty_flag"
        if uncertainty_flag_key in predictions:
            frame[f"{target}_uncertainty_flag"] = predictions[uncertainty_flag_key]

        finite_interval = (
            np.isfinite(truth)
            & np.isfinite(lower)
            & np.isfinite(upper)
            & np.isfinite(calibrated)
        )
        covered = np.full(len(frame), np.nan)
        covered[finite_interval] = (
            (truth[finite_interval] >= lower[finite_interval])
            & (truth[finite_interval] <= upper[finite_interval])
        ).astype(float)
        frame[f"{target}_interval_covered"] = covered
        observed_coverage = (
            float(np.nanmean(covered)) if np.isfinite(covered).any() else None
        )
        if observed_coverage is None:
            assessment = "unavailable"
        elif observed_coverage < confidence - 0.02:
            assessment = "overconfident"
        elif observed_coverage > confidence + 0.02:
            assessment = "underconfident"
        else:
            assessment = "approximately_calibrated"
        valid_uq = np.isfinite(mc_sd) & np.isfinite(truth) & np.isfinite(calibrated)
        uncertainty_error_spearman = (
            _finite_statistic(
                spearmanr,
                mc_sd[valid_uq],
                np.abs(truth[valid_uq] - calibrated[valid_uq]),
            )
            if valid_uq.any()
            else None
        )
        diagnostics[target] = {
            "nominal_coverage": float(confidence),
            "observed_coverage": observed_coverage,
            "coverage_error": (
                observed_coverage - float(confidence)
                if observed_coverage is not None else None
            ),
            "confidence_assessment": assessment,
            "mean_interval_width": (
                float(np.nanmean(upper - lower))
                if np.isfinite(upper - lower).any() else None
            ),
            "mean_mc_sd": (
                float(np.nanmean(mc_sd)) if np.isfinite(mc_sd).any() else None
            ),
            "uncertainty_error_spearman": uncertainty_error_spearman,
            "ood_count": (
                int(np.nansum(frame[f"{target}_ood_flag"].astype(float)))
                if f"{target}_ood_flag" in frame else None
            ),
            "weak_local_calibration_count": (
                int(np.nansum(frame[f"{target}_local_calibration_weak"].astype(float)))
                if f"{target}_local_calibration_weak" in frame else None
            ),
            "uncertainty_toolbox": uncertainty_toolbox_metrics(
                truth, mc, mc_sd
            ),
        }

    summary["test_metrics"] = variants
    summary["test_uncertainty_diagnostics"] = diagnostics
    summary["test_calibration"] = {
        "source": "validation_only",
        "method": "local_validation_residual",
        "global_fallback": False,
        "confidence": float(confidence),
    }
    summary["test_flags"] = {
        "ood_flag": "empirical applicability score >= 0.95",
        "local_calibration_weak": (
            "similarity-qualified validation neighborhood was too small; "
            "nearest validation neighbors were used"
        ),
        "uncertainty_flag": (
            "MC-dropout radius exceeds the local validation-residual radius"
        ),
    }
    frame.to_csv(workdir / "test_predictions.csv", index=False)
    try:
        from chemflow.deep_learning.uncertainty_report import (
            generate_uncertainty_report,
        )

        summary["uncertainty_report"] = generate_uncertainty_report(
            workdir=workdir,
            predictions=frame,
            targets=targets,
            task_types=task_types,
            confidence=confidence,
        )
        summary["uncertainty_report"]["status"] = "completed"
    except Exception as error:
        # A plotting backend problem should never destroy a completed training
        # run or its numerical predictions. Record the failure prominently.
        summary["uncertainty_report"] = {
            "status": "failed",
            "output_directory": str(workdir / "uncertainty"),
            "error": f"{type(error).__name__}: {error}",
        }
    (workdir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return frame, summary
