from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from src.chemflow.uncertainty.metrics import METRICS
from src.chemflow.uncertainty.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    rmse_score,
    pearson_score,
    spearman_score,
    kendall_score,
)


# ============================================================
# Bootstrap result container
# ============================================================

@dataclass
class BootstrapResult:
    metric: str
    observed: float
    bootstrap_mean: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    samples: np.ndarray


# ============================================================
# Bootstrap one metric
# ============================================================

def bootstrap_metric(
    y_true,
    y_pred,
    *,
    metric: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:

    metric = metric.strip().lower()

    if metric not in METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Supported metrics are: {sorted(METRICS)}"
        )

    if n_bootstrap < 1:
        raise ValueError(
            "n_bootstrap must be greater than 0."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    y_true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    valid_mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[
        valid_mask
    ]

    y_pred = y_pred[
        valid_mask
    ]

    if len(y_true) < 2:
        raise ValueError(
            "At least two valid paired samples are required."
        )

    metric_fn = METRICS[
        metric
    ]

    observed = float(
        metric_fn(
            y_true,
            y_pred,
        )
    )

    rng = np.random.default_rng(
        seed
    )

    n_samples = len(
        y_true
    )

    bootstrap_scores: list[float] = []

    for _ in range(
        n_bootstrap
    ):
        indices = rng.integers(
            low=0,
            high=n_samples,
            size=n_samples,
        )

        try:
            score = metric_fn(
                y_true[indices],
                y_pred[indices],
            )

        except Exception:
            continue

        if np.isfinite(
            score
        ):
            bootstrap_scores.append(
                float(score)
            )

    scores = np.asarray(
        bootstrap_scores,
        dtype=float,
    )

    if len(scores) == 0:
        raise RuntimeError(
            f"No valid bootstrap values were generated "
            f"for metric '{metric}'."
        )

    alpha = (
        1.0
        - confidence_level
    )

    ci_lower = float(
        np.quantile(
            scores,
            alpha / 2,
        )
    )

    ci_upper = float(
        np.quantile(
            scores,
            1 - alpha / 2,
        )
    )

    standard_error = float(
        np.std(
            scores,
            ddof=1,
        )
    )

    return BootstrapResult(
        metric=metric,
        observed=observed,
        bootstrap_mean=float(
            np.mean(
                scores
            )
        ),
        standard_error=standard_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=len(
            scores
        ),
        samples=scores,
    )


# ============================================================
# Bootstrap several metrics for one task
# ============================================================

def evaluate_bootstrap_metrics(
    y_true,
    y_pred,
    *,
    metrics: list[str],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
]:

    records: list[dict] = []

    distributions: dict[
        str,
        np.ndarray,
    ] = {}

    for metric in metrics:
        result = bootstrap_metric(
            y_true=y_true,
            y_pred=y_pred,
            metric=metric,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )

        records.append(
            {
                "metric": result.metric,
                "observed": result.observed,
                "bootstrap_mean": result.bootstrap_mean,
                "standard_error": result.standard_error,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "n_bootstrap": result.n_bootstrap,
            }
        )

        distributions[
            result.metric
        ] = result.samples

    return (
        pd.DataFrame(
            records
        ),
        distributions,
    )


# ============================================================
# Multi-task bootstrap
# ============================================================

def evaluate_multitask_bootstrap(
    frame: pd.DataFrame,
    task_columns: dict[
        str,
        tuple[str, str],
    ],
    *,
    metrics: list[str],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    dict[
        str,
        dict[str, np.ndarray],
    ],
]:

    summary_frames: list[
        pd.DataFrame
    ] = []

    distributions: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    for (
        task_name,
        (
            target_column,
            prediction_column,
        ),
    ) in task_columns.items():

        if target_column not in frame.columns:
            raise KeyError(
                f"Target column '{target_column}' "
                f"was not found for task '{task_name}'."
            )

        if prediction_column not in frame.columns:
            raise KeyError(
                f"Prediction column '{prediction_column}' "
                f"was not found for task '{task_name}'."
            )

        task_frame = (
            frame[
                [
                    target_column,
                    prediction_column,
                ]
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        if task_frame.empty:
            continue

        y_true = task_frame[
            target_column
        ].to_numpy(
            dtype=float
        )

        y_pred = task_frame[
            prediction_column
        ].to_numpy(
            dtype=float
        )

        (
            task_metrics,
            task_distributions,
        ) = evaluate_bootstrap_metrics(
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )

        task_metrics.insert(
            0,
            "task",
            task_name,
        )

        task_metrics[
            "target_column"
        ] = target_column

        task_metrics[
            "prediction_column"
        ] = prediction_column

        task_metrics[
            "n_samples"
        ] = len(
            task_frame
        )

        summary_frames.append(
            task_metrics
        )

        distributions[
            task_name
        ] = task_distributions

    if not summary_frames:
        return (
            pd.DataFrame(),
            {},
        )

    summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    return (
        summary,
        distributions,
    )
