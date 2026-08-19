from scipy.stats import pearsonr, spearmanr
from typing import Callable
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import kendalltau

# ============================================================
# Metric functions
# ============================================================

def rmse_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def pearson_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    if len(y_true) < 2:
        return np.nan

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return np.nan

    return float(
        pearsonr(
            y_true,
            y_pred,
        ).statistic
    )


def spearman_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    if len(y_true) < 2:
        return np.nan

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return np.nan

    return float(
        spearmanr(
            y_true,
            y_pred,
        ).statistic
    )


def kendall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    if len(y_true) < 2:
        return np.nan

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return np.nan

    return float(
        kendalltau(
            y_true,
            y_pred,
        ).statistic
    )


METRICS: dict[
    str,
    Callable[[np.ndarray, np.ndarray], float],
] = {
    "r2": r2_score,
    "mae": mean_absolute_error,
    "rmse": rmse_score,
    "pearson": pearson_score,
    "spearman": spearman_score,
    "kendall": kendall_score,
}
