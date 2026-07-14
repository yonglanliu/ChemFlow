# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import pearsonr, spearmanr

from src.deep_learning.graphormer.evaluation.base import GraphormerEvaluator

class RegressionEvaluator(GraphormerEvaluator):
    """
    Evaluator for regression tasks.

    Computes:

    - Validation loss
    - Mean absolute error
    - Root mean squared error
    - Median absolute error
    - R-squared
    - Pearson correlation
    - Spearman rank correlation
    """

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        prefix: str = "val",
    ) -> dict[str, float]:
        predictions = self._prepare_values(
            predictions,
            name="predictions",
        )
        targets = self._prepare_values(
            targets,
            name="targets",
        )

        if predictions.shape != targets.shape:
            raise ValueError(
                "Regression predictions and targets must have the same "
                f"number of values, got {predictions.shape} and "
                f"{targets.shape}."
            )

        valid_mask = (
            np.isfinite(predictions)
            & np.isfinite(targets)
        )

        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        if targets.size == 0:
            raise ValueError(
                "No finite regression predictions and targets remain."
            )

        mae = mean_absolute_error(
            targets,
            predictions,
        )

        rmse = mean_squared_error(
            targets,
            predictions,
        ) ** 0.5

        median_ae = median_absolute_error(
            targets,
            predictions,
        )

        r2 = self._safe_r2(
            targets,
            predictions,
        )

        pearson = self._safe_pearson(
            targets,
            predictions,
        )

        spearman = self._safe_spearman(
            targets,
            predictions,
        )

        return {
            f"{prefix}_loss": float(loss),
            f"{prefix}_mae": float(mae),
            f"{prefix}_rmse": float(rmse),
            f"{prefix}_median_ae": float(median_ae),
            f"{prefix}_r2": float(r2),
            f"{prefix}_pearson": float(pearson),
            f"{prefix}_spearman": float(spearman),
        }

    @staticmethod
    def _prepare_values(
        value: torch.Tensor,
        name: str,
    ) -> np.ndarray:
        if not torch.is_tensor(value):
            raise TypeError(
                f"{name} must be a torch.Tensor, "
                f"got {type(value).__name__}."
            )

        if value.numel() == 0:
            raise ValueError(f"{name} is empty.")

        return (
            value
            .detach()
            .cpu()
            .to(dtype=torch.float64)
            .reshape(-1)
            .numpy()
        )

    @staticmethod
    def _safe_r2(
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        if targets.size < 2:
            return float("nan")

        try:
            return float(
                r2_score(
                    targets,
                    predictions,
                )
            )
        except ValueError:
            return float("nan")

    @staticmethod
    def _safe_pearson(
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        if targets.size < 2:
            return float("nan")

        if np.all(targets == targets[0]):
            return float("nan")

        if np.all(predictions == predictions[0]):
            return float("nan")

        try:
            result = pearsonr(
                targets,
                predictions,
            )
            return float(result.statistic)
        except (ValueError, FloatingPointError):
            return float("nan")

    @staticmethod
    def _safe_spearman(
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        if targets.size < 2:
            return float("nan")

        if np.all(targets == targets[0]):
            return float("nan")

        if np.all(predictions == predictions[0]):
            return float("nan")

        try:
            result = spearmanr(
                targets,
                predictions,
            )
            return float(result.statistic)
        except (ValueError, FloatingPointError):
            return float("nan")