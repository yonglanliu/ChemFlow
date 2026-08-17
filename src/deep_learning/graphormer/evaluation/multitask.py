# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from src.deep_learning.graphormer.evaluation.base import GraphormerEvaluator


class MultiTaskEvaluator(GraphormerEvaluator):
    """
    Evaluator for multi-task regression.

    Computes metrics for each task independently and reports
    the mean metric across tasks as the overall metric.

    Expected inputs
    ---------------
    outputs:
        Mapping from task name to prediction tensor.

        Example:
            {
                "task_0": Tensor[N],
                "task_1": Tensor[N],
                ...
            }

        Prediction tensors with shape [N, 1] are also supported.

    targets:
        Tensor with shape [N, num_tasks].

    task_names:
        Optional user-defined task names. If provided, outputs are
        expected to use the default keys ``task_0``, ``task_1``, ...
        and will be mapped to the supplied task names.
    """

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        task_names: list[str] | None = None,
        prefix: str = "val",
    ) -> dict[str, float]:

        # ---------------------------------------------------------
        # Validate inputs
        # ---------------------------------------------------------

        if not torch.is_tensor(predictions):
            raise TypeError(
                f"predictions must be a torch.Tensor, "
                f"got {type(predictions).__name__}."
            )

        if not torch.is_tensor(targets):
            raise TypeError(
                f"targets must be a torch.Tensor, "
                f"got {type(targets).__name__}."
            )

        if targets.numel() == 0:
            raise ValueError("targets is empty.")

        if targets.ndim != 2:
            raise ValueError(
                "targets must have shape "
                "[num_samples, num_tasks], "
                f"got {tuple(targets.shape)}."
            )

        num_tasks = targets.shape[1]

        # ---------------------------------------------------------
        # Resolve task names
        # ---------------------------------------------------------

        default_task_names = [
            f"task_{i}"
            for i in range(num_tasks)
        ]

        if task_names is None:
            task_names = default_task_names

        else:
            if not isinstance(task_names, list):
                raise TypeError(
                    "task_names must be a list of strings."
                )

            if not all(
                isinstance(name, str)
                for name in task_names
            ):
                raise TypeError(
                    "All elements in task_names must be strings."
                )

            if len(task_names) != num_tasks:
                raise ValueError(
                    f"Length of task_names ({len(task_names)}) "
                    f"does not match number of target tasks "
                    f"({num_tasks})."
                )

            # Rename default output keys to user-defined task names.
            predictions = {task_name: predictions[:, i:i+1] for i, task_name in enumerate(task_names)}

        print(
            f"Computing metrics for tasks: {task_names}."
        )

        # ---------------------------------------------------------
        # Compute per-task metrics
        # ---------------------------------------------------------

        metrics: dict[str, float] = {}

        task_metrics = {
            "mae": [],
            "rmse": [],
            "medae": [],
            "r2": [],
            "pearsonr": [],
            "spearmanr": [],
            "kendalltau": [],
        }

        for task_index, task_name in enumerate(task_names):

            if task_name not in predictions:
                raise ValueError(
                    f"Missing output for '{task_name}'. "
                    f"Available output keys: "
                    f"{list(predictions.keys())}."
                )

            task_predictions = self._prepare_values(
                predictions[task_name],
                name=f"predictions['{task_name}']",
            )

            task_targets = self._prepare_values(
                targets[:, task_index],
                name=f"targets[:, {task_index}]",
            )

            if task_predictions.shape != task_targets.shape:
                raise ValueError(
                    f"Predictions and targets for "
                    f"'{task_name}' must have the same shape. "
                    f"Got predictions shape "
                    f"{task_predictions.shape} and targets shape "
                    f"{task_targets.shape}."
                )

            # ---------------------------------------------------------
            # Mask missing values (NaN) in predictions and targets.
            # ---------------------------------------------------------
            valid_mask = np.isfinite(task_targets) & np.isfinite(task_predictions)
            task_targets = task_targets[valid_mask]
            task_predictions = task_predictions[valid_mask]
            num_valid = len(task_targets)

            if num_valid == 0:
                metrics[f"{prefix}_{task_name}_mae"] = np.nan
                metrics[f"{prefix}_{task_name}_rmse"] = np.nan
                metrics[f"{prefix}_{task_name}_medae"] = np.nan
                metrics[f"{prefix}_{task_name}_r2"] = np.nan
                metrics[f"{prefix}_{task_name}_pearsonr"] = np.nan
                metrics[f"{prefix}_{task_name}_spearmanr"] = np.nan
                metrics[f"{prefix}_{task_name}_kendalltau"] = np.nan
                continue

            mae = mean_absolute_error(task_targets, task_predictions)
            rmse = np.sqrt(mean_squared_error(task_targets, task_predictions))
            medae = median_absolute_error(task_targets, task_predictions)
            r2 = r2_score(task_targets, task_predictions)
            pearson = pearsonr(task_targets, task_predictions)[0]
            spearman = spearmanr(task_targets, task_predictions)[0]
            kendall = kendalltau(task_targets, task_predictions)[0]

            # Store individual task metrics.
            metrics.update(
                {
                    f"{prefix}_{task_name}_mae": float(mae),
                    f"{prefix}_{task_name}_rmse": float(rmse),
                    f"{prefix}_{task_name}_medae": float(medae),
                    f"{prefix}_{task_name}_r2": float(r2),
                    f"{prefix}_{task_name}_pearsonr": float(pearson),
                    f"{prefix}_{task_name}_spearmanr": float(spearman),
                    f"{prefix}_{task_name}_kendalltau": float(kendall),
                }
            )

            # Collect metrics for overall averages.
            task_metrics["mae"].append(mae)
            task_metrics["rmse"].append(rmse)
            task_metrics["medae"].append(medae)
            task_metrics["r2"].append(r2)
            task_metrics["pearsonr"].append(pearson)
            task_metrics["spearmanr"].append(spearman)
            task_metrics["kendalltau"].append(kendall)

        # ---------------------------------------------------------
        # Overall metrics
        #
        # Average task-level metrics rather than flattening all
        # tasks together.
        # ---------------------------------------------------------

        metrics[f"{prefix}_loss"] = float(loss)
        for metric_name, values in task_metrics.items():
            metrics[f"{prefix}_{metric_name}"] = float(np.nanmean(values))

        return metrics

    @staticmethod
    def _prepare_values(
        value: torch.Tensor,
        name: str,
    ) -> np.ndarray:
        """
        Convert a single-task prediction or target tensor
        to a 1D NumPy array.

        Accepted shapes:
            [N]
            [N, 1]

        Returns
        -------
        np.ndarray
            Array with shape [N].
        """

        if not torch.is_tensor(value):
            raise TypeError(
                f"{name} must be a torch.Tensor, "
                f"got {type(value).__name__}."
            )

        if value.numel() == 0:
            raise ValueError(
                f"{name} is empty."
            )

        if value.ndim == 2:
            if value.shape[1] != 1:
                raise ValueError(
                    f"{name} must have shape [N] or [N, 1], "
                    f"got {tuple(value.shape)}."
                )

            value = value.squeeze(1)

        elif value.ndim != 1:
            raise ValueError(
                f"{name} must have shape [N] or [N, 1], "
                f"got {tuple(value.shape)}."
            )

        return (value.detach().cpu().to(dtype=torch.float64).numpy())