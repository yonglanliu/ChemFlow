# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

ClassificationTask = Literal[
    "binary",
    "multiclass",
    "multilabel",
]
from src.deep_learning.graphormer.evaluation.base import GraphormerEvaluator

class ClassificationEvaluator(GraphormerEvaluator):
    """
    Evaluator for binary, multiclass, and multilabel classification.

    Parameters
    ----------
    loss_type
        Classification task type:

        - ``"binary"``:
          One binary label per sample. Predictions may have shape ``(N,)``,
          ``(N, 1)``, or ``(N, 2)``.

        - ``"multiclass"``:
          Exactly one class per sample. Predictions must have shape
          ``(N, C)``.

        - ``"multilabel"``:
          Multiple independent binary labels per sample. Predictions and
          targets must have shape ``(N, C)``.

    num_classes
        Number of classes. Required for multiclass classification.

    threshold
        Probability threshold used to convert binary or multilabel
        probabilities into class predictions.

    average
        Averaging strategy for multiclass or multilabel precision, recall,
        and F1. Common values are ``"macro"``, ``"weighted"``, and ``"micro"``.

    positive_class
        Column representing the positive class when binary predictions have
        shape ``(N, 2)``.
    """

    VALID_TASKS = {"binary", "multiclass", "multilabel",}

    VALID_AVERAGES = {"micro", "macro", "weighted", "samples",}

    def __init__(
        self,
        loss_type: ClassificationTask = "binary",
        num_classes: int = 2,
        threshold: float = 0.5,
        average: str = "macro",
        positive_class: int = 1,
    ) -> None:
        self.loss_type = str(loss_type).lower()
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.average = str(average).lower()
        self.positive_class = int(positive_class)

        if self.loss_type not in self.VALID_TASKS:
            raise ValueError(
                f"Unsupported loss_type/task '{loss_type}'. "
                f"Expected one of {sorted(self.VALID_TASKS)}."
            )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}.")

        if self.average not in self.VALID_AVERAGES:
            raise ValueError(
                f"Unsupported average '{average}'. "
                f"Expected one of {sorted(self.VALID_AVERAGES)}."
            )

        if self.loss_type == "multiclass" and self.num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2 for multiclass "
                "classification."
            )

        if self.loss_type == "binary" and self.positive_class not in {0, 1}:
            raise ValueError(
                "positive_class must be either 0 or 1 for binary "
                "classification."
            )

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        prefix: str = "val",
    ) -> dict[str, float]:
        """
        Compute classification metrics.
        """
        predictions = self._validate_tensor(predictions, name="predictions",)
        targets = self._validate_tensor(targets, name="targets",)

        if self.loss_type == "binary":
            metrics = self._compute_binary(predictions=predictions, targets=targets,)

        elif self.loss_type == "multiclass":
            metrics = self._compute_multiclass(predictions=predictions, targets=targets,)

        else:
            metrics = self._compute_multilabel(predictions=predictions, targets=targets,)

        renamed_metrics = {f"{prefix}_{name}": float(value) for name, value in metrics.items()}

        return {f"{prefix}_loss": float(loss), **renamed_metrics}

    def _compute_binary(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:

        """
        Compute metrics for binary classification.
        """
        probabilities = self._binary_probabilities(predictions)

        true_labels = (targets.reshape(-1).to(dtype=torch.long).numpy())

        probabilities_np = probabilities.numpy()

        predicted_labels = (probabilities_np >= self.threshold).astype(np.int64)

        if probabilities_np.shape[0] != true_labels.shape[0]:
            raise ValueError(
                "Binary predictions and targets contain different numbers "
                f"of samples: {probabilities_np.shape[0]} and "
                f"{true_labels.shape[0]}."
            )

        self._validate_binary_targets(true_labels)

        metrics = {
            "accuracy": accuracy_score(
                true_labels,
                predicted_labels,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                true_labels,
                predicted_labels,
            ),
            "precision": precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),
            "recall": recall_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),
            "f1": f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            ),
            "mcc": matthews_corrcoef(
                true_labels,
                predicted_labels,
            ),
            "roc_auc": self._safe_binary_roc_auc(
                true_labels,
                probabilities_np,
            ),
            "pr_auc": self._safe_binary_pr_auc(
                true_labels,
                probabilities_np,
            ),
        }

        return self._convert_metrics_to_float(metrics)

    def _compute_multiclass(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute metrics for single-label multiclass classification.
        """
        if predictions.ndim != 2:
            raise ValueError(
                "Multiclass predictions must have shape (N, C), "
                f"got {tuple(predictions.shape)}."
            )

        if predictions.shape[1] != self.num_classes:
            raise ValueError(
                "Prediction class dimension does not match num_classes: "
                f"{predictions.shape[1]} versus {self.num_classes}."
            )

        probabilities = torch.softmax(predictions, dim=-1).numpy()

        predicted_labels = probabilities.argmax(axis=1)

        true_labels = (targets.reshape(-1).to(dtype=torch.long).numpy())

        if probabilities.shape[0] != true_labels.shape[0]:
            raise ValueError(
                "Multiclass predictions and targets contain different "
                f"numbers of samples: {probabilities.shape[0]} and "
                f"{true_labels.shape[0]}."
            )

        self._validate_multiclass_targets(true_labels)

        one_hot_targets = np.eye(self.num_classes, dtype=np.float64,)[true_labels]

        metrics = {
            "accuracy": accuracy_score(
                true_labels,
                predicted_labels,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                true_labels,
                predicted_labels,
            ),
            "precision": precision_score(
                true_labels,
                predicted_labels,
                average=self.average,
                zero_division=0,
            ),
            "recall": recall_score(
                true_labels,
                predicted_labels,
                average=self.average,
                zero_division=0,
            ),
            "f1": f1_score(
                true_labels,
                predicted_labels,
                average=self.average,
                zero_division=0,
            ),
            "mcc": matthews_corrcoef(
                true_labels,
                predicted_labels,
            ),
            "roc_auc": self._safe_multiclass_roc_auc(
                true_labels=true_labels,
                probabilities=probabilities,
            ),
            "pr_auc": self._safe_multiclass_pr_auc(
                one_hot_targets=one_hot_targets,
                probabilities=probabilities,
            ),
        }

        return self._convert_metrics_to_float(metrics)

    def _compute_multilabel(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute metrics for multilabel classification.
        """
        if predictions.ndim != 2:
            raise ValueError(
                "Multilabel predictions must have shape (N, C), "
                f"got {tuple(predictions.shape)}."
            )

        if targets.ndim != 2:
            raise ValueError(
                "Multilabel targets must have shape (N, C), "
                f"got {tuple(targets.shape)}."
            )

        if predictions.shape != targets.shape:
            raise ValueError(
                "Multilabel predictions and targets must have the same "
                f"shape, got {tuple(predictions.shape)} and "
                f"{tuple(targets.shape)}."
            )

        probabilities = torch.sigmoid(predictions).numpy()

        true_labels = (targets.to(dtype=torch.long).numpy())

        predicted_labels = (probabilities >= self.threshold).astype(np.int64)

        self._validate_multilabel_targets(true_labels)

        metrics = {
            # Subset accuracy: every label for a sample must be correct.
            "accuracy": accuracy_score(
                true_labels,
                predicted_labels,
            ),
            "precision": precision_score(
                true_labels,
                predicted_labels,
                average=self.average,
                zero_division=0,
            ),
            "recall": recall_score(
                true_labels,
                predicted_labels,
                average=self.average,
                zero_division=0,
            ),
            "f1": f1_score(
                true_labels,
                predicted_labels,
                average=self.average,
                zero_division=0,
            ),
            "mcc": self._multilabel_mcc(
                true_labels,
                predicted_labels,
            ),
            "roc_auc": self._safe_multilabel_roc_auc(
                true_labels,
                probabilities,
            ),
            "pr_auc": self._safe_multilabel_pr_auc(
                true_labels,
                probabilities,
            ),
        }

        return self._convert_metrics_to_float(metrics)

    def _binary_probabilities(
        self,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert binary logits into positive-class probabilities.

        Accepted shapes
        ---------------
        (N,)
            One logit per sample. Sigmoid is applied.

        (N, 1)
            One logit per sample. Sigmoid is applied.

        (N, 2)
            Two class logits per sample. Softmax is applied and the selected
            positive-class column is returned.
        """
        if predictions.ndim == 1: # (N,)
            return torch.sigmoid(predictions)

        if predictions.ndim != 2:
            raise ValueError(
                "Binary predictions must have shape (N,), (N, 1), or "
                f"(N, 2), got {tuple(predictions.shape)}."
            )

        if predictions.shape[1] == 1: # (N, 1)
            return torch.sigmoid(predictions[:, 0])

        if predictions.shape[1] == 2: # (N, 2)
            probabilities = torch.softmax(predictions, dim=-1,)
            return probabilities[:, self.positive_class]

        raise ValueError(
            "Binary predictions must have one or two output columns, "
            f"got {predictions.shape[1]}."
        )

    def _validate_binary_targets(
        self,
        targets: np.ndarray,
    ) -> None:
        unique_values = np.unique(targets)

        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(
                "Binary targets must contain only 0 and 1, "
                f"got {unique_values.tolist()}."
            )

    def _validate_multiclass_targets(
        self,
        targets: np.ndarray,
    ) -> None:
        if targets.size == 0:
            raise ValueError("Multiclass targets are empty.")

        minimum = int(targets.min())
        maximum = int(targets.max())

        if minimum < 0 or maximum >= self.num_classes:
            raise ValueError(
                "Multiclass targets must be in the range "
                f"[0, {self.num_classes - 1}], got minimum={minimum}, "
                f"maximum={maximum}."
            )

    @staticmethod
    def _validate_multilabel_targets(
        targets: np.ndarray,
    ) -> None:
        unique_values = np.unique(targets)

        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(
                "Multilabel targets must contain only 0 and 1, "
                f"got {unique_values.tolist()}."
            )

    @staticmethod
    def _safe_binary_roc_auc(
        targets: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        if np.unique(targets).size < 2:
            return float("nan")

        try:
            return float(
                roc_auc_score(
                    targets,
                    probabilities,
                )
            )
        except ValueError:
            return float("nan")

    @staticmethod
    def _safe_binary_pr_auc(
        targets: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        if np.unique(targets).size < 2:
            return float("nan")

        try:
            return float(
                average_precision_score(
                    targets,
                    probabilities,
                )
            )
        except ValueError:
            return float("nan")

    def _safe_multiclass_roc_auc(
        self,
        true_labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        # ROC-AUC cannot be calculated if the validation split does not
        # contain all configured classes.
        if np.unique(true_labels).size != self.num_classes:
            return float("nan")

        try:
            return float(
                roc_auc_score(
                    true_labels,
                    probabilities,
                    multi_class="ovr",
                    average=self.average,
                    labels=np.arange(self.num_classes),
                )
            )
        except ValueError:
            return float("nan")

    def _safe_multiclass_pr_auc(
        self,
        one_hot_targets: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        valid_columns = (
            one_hot_targets.sum(axis=0) > 0
        ) & (
            one_hot_targets.sum(axis=0) < one_hot_targets.shape[0]
        )

        if not np.any(valid_columns):
            return float("nan")

        try:
            return float(
                average_precision_score(
                    one_hot_targets[:, valid_columns],
                    probabilities[:, valid_columns],
                    average=self.average,
                )
            )
        except ValueError:
            return float("nan")

    def _safe_multilabel_roc_auc(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        valid_columns = np.array(
            [
                np.unique(targets[:, index]).size == 2
                for index in range(targets.shape[1])
            ],
            dtype=bool,
        )

        if not np.any(valid_columns):
            return float("nan")

        try:
            return float(
                roc_auc_score(
                    targets[:, valid_columns],
                    probabilities[:, valid_columns],
                    average=self.average,
                )
            )
        except ValueError:
            return float("nan")

    def _safe_multilabel_pr_auc(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        valid_columns = np.array(
            [
                np.unique(targets[:, index]).size == 2
                for index in range(targets.shape[1])
            ],
            dtype=bool,
        )

        if not np.any(valid_columns):
            return float("nan")

        try:
            return float(
                average_precision_score(
                    targets[:, valid_columns],
                    probabilities[:, valid_columns],
                    average=self.average,
                )
            )
        except ValueError:
            return float("nan")

    @staticmethod
    def _multilabel_mcc(
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        """
        Compute macro-averaged MCC over multilabel output columns.
        """
        scores = []

        for index in range(targets.shape[1]):
            target_column = targets[:, index]
            prediction_column = predictions[:, index]

            if np.unique(target_column).size < 2:
                continue

            scores.append(
                matthews_corrcoef(
                    target_column,
                    prediction_column,
                )
            )

        if not scores:
            return float("nan")

        return float(np.mean(scores))

    @staticmethod
    def _validate_tensor(
        value: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(
                f"{name} must be a torch.Tensor, "
                f"got {type(value).__name__}."
            )

        if value.numel() == 0:
            raise ValueError(f"{name} is empty.")

        return value.detach().cpu()

    @staticmethod
    def _convert_metrics_to_float(
        metrics: dict[str, float],
    ) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in metrics.items()
        }

    def compute_curve_data(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        if self.loss_type == "binary":
            probabilities = self._binary_probabilities(
                predictions
            ).numpy().reshape(-1)

            true_labels = (
                targets
                .numpy()
                .reshape(-1)
                .astype(np.int64)
            )

            predicted_labels = (
                probabilities >= self.threshold
            ).astype(np.int64)

            fpr, tpr, roc_thresholds = roc_curve(
                true_labels,
                probabilities,
            )

            precision, recall, pr_thresholds = (
                precision_recall_curve(
                    true_labels,
                    probabilities,
                )
            )

            cm = confusion_matrix(
                true_labels,
                predicted_labels,
                labels=[0, 1],
            )

            return {
                "roc": {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": roc_thresholds,
                    "auc": roc_auc_score(
                        true_labels,
                        probabilities,
                    ),
                },
                "pr": {
                    "precision": precision,
                    "recall": recall,
                    "thresholds": pr_thresholds,
                    "average_precision": average_precision_score(
                        true_labels,
                        probabilities,
                    ),
                    "positive_prevalence": float(
                        np.mean(true_labels == 1)
                    ),
                },
                "confusion_matrix": cm,
                "class_names": [
                    "Negative",
                    "Positive",
                ],
            }

