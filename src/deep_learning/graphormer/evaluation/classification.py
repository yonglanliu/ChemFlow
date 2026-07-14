from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class ClassificationEvaluator:
    """
    Evaluator for binary and multi-class classification.

    Args:
        loss_type:
            "bce" or "cross_entropy"

        num_classes:
            Number of output classes.
    """

    def __init__(
        self,
        loss_type: str,
        num_classes: int,
    ) -> None:
        self.loss_type = loss_type.lower()
        self.num_classes = num_classes

    @torch.no_grad()
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float | None = None,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions:
                Model outputs.

                BCE:
                    (B,) or (B,1)

                CrossEntropy:
                    (B,C)

            targets:
                Ground truth labels.

            loss:
                Optional validation loss.

        Returns:
            Dictionary of metrics.
        """

        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        metrics: Dict[str, float] = {}

        if loss is not None:
            metrics["val_loss"] = float(loss)

        # ==========================================================
        # Binary classification (BCE)
        # ==========================================================
        if self.loss_type == "bce":

            predictions = predictions.squeeze(-1)

            if targets.ndim == 2:
                targets = targets.squeeze(-1)

            probabilities = torch.sigmoid(predictions)

            predicted_labels = (
                probabilities >= 0.5
            ).long()

            y_true = targets.numpy()
            y_pred = predicted_labels.numpy()
            y_prob = probabilities.numpy()

            metrics["val_accuracy"] = accuracy_score(
                y_true,
                y_pred,
            )

            metrics["val_precision"] = precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            metrics["val_recall"] = recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            metrics["val_f1"] = f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            try:
                metrics["val_auroc"] = roc_auc_score(
                    y_true,
                    y_prob,
                )
            except ValueError:
                metrics["val_auroc"] = float("nan")

            return metrics

        # ==========================================================
        # Multi-class classification
        # ==========================================================

        probabilities = torch.softmax(
            predictions,
            dim=-1,
        )

        predicted_labels = probabilities.argmax(
            dim=-1
        )

        y_true = targets.numpy()
        y_pred = predicted_labels.numpy()
        y_prob = probabilities.numpy()

        metrics["val_accuracy"] = accuracy_score(
            y_true,
            y_pred,
        )

        metrics["val_precision"] = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        metrics["val_recall"] = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        metrics["val_f1"] = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        try:
            metrics["val_auroc"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            metrics["val_auroc"] = float("nan")

        return metrics