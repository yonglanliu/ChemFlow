from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class RegressionEvaluator:
    """
    Evaluator for regression tasks.
    """

    @torch.no_grad()
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float | None = None,
    ) -> Dict[str, float]:
        """
        Compute regression metrics.

        Args:
            predictions:
                Regression predictions.

                Shape:
                    (B,)
                    (B,1)
                    (B,num_targets)

            targets:
                Ground-truth values.

                Same shape as predictions.

            loss:
                Optional validation loss computed by the model.

        Returns:
            Dictionary containing regression metrics.
        """

        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        if predictions.ndim == 2 and predictions.size(-1) == 1:
            predictions = predictions.squeeze(-1)

        if targets.ndim == 2 and targets.size(-1) == 1:
            targets = targets.squeeze(-1)

        y_pred = predictions.numpy()
        y_true = targets.numpy()

        metrics: Dict[str, float] = {}

        if loss is not None:
            metrics["val_loss"] = float(loss)

        metrics["val_mae"] = mean_absolute_error(
            y_true,
            y_pred,
        )

        metrics["val_rmse"] = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        try:
            metrics["val_r2"] = r2_score(
                y_true,
                y_pred,
            )
        except ValueError:
            metrics["val_r2"] = float("nan")

        return metrics