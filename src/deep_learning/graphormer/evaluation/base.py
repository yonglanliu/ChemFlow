# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from abc import ABC, abstractmethod
import torch


class GraphormerEvaluator(ABC):
    """
    Abstract base class for Graphormer evaluators.

    Evaluators receive model predictions, ground-truth targets, and the
    validation loss, then return a dictionary containing scalar metrics.

    Notes
    -----
    All returned values should be Python floats because the trainer saves
    them to JSON, CSV, and checkpoints.
    """

    @abstractmethod
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        prefix: str = "",
    ) -> dict[str, float]:
        """
        Compute evaluation metrics.

        Parameters
        ----------
        predictions
            Raw model outputs. These are normally logits for classification
            or predicted values for regression.

        targets
            Ground-truth labels or regression values.

        loss
            Validation loss calculated by the model.

        Returns
        -------
        dict[str, float]
            Mapping from metric name to scalar value.
        """
        raise NotImplementedError