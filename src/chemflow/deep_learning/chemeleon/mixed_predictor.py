"""ChemProp predictor supporting regression and binary tasks in one head."""

from __future__ import annotations

import torch
from chemprop.nn.metrics import ChempropMetric
from chemprop.nn.predictors import _FFNPredictorBase


class MixedTaskLoss(ChempropMetric):
    """MSE for regression columns and BCE-with-logits for binary columns."""

    def __init__(self, task_types, task_weights=1.0):
        super().__init__(task_weights=task_weights)
        self.task_types = [str(value) for value in task_types]

    def _calc_unreduced_loss(self, preds, targets, *args):
        loss = torch.zeros_like(preds)
        for index, task_type in enumerate(self.task_types):
            if task_type == "classification":
                loss[:, index] = torch.nn.functional.binary_cross_entropy_with_logits(
                    preds[:, index], targets[:, index], reduction="none"
                )
            else:
                loss[:, index] = (preds[:, index] - targets[:, index]).square()
        return loss


class MixedTaskMetric(ChempropMetric):
    """Combined validation loss for normalized regression values and logits."""

    def __init__(self, task_types, task_weights=1.0):
        super().__init__(task_weights=task_weights)
        self.task_types = [str(value) for value in task_types]

    def _calc_unreduced_loss(self, preds, targets, *args):
        loss = torch.zeros_like(preds)
        for index, task_type in enumerate(self.task_types):
            if task_type == "classification":
                loss[:, index] = torch.nn.functional.binary_cross_entropy_with_logits(
                    preds[:, index],
                    targets[:, index],
                    reduction="none",
                )
            else:
                loss[:, index] = (preds[:, index] - targets[:, index]).square()
        return loss


class MixedTaskFFN(_FFNPredictorBase):
    """One FFN whose columns receive task-specific output transforms."""

    n_targets = 1
    _T_default_criterion = MixedTaskLoss
    _T_default_metric = MixedTaskMetric

    def __init__(self, *, task_types, task_weights=None, criterion=None, **kwargs):
        self.task_types = [str(value) for value in task_types]
        kwargs.pop("n_tasks", None)
        if task_weights is None:
            task_weights = torch.ones(len(self.task_types))
        criterion = criterion or MixedTaskLoss(self.task_types, task_weights)
        super().__init__(
            n_tasks=len(self.task_types),
            task_weights=task_weights,
            criterion=criterion,
            **kwargs,
        )
        self.hparams["task_types"] = self.task_types
        self.register_buffer(
            "classification_mask",
            torch.tensor(
                [value == "classification" for value in self.task_types],
                dtype=torch.bool,
            ),
        )

    def forward(self, Z):
        values = self.output_transform(self.ffn(Z))
        if self.classification_mask.any():
            values = values.clone()
            values[:, self.classification_mask] = values[
                :, self.classification_mask
            ].sigmoid()
        return values

    def train_step(self, Z):
        return self.ffn(Z)
