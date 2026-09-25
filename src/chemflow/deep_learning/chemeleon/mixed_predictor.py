"""ChemProp predictor supporting regression and binary tasks in one head."""

from __future__ import annotations

import torch
from chemprop.nn.metrics import ChempropMetric
from chemprop.nn.predictors import _FFNPredictorBase


REGRESSION_LOSSES = ("l2", "mse", "mae", "huber", "nll", "gaussian_nll")


def _regression_loss_values(
    preds,
    targets,
    *,
    name: str,
    huber_delta: float,
    nll_scale: float,
    gaussian_nll_variance: float,
):
    """Calculate an unreduced point or fixed-noise regression loss."""
    if name in {"l2", "mse"}:
        return (preds - targets).square()
    if name == "mae":
        return (preds - targets).abs()
    if name == "huber":
        return torch.nn.functional.huber_loss(
            preds, targets, reduction="none", delta=huber_delta
        )
    if name == "nll":
        scale = preds.new_tensor(nll_scale)
        return (preds - targets).abs() / scale + torch.log(2.0 * scale)
    if name == "gaussian_nll":
        return torch.nn.functional.gaussian_nll_loss(
            preds,
            targets,
            torch.full_like(preds, gaussian_nll_variance),
            full=True,
            reduction="none",
        )
    raise ValueError(f"Unsupported regression loss: {name!r}")


class RegressionLoss(ChempropMetric):
    """Configurable regression loss compatible with Chemprop masking."""

    def __init__(
        self,
        task_weights=1.0,
        name: str = "mse",
        huber_delta: float = 1.0,
        nll_scale: float = 1.0,
        gaussian_nll_variance: float = 1.0,
    ):
        super().__init__(task_weights=task_weights)
        self.name = str(name).strip().lower()
        self.huber_delta = float(huber_delta)
        self.nll_scale = float(nll_scale)
        self.gaussian_nll_variance = float(gaussian_nll_variance)

    def _calc_unreduced_loss(self, preds, targets, *args):
        return _regression_loss_values(
            preds,
            targets,
            name=self.name,
            huber_delta=self.huber_delta,
            nll_scale=self.nll_scale,
            gaussian_nll_variance=self.gaussian_nll_variance,
        )


class HuberLoss(RegressionLoss):
    """Backward-compatible Huber criterion used by existing callers."""

    def __init__(self, task_weights=1.0, delta: float = 1.0):
        super().__init__(
            task_weights=task_weights,
            name="huber",
            huber_delta=delta,
        )


class MixedTaskLoss(ChempropMetric):
    """Configurable regression loss and BCE-with-logits for binary columns."""

    def __init__(
        self,
        task_types,
        task_weights=1.0,
        regression_loss: str = "mse",
        huber_delta: float = 1.0,
        nll_scale: float = 1.0,
        gaussian_nll_variance: float = 1.0,
    ):
        super().__init__(task_weights=task_weights)
        self.task_types = [str(value) for value in task_types]
        self.regression_loss = str(regression_loss).strip().lower()
        self.huber_delta = float(huber_delta)
        self.nll_scale = float(nll_scale)
        self.gaussian_nll_variance = float(gaussian_nll_variance)

    def _calc_unreduced_loss(self, preds, targets, *args):
        loss = torch.zeros_like(preds)
        for index, task_type in enumerate(self.task_types):
            if task_type == "classification":
                loss[:, index] = torch.nn.functional.binary_cross_entropy_with_logits(
                    preds[:, index], targets[:, index], reduction="none"
                )
            else:
                loss[:, index] = _regression_loss_values(
                    preds[:, index],
                    targets[:, index],
                    name=self.regression_loss,
                    huber_delta=self.huber_delta,
                    nll_scale=self.nll_scale,
                    gaussian_nll_variance=self.gaussian_nll_variance,
                )
        return loss


class MixedTaskMetric(ChempropMetric):
    """Combined validation loss for normalized regression values and logits."""

    def __init__(
        self,
        task_types,
        task_weights=1.0,
        regression_loss: str = "mse",
        huber_delta: float = 1.0,
        nll_scale: float = 1.0,
        gaussian_nll_variance: float = 1.0,
    ):
        super().__init__(task_weights=task_weights)
        self.task_types = [str(value) for value in task_types]
        self.regression_loss = str(regression_loss).strip().lower()
        self.huber_delta = float(huber_delta)
        self.nll_scale = float(nll_scale)
        self.gaussian_nll_variance = float(gaussian_nll_variance)

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
                loss[:, index] = _regression_loss_values(
                    preds[:, index],
                    targets[:, index],
                    name=self.regression_loss,
                    huber_delta=self.huber_delta,
                    nll_scale=self.nll_scale,
                    gaussian_nll_variance=self.gaussian_nll_variance,
                )
        return loss


class MixedTaskFFN(_FFNPredictorBase):
    """One FFN whose columns receive task-specific output transforms."""

    n_targets = 1
    _T_default_criterion = MixedTaskLoss
    _T_default_metric = MixedTaskMetric

    def __init__(
        self,
        *,
        task_types,
        task_weights=None,
        criterion=None,
        regression_loss: str = "mse",
        huber_delta: float = 1.0,
        nll_scale: float = 1.0,
        gaussian_nll_variance: float = 1.0,
        **kwargs,
    ):
        self.task_types = [str(value) for value in task_types]
        kwargs.pop("n_tasks", None)
        if task_weights is None:
            task_weights = torch.ones(len(self.task_types))
        criterion = criterion or MixedTaskLoss(
            self.task_types,
            task_weights,
            regression_loss=regression_loss,
            huber_delta=huber_delta,
            nll_scale=nll_scale,
            gaussian_nll_variance=gaussian_nll_variance,
        )
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
