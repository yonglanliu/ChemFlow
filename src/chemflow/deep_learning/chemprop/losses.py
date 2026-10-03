"""ChemFlow regression losses registered with Chemprop's native loss registry."""

from __future__ import annotations

import os
import math

import torch
from chemprop.nn.metrics import ChempropMetric, LossFunctionRegistry
from chemflow.deep_learning.censoring import censor_predictions_from_masks


def patch_chemprop_bounded_metric_masks() -> None:
    """Fix Chemprop 2.2's per-task test-mask broadcasting.

    Its test reporter slices predictions/targets as ``n x 1`` but inequality
    masks as ``n``. PyTorch then broadcasts a mask to ``n x n``, producing
    invalid bounded metrics and potentially very large temporary tensors.
    """
    from chemprop.nn.metrics import BoundedMixin

    if getattr(BoundedMixin, "_chemflow_mask_patch", False):
        return
    original = BoundedMixin._calc_unreduced_loss

    def corrected(self, preds, targets, mask, weights, lt_mask, gt_mask):
        if lt_mask is not None and lt_mask.ndim < targets.ndim:
            lt_mask = lt_mask.reshape(targets.shape)
        if gt_mask is not None and gt_mask.ndim < targets.ndim:
            gt_mask = gt_mask.reshape(targets.shape)
        return original(self, preds, targets, mask, weights, lt_mask, gt_mask)

    BoundedMixin._calc_unreduced_loss = corrected
    BoundedMixin._chemflow_mask_patch = True


def patch_chemprop_bounded_tracking_validation() -> None:
    """Allow a registered bounded metric to select Chemprop 2.2 checkpoints.

    Chemprop 2.2 checks ``tracking_metric.split("-")[0]`` against the complete
    metric aliases. Consequently, it rejects ``bounded-mae`` as ``bounded``
    even when ``bounded-mae`` is present in ``--metrics``. Later training code
    correctly looks up and monitors the complete alias, so only the faulty
    validation check needs this compatibility shim.
    """
    from chemprop.cli import train as train_module

    original = train_module.validate_train_args
    if getattr(original, "_chemflow_bounded_tracking_patch", False):
        return

    def corrected(args):
        metrics = args.metrics or []
        tracking = args.tracking_metric
        if tracking.startswith("bounded-") and tracking in metrics:
            args.tracking_metric = "val_loss"
            try:
                return original(args)
            finally:
                args.tracking_metric = tracking
        return original(args)

    corrected._chemflow_bounded_tracking_patch = True
    train_module.validate_train_args = corrected


def _positive_environment(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


@LossFunctionRegistry.register("huber")
class HuberLoss(ChempropMetric):
    """Huber loss with a configurable transition point."""

    def __init__(self, task_weights=1.0, **kwargs):
        super().__init__(task_weights=task_weights)
        self.delta = _positive_environment("CHEMFLOW_HUBER_DELTA", 1.0)

    def _calc_unreduced_loss(self, preds, targets, *args):
        return torch.nn.functional.huber_loss(
            preds, targets, reduction="none", delta=self.delta
        )


@LossFunctionRegistry.register("nll")
class LaplaceNLLLoss(ChempropMetric):
    """Laplace negative log likelihood with a fixed scale."""

    def __init__(self, task_weights=1.0, **kwargs):
        super().__init__(task_weights=task_weights)
        self.scale = _positive_environment("CHEMFLOW_NLL_SCALE", 1.0)

    def _calc_unreduced_loss(self, preds, targets, *args):
        scale = preds.new_tensor(self.scale)
        return (preds - targets).abs() / scale + torch.log(2.0 * scale)


@LossFunctionRegistry.register("gaussian-nll")
class GaussianNLLLoss(ChempropMetric):
    """Gaussian negative log likelihood with a fixed variance."""

    def __init__(self, task_weights=1.0, **kwargs):
        super().__init__(task_weights=task_weights)
        self.variance = _positive_environment(
            "CHEMFLOW_GAUSSIAN_NLL_VARIANCE", 1.0
        )

    def _calc_unreduced_loss(self, preds, targets, *args):
        return torch.nn.functional.gaussian_nll_loss(
            preds,
            targets,
            torch.full_like(preds, self.variance),
            full=True,
            reduction="none",
        )


class _BoundedLossMixin:
    """Apply Chemprop inequality masks before a symmetric point loss."""

    def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask):
        preds = censor_predictions_from_masks(preds, targets, lt_mask, gt_mask)
        return super()._calc_unreduced_loss(preds, targets, mask, weights)


@LossFunctionRegistry.register("bounded-huber")
class BoundedHuberLoss(_BoundedLossMixin, HuberLoss):
    pass


@LossFunctionRegistry.register("bounded-nll")
class BoundedLaplaceNLLLoss(_BoundedLossMixin, LaplaceNLLLoss):
    pass


@LossFunctionRegistry.register("bounded-gaussian-nll")
class BoundedGaussianNLLLoss(_BoundedLossMixin, GaussianNLLLoss):
    pass
