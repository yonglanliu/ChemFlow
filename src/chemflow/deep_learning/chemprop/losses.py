"""ChemFlow regression losses registered with Chemprop's native loss registry."""

from __future__ import annotations

import os
import math

import torch
from chemprop.nn.metrics import ChempropMetric, LossFunctionRegistry


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
