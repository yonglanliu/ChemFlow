# Modified by ChemFlow: namespaced vendored imports; optional cuik_molmaker support.
import torch
from torch import nn


REGRESSION_LOSSES = (
    "l2",
    "mse",
    "mae",
    "huber",
    "nll",
    "gaussian_nll",
)


def get_regression_loss(args):
    """Return an unreduced point-regression or fixed-noise likelihood loss."""
    name = str(getattr(args, "regression_loss", "l2")).strip().lower()
    if name not in REGRESSION_LOSSES:
        raise ValueError(
            f"Unsupported regression loss {name!r}; expected one of "
            f"{REGRESSION_LOSSES}."
        )
    if name in {"l2", "mse"}:
        return nn.MSELoss(reduction="none")
    if name == "mae":
        return nn.L1Loss(reduction="none")
    if name == "huber":
        delta = float(getattr(args, "huber_delta", 1.0))
        if not delta > 0.0:
            raise ValueError("huber_delta must be positive.")
        return nn.HuberLoss(delta=delta, reduction="none")
    if name == "nll":
        scale = float(getattr(args, "nll_scale", 1.0))
        if not scale > 0.0:
            raise ValueError("nll_scale must be positive.")

        def laplace_nll(preds, targets):
            scale_tensor = preds.new_tensor(scale)
            return (
                torch.abs(preds - targets) / scale_tensor
                + torch.log(2.0 * scale_tensor)
            )

        return laplace_nll

    variance = float(getattr(args, "gaussian_nll_variance", 1.0))
    if not variance > 0.0:
        raise ValueError("gaussian_nll_variance must be positive.")

    def gaussian_nll(preds, targets):
        return torch.nn.functional.gaussian_nll_loss(
            preds,
            targets,
            torch.full_like(preds, variance),
            full=True,
            reduction="none",
        )

    return gaussian_nll


class MTLLoss(torch.nn.Module):
    """Args:
            losses: a list of task specific loss terms
            num_tasks: number of tasks
    """

    def __init__(self, num_tasks):
        super(MTLLoss, self).__init__()
        assert num_tasks > 1, "Number of tasks must be greater than 1"
        self.num_tasks = num_tasks
        self.log_sigma = torch.nn.Parameter(torch.zeros((num_tasks)))

    def get_precisions(self):
        return 0.5 * torch.exp(- 2.0 * self.log_sigma)

    def forward(self, loss_terms):
        assert loss_terms.numel() == self.num_tasks, f"Expected {self.num_tasks} loss terms, got {loss_terms.numel()}"

        total_loss = 0
        precisions = self.get_precisions()

        for task in range(self.num_tasks):
            total_loss += precisions[task] * loss_terms[task] + self.log_sigma[task]

        return total_loss
