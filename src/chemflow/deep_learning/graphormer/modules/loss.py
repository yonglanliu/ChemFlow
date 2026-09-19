import torch
import torch.nn as nn


class MultiTaskGaussianNLLLoss(nn.Module):
    """
    Gaussian NLL with one learnable log-variance per task.

    For task i:

        loss_i =
            0.5 * exp(-log_var_i) * (y - mu)^2
            + 0.5 * log_var_i
    """

    def __init__(
        self,
        num_tasks: int,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.num_tasks = num_tasks
        self.reduction = reduction

        self.log_vars = nn.Parameter(
            torch.zeros(num_tasks)
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        task_index: int,
    ) -> torch.Tensor:

        if not isinstance(task_index, int):
            raise TypeError(
                f"task_index must be int, "
                f"got {type(task_index).__name__}."
            )

        if task_index < 0 or task_index >= self.num_tasks:
            raise IndexError(
                f"task_index must be in [0, {self.num_tasks}), "
                f"got {task_index}."
            )

        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target must have the same shape, "
                f"got {tuple(prediction.shape)} and "
                f"{tuple(target.shape)}."
            )

        log_var = self.log_vars[task_index]

        loss = (
            0.5
            * torch.exp(-log_var)
            * (target - prediction).pow(2)
            + 0.5 * log_var
        )

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "none":
            return loss

        raise ValueError(
            f"Unsupported reduction: {self.reduction}"
        )

class MultiTaskLaplaceNLLLoss(nn.Module):
    """
    Laplace NLL with one learnable log-scale per task.

    For task i:

        loss_i =
            exp(-log_scale_i) * |y - mu|
            + log_scale_i
    """

    def __init__(
        self,
        num_tasks: int,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.num_tasks = num_tasks
        self.reduction = reduction

        self.log_scales = nn.Parameter(
            torch.zeros(num_tasks)
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        task_index: int,
    ) -> torch.Tensor:

        if not isinstance(task_index, int):
            raise TypeError(
                f"task_index must be int, "
                f"got {type(task_index).__name__}."
            )

        if task_index < 0 or task_index >= self.num_tasks:
            raise IndexError(
                f"task_index must be in [0, {self.num_tasks}), "
                f"got {task_index}."
            )

        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target must have the same shape, "
                f"got {tuple(prediction.shape)} and "
                f"{tuple(target.shape)}."
            )

        log_scale = self.log_scales[task_index]

        loss = (
            torch.exp(-log_scale)
            * torch.abs(target - prediction)
            + log_scale
        )

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "none":
            return loss

        raise ValueError(
            f"Unsupported reduction: {self.reduction}"
        )