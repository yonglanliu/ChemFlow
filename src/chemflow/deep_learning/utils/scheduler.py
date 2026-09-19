import torch
from types import SimpleNamespace
from typing import Optional

# ============================================================
# Scheduler
# ============================================================
def build_scheduler(
    optimizer,
    training_config: SimpleNamespace,
    total_epochs: int,
):
    name = getattr(
        training_config,
        "scheduler",
        getattr(training_config, "schedular", None),
    )

    if name in [None, "none"]:
        return None

    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_epochs,
        )

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=0.0,
        )

    if name == "exponential":
        gamma = getattr(training_config, "scheduler_gamma", 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
        )

    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=getattr(training_config, "plateau_factor", 0.5),
            patience=getattr(training_config, "plateau_patience", 5),
        )

    raise ValueError(f"Unknown scheduler type: {name}")


def step_scheduler(
    scheduler,
    scheduler_name: Optional[str],
    val_loss: float,
) -> None:
    if scheduler is None:
        return

    if scheduler_name == "plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()