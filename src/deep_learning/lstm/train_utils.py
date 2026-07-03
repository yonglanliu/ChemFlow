import os
import random
import sys
import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from src.deep_learning.utils import (
    is_dist_available_and_initialized,
    get_world_size,
    set_seed,
    namespace_to_dict,
    save_json,
    build_scheduler,
    step_scheduler,
)

from src.deep_learning.utils import (
    is_dist_available_and_initialized,
    get_rank,
    get_world_size,
    is_main_process,
    main_print,
    disable_tqdm,
    setup_distributed,
    cleanup_distributed,
    unwrap_model,
    barrier,
)
import csv

# ============================================================
# Utils
# ============================================================
def reduce_mean(value: float, device: torch.device) -> float:
    if not is_dist_available_and_initialized():
        return value

    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()

    return tensor.item()


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    """Needed when resuming optimizer states onto GPU/MPS."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def get_resume_path(training_config: SimpleNamespace, checkpoint_dir: Path) -> Optional[Path]:
    """
    Resume options in TOML:

    resume = true                         -> resume from checkpoints/last_model.pt
    resume_from = "/path/to/last_model.pt" -> resume from explicit checkpoint
    """
    resume_from = getattr(training_config, "resume_from", None)
    resume = bool(getattr(training_config, "resume", False))

    if resume_from is not None and str(resume_from).strip().lower() not in ["", "none", "false"]:
        return Path(resume_from).expanduser().resolve()

    if resume:
        return checkpoint_dir / "last_model.pt"

    return None

# ============================================================
# Checkpoint
# ============================================================
def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_perplexity: float,
    best_val_loss: Optional[float],
    best_val_perplexity: Optional[float],
    best_epoch: int,
    patience_counter: int,
    scheduler=None,
    history: Optional[dict] = None,
    config: Optional[dict] = None,
) -> None:
    if not is_main_process():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = unwrap_model(model)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "best_val_loss": best_val_loss,
            "best_val_perplexity": best_val_perplexity,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "history": history,
            "vocab_size": getattr(model_to_save, "vocab_size", None),
            "pad_token_id": getattr(model_to_save, "pad_token_id", None),
            "bos_token_id": getattr(model_to_save, "bos_token_id", None),
            "eos_token_id": getattr(model_to_save, "eos_token_id", None),
            "config": config,
        },
        path,
    )

def load_checkpoint_for_resume(
    checkpoint_path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    device: torch.device | str = "cpu",
):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, torch.device(device))

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_val_perplexity = checkpoint.get("best_val_perplexity", None)
    best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
    patience_counter = int(checkpoint.get("patience_counter", 0))
    history = checkpoint.get("history", None)

    return {
        "start_epoch": start_epoch,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": best_val_perplexity,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }

def append_history_csv(
    path: str | Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_perplexity: float,
    learning_rate: float,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_perplexity",
                "learning_rate",
            ])

        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            val_perplexity,
            learning_rate,
        ])


# ============================================================
# Plotting
# ============================================================

def plot_training_history(
    history: dict,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    if "val_perplexity" in history:
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history["val_perplexity"], linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.title("Validation Perplexity")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "perplexity_curve.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    if "learning_rate" in history:
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history["learning_rate"], linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "learning_rate.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"Saved training curves to {output_dir}")