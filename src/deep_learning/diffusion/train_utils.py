import csv
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from src.deep_learning.utils import (
    unwrap_model,
    move_optimizer_state_to_device,
)
from src.deep_learning.utils.distributed import main_print
from src.deep_learning.graphormer.utils.load_pretrained_model import load_graphormer_backbone

def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


# ============================================================
# Plot
# ============================================================

def plot_training_history(history: dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not history["epoch"]:
        print("No history to plot.")
        return

    epochs = history["epoch"]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_atom_loss"], label="Train Atom", linewidth=2)
    plt.plot(epochs, history["val_atom_loss"], label="Val Atom", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Atom Loss")
    plt.title("Training and Validation Atom Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "atom_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_bond_loss"], label="Train Bond", linewidth=2)
    plt.plot(epochs, history["val_bond_loss"], label="Val Bond", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Bond Loss")
    plt.title("Training and Validation Bond Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bond_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_bond_acc"], label="Train Bond Acc", linewidth=2)
    plt.plot(epochs, history["val_bond_acc"], label="Val Bond Acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Bond Accuracy")
    plt.title("Training and Validation Bond Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bond_acc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["real_bond_acc"], label="Train Real Bond Acc", linewidth=2)
    plt.plot(epochs, history["val_real_bond_acc"], label="Val Real Bond Acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Real Bond Accuracy")
    plt.title("Training and Validation Real Bond Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "real_bond_acc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["learning_rate"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved training curves to {output_dir}")


def load_checkpoint_for_resume(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
    load_optimizer=True,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if load_optimizer and scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "start_epoch": checkpoint["epoch"] + 1,
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "best_epoch": checkpoint.get("best_epoch", 0),
        "patience_counter": checkpoint.get("patience_counter", 0),
        "history": checkpoint.get("history", None),
        "checkpoint_path": checkpoint_path,
    }

# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    train_atom_loss: float,
    train_bond_loss: float,
    val_loss: float,
    val_atom_loss: float,
    val_bond_loss: float,
    #best_val_loss: float,
    best_pred_no_bond_ratio: float,
    best_epoch: int,
    patience_counter: int,
    history: dict,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = unwrap_model(model)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "train_loss": train_loss,
            "train_atom_loss": train_atom_loss,
            "train_bond_loss": train_bond_loss,
            "val_loss": val_loss,
            "val_atom_loss": val_atom_loss,
            "val_bond_loss": val_bond_loss,
            #"best_val_loss": best_val_loss,
            "best_pred_no_bond_ratio": best_pred_no_bond_ratio,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "history": history,
            "config": config,
        },
        path,
    )



def append_history_csv(
    path: str | Path,
    epoch: int,
    train_loss: float,
    train_atom_loss: float,
    train_bond_loss: float,
    train_bond_acc: float,
    train_real_bond_acc: float,
    train_real_bond_recall: float,
    train_real_bond_type_acc_when_pred_real: float,
    train_no_bond_ratio: float,
    train_pred_no_bond_ratio: float,
    val_loss: float,
    val_atom_loss: float,
    val_bond_loss: float,
    val_bond_acc: float,
    val_real_bond_acc: float,
    val_real_bond_recall: float,
    val_real_bond_type_acc_when_pred_real: float,
    val_no_bond_ratio: float,
    val_pred_no_bond_ratio: float,
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
                "train_atom_loss",
                "train_bond_loss",
                "train_bond_acc",
                "train_real_bond_acc",
                "train_real_bond_recall",
                "train_real_bond_type_acc_when_pred_real",
                "train_no_bond_ratio",
                "train_pred_no_bond_ratio",
                "val_loss",
                "val_atom_loss",
                "val_bond_loss",
                "val_bond_acc",
                "val_real_bond_acc",
                "val_real_bond_recall",
                "val_real_bond_type_acc_when_pred_real",
                "val_no_bond_ratio",
                "val_pred_no_bond_ratio",
                "learning_rate",
            ])

        writer.writerow([
            epoch,
            train_loss,
            train_atom_loss,
            train_bond_loss,
            train_bond_acc,
            train_real_bond_acc,
            train_real_bond_recall,
            train_real_bond_type_acc_when_pred_real,
            train_no_bond_ratio,
            train_pred_no_bond_ratio,
            val_loss,
            val_atom_loss,
            val_bond_loss,
            val_bond_acc,
            val_real_bond_acc,
            val_real_bond_recall,
            val_real_bond_type_acc_when_pred_real,
            val_no_bond_ratio,
            val_pred_no_bond_ratio,
            learning_rate,
        ])

