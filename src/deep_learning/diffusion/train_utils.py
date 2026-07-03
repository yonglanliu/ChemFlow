import csv
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from src.deep_learning.utils import (
    unwrap_model,
    move_optimizer_state_to_device,
)

# ============================================================
# Collate utilities
# ============================================================

def get_item(item, key):
    if isinstance(item, dict):
        return item[key]
    return getattr(item, key)


def has_item(item, key):
    if isinstance(item, dict):
        return key in item
    return hasattr(item, key)


def pad_1d(x, target_nodes, pad_value=0):
    out = x.new_full((target_nodes,), pad_value)
    out[: x.size(0)] = x
    return out


def pad_2d(x, target_nodes, pad_value=0):
    out = x.new_full((target_nodes, x.size(1)), pad_value)
    out[: x.size(0), :] = x
    return out


def pad_square(x: torch.Tensor, size: int, pad_value: int = 0) -> torch.Tensor:
    if x.dim() == 2:
        out = x.new_full((size, size), pad_value)
        out[: x.size(0), : x.size(1)] = x
        return out

    if x.dim() == 3:
        out = x.new_full((size, size, x.size(2)), pad_value)
        out[: x.size(0), : x.size(1), :] = x
        return out

    raise ValueError(f"pad_square only supports 2D or 3D tensors, got {x.shape}")


def pad_spatial_pos(
    x: torch.Tensor,
    size: int,
    pad_value: int = 0,
    max_pos: int = 20,
) -> torch.Tensor:
    out = x.new_full((size, size), pad_value)
    out[: x.size(0), : x.size(1)] = x
    return out.clamp(min=0, max=max_pos)


def pad_attn_bias(x, target_nodes, pad_value=0):
    out = x.new_full((target_nodes + 1, target_nodes + 1), pad_value)
    out[: x.size(0), : x.size(1)] = x
    return out


def graphormer_collate_fn(batch):
    max_nodes = max(get_item(item, "x").size(0) for item in batch)
    batch_size = len(batch)

    collated = {}

    collated["x"] = torch.stack([
        pad_2d(get_item(item, "x"), max_nodes)
        for item in batch
    ])

    collated["node_feat"] = collated["x"]

    collated["attn_bias"] = torch.stack([
        pad_attn_bias(get_item(item, "attn_bias"), max_nodes)
        for item in batch
    ])

    collated["attn_edge_type"] = torch.stack([
        pad_square(get_item(item, "attn_edge_type"), max_nodes)
        for item in batch
    ])

    collated["spatial_pos"] = torch.stack([
        pad_spatial_pos(get_item(item, "spatial_pos"), max_nodes, max_pos=20)
        for item in batch
    ])

    collated["in_degree"] = torch.stack([
        pad_1d(get_item(item, "in_degree"), max_nodes)
        for item in batch
    ])

    collated["out_degree"] = torch.stack([
        pad_1d(get_item(item, "out_degree"), max_nodes)
        for item in batch
    ])

    edge_input_list = []
    for item in batch:
        edge_input = get_item(item, "edge_input")
        max_dist = edge_input.size(2)
        feat_dim = edge_input.size(3)

        out = edge_input.new_full(
            (max_nodes, max_nodes, max_dist, feat_dim),
            0,
        )
        out[: edge_input.size(0), : edge_input.size(1), :, :] = edge_input
        edge_input_list.append(out)

    collated["edge_input"] = torch.stack(edge_input_list)

    collated["atom_types"] = torch.stack([
        pad_1d(get_item(item, "atom_types").long(), max_nodes, pad_value=0)
        for item in batch
    ])

    collated["bond_types"] = torch.stack([
        pad_square(get_item(item, "bond_types").long(), max_nodes, pad_value=0)
        for item in batch
    ])

    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = get_item(item, "x").size(0)
        node_mask[i, :n] = True

    collated["node_mask"] = node_mask

    if has_item(batch[0], "smiles"):
        collated["smiles"] = [get_item(item, "smiles") for item in batch]

    if has_item(batch[0], "y"):
        ys = [get_item(item, "y") for item in batch]
        if all(y is not None for y in ys):
            collated["y"] = torch.stack([
                y if torch.is_tensor(y) else torch.tensor(y)
                for y in ys
            ])
        else:
            collated["y"] = None

    return collated


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

    return {
        "start_epoch": int(checkpoint.get("epoch", 0)) + 1,
        "best_val_loss": float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))),
        "best_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))),
        "patience_counter": int(checkpoint.get("patience_counter", 0)),
        "history": checkpoint.get("history", None),
        "checkpoint_path": str(checkpoint_path),
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
    best_val_loss: float,
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
            "best_val_loss": best_val_loss,
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
    val_loss: float,
    val_atom_loss: float,
    val_bond_loss: float,
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
                "val_loss",
                "val_atom_loss",
                "val_bond_loss",
                "learning_rate",
            ])

        writer.writerow([
            epoch,
            train_loss,
            train_atom_loss,
            train_bond_loss,
            val_loss,
            val_atom_loss,
            val_bond_loss,
            learning_rate,
        ])