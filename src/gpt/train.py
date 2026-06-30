# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.gpt.dataset import (
    tokenize_and_cache_dataset,
    TokenizedSmilesCacheDataset,
)
from src.gpt.model import GPT
import matplotlib.pyplot as plt
import sys

# ============================================================
# Utils
# ============================================================

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def namespace_to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}

    if isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [namespace_to_dict(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    return obj


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


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
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "vocab_size": getattr(model, "vocab_size", None),
            "pad_token_id": getattr(model, "pad_token_id", None),
            "bos_token_id": getattr(model, "bos_token_id", None),
            "eos_token_id": getattr(model, "eos_token_id", None),
            "config": config,
        },
        path,
    )


# ============================================================
# Train / Eval
# ============================================================

def train_step(
    model,
    batch,
    optimizer,
    criterion,
    device,
    gradient_clip_value: Optional[float] = 1.0,
) -> Dict[str, float]:
    model.train()

    input_ids = batch["input_ids"].to(device)

    x = input_ids[:, :-1]
    y = input_ids[:, 1:]

    optimizer.zero_grad(set_to_none=True)

    logits = model(x)

    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )

    loss.backward()

    if gradient_clip_value is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_value,
        )

    optimizer.step()

    return {"loss": loss.item()}


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0

    for batch in tqdm(loader, desc="Validation"):
        input_ids = batch["input_ids"].to(device)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        logits = model(x)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)

    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    return {
        "val_loss": avg_loss,
        "val_perplexity": ppl,
    }


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


# ============================================================
# Main training loop
# ============================================================

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    training_config: SimpleNamespace,
    checkpoint_dir: Path,
    device,
    full_config: dict,
):
    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_model.pt"

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "learning_rate": [],
    }

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=model.pad_token_id,
    )

    epochs = getattr(
        training_config,
        "num_epochs",
        getattr(training_config, "epochs", 10),
    )

    early_stop_patience = getattr(
        training_config,
        "early_stopping_patience",
        10,
    )

    gradient_clip_value = getattr(
        training_config,
        "gradient_clip_value",
        1.0,
    )

    scheduler_name = getattr(
        training_config,
        "scheduler",
        getattr(training_config, "schedular", None),
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}/{epochs}",
        )

        for batch in progress:
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                gradient_clip_value=gradient_clip_value,
            )

            running_loss += metrics["loss"]
            progress.set_postfix(loss=metrics["loss"])

        avg_train_loss = running_loss / max(len(train_loader), 1)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_loss = val_metrics["val_loss"]
        val_ppl = val_metrics["val_perplexity"]

        step_scheduler(
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            val_loss=val_loss,
        )

        lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_ppl)
        history["learning_rate"].append(lr)

        print(
            f"[GPT] epoch={epoch}/{epochs} "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_ppl={val_ppl:.4f} "
            f"lr={lr:.6g}"
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            val_perplexity=val_ppl,
            config=full_config,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                val_perplexity=val_ppl,
                config=full_config,
            )

            print(f"Saved best checkpoint to {best_path}")

        else:
            patience_counter += 1
            print(
                f"No improvement: "
                f"{patience_counter}/{early_stop_patience}"
            )

        if patience_counter >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch={best_epoch}, "
                f"best val_loss={best_val_loss:.4f}"
            )
            break

    return history, best_path


# ============================================================
# Plotting helpers
# ============================================================
def plot_training_history(
    history: dict,
    output_dir: str | Path,
):
    """
    Plot training history.

    Parameters
    ----------
    history : dict
        {
            "epoch": [...],
            "train_loss": [...],
            "val_loss": [...],
            "val_perplexity": [...],
            "learning_rate": [...],
        }
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]

    # -------------------------
    # Loss
    # -------------------------
    plt.figure(figsize=(7, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train",
        linewidth=2,
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------
    # Perplexity
    # -------------------------
    if "val_perplexity" in history:

        plt.figure(figsize=(7, 5))

        plt.plot(
            epochs,
            history["val_perplexity"],
            linewidth=2,
        )

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

    # -------------------------
    # Learning rate
    # -------------------------
    if "learning_rate" in history:

        plt.figure(figsize=(7, 5))

        plt.plot(
            epochs,
            history["learning_rate"],
            linewidth=2,
        )

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

# ============================================================
# Entry point
# ============================================================

def train(config_path: str | Path):

    import tomllib
    if config_path is None:
        raise ValueError("Please provide a valid config path.")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    if isinstance(config, dict):
        full_config = config

        training_config = SimpleNamespace(**config["GPTTrainingConfig"])
        gpt_config = SimpleNamespace(**config["GPTConfig"])
        tokenizer_config = SimpleNamespace(**config["TokenizerConfig"])
        dataset_config = SimpleNamespace(**config["DatasetConfig"])

    else:
        training_config = config.GPTTrainingConfig
        gpt_config = config.GPTConfig
        tokenizer_config = config.TokenizerConfig
        dataset_config = config.DatasetConfig

        full_config = namespace_to_dict(config)

    seed = getattr(training_config, "seed", 42)
    set_seed(seed)

    workdir = Path(training_config.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = workdir / "checkpoints"
    cache_dir = workdir / "cache"
    tokenizer_dir = workdir / "tokenizer"

    for directory in [
        checkpoint_dir,
        cache_dir,
        tokenizer_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config.tokenizer_name,
    )

    condition_tokens = getattr(
        tokenizer_config,
        "condition_tokens",
        None,
    )

    if condition_tokens is not None and len(condition_tokens) > 0:
        num_added = tokenizer.add_special_tokens(
            {
                "additional_special_tokens": condition_tokens,
            }
        )
        print(f"Added {num_added} condition tokens")

    tokenizer.save_pretrained(tokenizer_dir)

    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    print(f"vocab_size              : {tokenizer.vocab_size}")
    print(f"len(tokenizer)          : {len(tokenizer)}")
    print(f"pad_token_id            : {pad_token_id}")
    print(f"bos_token_id            : {bos_token_id}")
    print(f"eos_token_id            : {eos_token_id}")
    print(f"additional_special_tokens: {condition_tokens}")

    # --------------------------------------------------------
    # Save resolved config
    # --------------------------------------------------------

    resolved_config = {
        **full_config,
        "ResolvedConfig": {
            "workdir": str(workdir),
            "checkpoint_dir": str(checkpoint_dir),
            "cache_dir": str(cache_dir),
            "tokenizer_dir": str(tokenizer_dir),
            "device": str(device),
            "seed": seed,
            "vocab_size_original": tokenizer.vocab_size,
            "vocab_size_with_added_tokens": len(tokenizer),
            "pad_token": tokenizer.pad_token,
            "pad_token_id": pad_token_id,
            "bos_token": tokenizer.bos_token,
            "bos_token_id": bos_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": eos_token_id,
            "unk_token": tokenizer.unk_token,
            "unk_token_id": tokenizer.unk_token_id,
            "condition_tokens": condition_tokens,
        },
    }

    save_json(resolved_config, workdir / "out_config.json")

    # --------------------------------------------------------
    # Cache + Dataset
    # --------------------------------------------------------

    max_length = tokenizer_config.max_length

    manifest = tokenize_and_cache_dataset(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        max_length=max_length,
        cache_dir=cache_dir,
    )

    train_dataset = TokenizedSmilesCacheDataset(manifest["train"])
    val_dataset = TokenizedSmilesCacheDataset(manifest["val"])

    print(
        f"Loaded {len(train_dataset):,} training samples and "
        f"{len(val_dataset):,} validation samples"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=getattr(training_config, "num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=getattr(
            training_config,
            "eval_batch_size",
            training_config.batch_size,
        ),
        shuffle=False,
        drop_last=False,
        num_workers=getattr(training_config, "num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GPT(
        vocab_size=len(tokenizer),
        max_len=max_length,
        d_model=gpt_config.d_model,
        n_heads=gpt_config.n_heads,
        n_layers=gpt_config.n_layers,
        d_ff=gpt_config.d_ff,
        dropout=gpt_config.dropout,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        use_quant_noise=getattr(gpt_config, "use_quant_noise", False),
        quant_noise_p=getattr(gpt_config, "quant_noise_p", 0.0),
        quant_noise_block_size=getattr(
            gpt_config,
            "quant_noise_block_size",
            8,
        ),
    ).to(device)

    full_config["GPTConfig"]= model.get_config()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    epochs = getattr(
        training_config,
        "num_epochs",
        getattr(training_config, "epochs", 100),
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        training_config=training_config,
        total_epochs=epochs,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history, best_path = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        device=device,
        full_config=resolved_config,
    )

    history_path = checkpoint_dir / "history.pt"

    torch.save(history, history_path)

    print(f"Saved training history to {history_path}")
    print(f"Best model saved to {best_path}")

    if getattr(training_config, "plot_training_history", True):
        plot_output_dir = workdir/ "plots"
        os.makedirs(plot_output_dir, exist_ok=True)
        plot_training_history(
            history=history,
            output_dir=plot_output_dir,
        )
    # return {
    #     "history": history,
    #     "best_path": best_path,
    #     "workdir": workdir,
    #     "config_path": workdir / "out_config.json",
    # }

def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Missing config path. Usage: python trainer_llm_ce_only.py path/to/config.toml"
        )

    config_path = Path(sys.argv[1]).expanduser().resolve()
    train(config_path=str(config_path))

if __name__ == "__main__":
    main()