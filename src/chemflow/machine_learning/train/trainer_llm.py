# trainer_llm_ce_only.py

from __future__ import annotations

import os
import sys
import random
import tomllib
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.chemflow.machine_learning.llm.rnn import SmilesLSTMGenerator
from src.chemflow.machine_learning.data.dataset import SmilesDataset
from src.chemflow.machine_learning.configs import LSTMConfig
from src.chemflow.machine_learning.eval.eval_lstm import evaluate_ce_loss


# =========================
# Reproducibility
# =========================

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Config helper
# =========================

def get_lstm_config(
    vocab_size: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    lstm_config: dict,
):
    return LSTMConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        embedding_dim=lstm_config["embedding_dim"],
        hidden_dim=lstm_config["hidden_dim"],
        num_layers=lstm_config["num_layers"],
        dropout=lstm_config["dropout"],
        parameter_init=lstm_config.get("parameter_init", "xavier"),
    )


# =========================
# Checkpoint helpers
# =========================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    val_loss: float,
    config=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def load_pretrained_for_finetune(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model_state = model.state_dict()

    for name, weight in state_dict.items():
        if name not in model_state:
            continue

        if model_state[name].shape == weight.shape:
            model_state[name] = weight

        elif name in ["token_encoder.weight", "lm_head.weight"]:
            old_size = weight.size(0)
            model_state[name][:old_size] = weight

        elif name == "lm_head.bias":
            old_size = weight.size(0)
            model_state[name][:old_size] = weight

    model.load_state_dict(model_state)
    return model


# =========================
# Dataset loading
# =========================

def load_or_cache_dataset(training_config: dict):
    dataset_path = Path(training_config["dataset_path"]).expanduser().resolve()
    cached_data_path = Path(training_config["cached_data_path"]).expanduser().resolve()

    training_X = training_config.get("training_X", "train_smiles")
    validation_X = training_config.get("validation_X", "val_smiles")

    if cached_data_path.exists():
        print(f"Loading cached data from {cached_data_path}")
        cached_data = torch.load(cached_data_path)
        return cached_data[training_X], cached_data[validation_X]

    print(f"Loading dataset from {dataset_path}")

    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        df = pd.read_csv(dataset_path)
    elif dataset_path.suffix == ".json":
        df = pd.read_json(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    smiles_column = training_config.get("smiles_column", "SMILES")
    smiles = df[smiles_column].dropna().astype(str).str.strip().tolist()
    smiles = sorted(set(smiles))

    train_smiles, val_smiles = train_test_split(
        smiles,
        test_size=training_config["val_train_split"],
        random_state=training_config.get("seed", 42),
        shuffle=True,
    )

    cached_data_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            training_X: train_smiles,
            validation_X: val_smiles,
        },
        cached_data_path,
    )

    print(f"Cached split saved to {cached_data_path}")

    return train_smiles, val_smiles


# =========================
# Train step
# =========================

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
    labels = batch["labels"].to(device)

    optimizer.zero_grad(set_to_none=True)

    logits, _ = model(input_ids)

    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )

    loss.backward()

    if gradient_clip_value is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_value,
        )

    optimizer.step()

    return {"total_loss": loss.item()}


# =========================
# Scheduler helper
# =========================

def build_scheduler(optimizer, training_config: dict, total_epochs: int):
    name = training_config.get("schedular", training_config.get("scheduler", None))

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
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95,
        )

    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=training_config.get("plateau_patience", 5),
        )

    if name in [None, "none"]:
        return None

    raise ValueError(f"Unknown scheduler type: {name}")


def step_scheduler(scheduler, scheduler_name: str | None, val_loss: float):
    if scheduler is None:
        return

    if scheduler_name == "plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


# =========================
# CE training loop
# =========================

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    training_config: dict,
    checkpoint_dir: Path,
    device,
):
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_ce_path = checkpoint_dir / "best_ce_model.pt"
    last_ce_path = checkpoint_dir / "last_ce_model.pt"

    history = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_perplexity": [],
    }

    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.pad_token_id)

    epochs = training_config["epochs"]
    early_stop_patience = training_config.get("early_stop_patience", 20)
    gradient_clip_value = training_config.get("gradient_clip_value", 1.0)
    scheduler_name = training_config.get("schedular", training_config.get("scheduler", None))

    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                gradient_clip_value=gradient_clip_value,
            )

            running_loss += metrics["total_loss"]

            if step == 1 or step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[CE] epoch={epoch} step={step}/{len(train_loader)} "
                    f"total_loss={metrics['total_loss']:.4f} lr={lr:.6f}"
                )

        avg_train_loss = running_loss / len(train_loader)

        val_metrics = evaluate_ce_loss(
            model=model,
            val_loader=val_loader,
            device=device,
        )

        val_loss = val_metrics["val_loss"]
        val_ppl = val_metrics.get("val_perplexity", float("nan"))

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["valid_loss"].append(val_loss)
        history["valid_perplexity"].append(val_ppl)

        step_scheduler(
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            val_loss=val_loss,
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[CE] epoch={epoch}/{epochs} "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_ppl={val_ppl:.4f} "
            f"lr={lr:.6f}"
        )

        save_checkpoint(
            path=last_ce_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            config=training_config,
        )

        print(f"Saved last CE checkpoint to {last_ce_path}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                path=best_ce_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                config=training_config,
            )

            print(f"Saved best CE checkpoint to {best_ce_path}")

        else:
            patience_counter += 1
            print(f"No CE improvement: {patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            print(
                f"Early stopping CE at epoch {epoch}. "
                f"Best epoch={best_epoch}, best val_loss={best_val:.4f}"
            )
            break

    return history, best_ce_path


# =========================
# Main training
# =========================

def train(config_path: Optional[str] = None):
    if config_path is None:
        raise ValueError("Please provide a valid config path.")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    training_config = config["training_config"]
    tokenizer_config = config["tokenizer_config"]
    lstm_config = config["model_config"]

    seed = training_config.get("seed", 42)
    set_seed(seed)

    work_dir = Path(training_config["workdir"]).expanduser().resolve()
    cache_dir = work_dir / training_config["cache_dir"]
    checkpoint_dir = work_dir / training_config["checkpoint_dir"]

    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_smiles, val_smiles = load_or_cache_dataset(training_config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["tokenizer_name"])

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": "<BOS>"})

    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<EOS>"})

    condition_tokens = tokenizer_config.get("condition_tokens", None)

    if condition_tokens is not None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": condition_tokens}
        )

    tokenizer_save_dir = work_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_save_dir)
    print(f"Tokenizer saved to {tokenizer_save_dir}")

    max_length = tokenizer_config["max_length"]

    train_dataset = SmilesDataset(
        smiles_list=train_smiles,
        tokenizer=tokenizer,
        max_length=max_length,
        condition_list=None,
        ignore_condition_loss=False,
    )

    val_dataset = SmilesDataset(
        smiles_list=val_smiles,
        tokenizer=tokenizer,
        max_length=max_length,
        condition_list=None,
        ignore_condition_loss=False,
    )

    batch_size = training_config["batch_size"]
    num_workers = training_config.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cfg = get_lstm_config(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        lstm_config=lstm_config,
    )

    model = SmilesLSTMGenerator.build_model(cfg)

    if training_config.get("fine_tune", False):
        pretrained_ckpt_path = training_config["pretrained_ckpt_path"]

        print(f"Fine-tuning from {pretrained_ckpt_path}")

        model = load_pretrained_for_finetune(
            model=model,
            ckpt_path=pretrained_ckpt_path,
            device=device,
        )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        training_config=training_config,
        total_epochs=training_config["epochs"],
    )

    ce_history, best_ce_path = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    history_path = checkpoint_dir / "ce_history.pt"
    torch.save(ce_history, history_path)

    print(f"Best CE model saved at: {best_ce_path}")
    print(f"CE history saved at: {history_path}")


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Missing config path. Usage: python trainer_llm_ce_only.py path/to/config.toml"
        )

    config_path = Path(sys.argv[1]).expanduser().resolve()
    train(config_path=str(config_path))


if __name__ == "__main__":
    main()