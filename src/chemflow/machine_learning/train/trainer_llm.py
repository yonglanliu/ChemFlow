# Copyright (c) Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.chemflow.machine_learning.Loss.losses import compute_total_loss
from src.chemflow.machine_learning.configs import LSTMConfig
from src.chemflow.machine_learning.data.dataset import SmilesDataset
from src.chemflow.machine_learning.eval.eval_lstm import evaluate_ce_loss
from src.chemflow.machine_learning.llm.rnn import SmilesLSTMGenerator


# =========================
# Small config helpers
# =========================
def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """Read config values from either a dict or a dataclass/object."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def cfg_require(config: Any, key: str) -> Any:
    value = cfg_get(config, key, None)
    if value is None:
        raise KeyError(f"Missing required config key: {key}")
    return value


def as_serializable(config: Any) -> Any:
    """Convert config-like objects to checkpoint-safe Python containers."""
    if isinstance(config, Mapping):
        return {k: as_serializable(v) for k, v in config.items()}
    if hasattr(config, "__dict__"):
        return {k: as_serializable(v) for k, v in vars(config).items()}
    if isinstance(config, (list, tuple)):
        return [as_serializable(v) for v in config]
    return config


# =========================
# Logging
# =========================
def setup_logging(log_dir: Path, log_name: str = "train.log") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    logger = logging.getLogger("chemflow.train_lstm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Logging to %s", log_path)
    return logger


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Config loading
# =========================
def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if suffix == ".toml":
        import tomllib

        with open(config_path, "rb") as f:
            return tomllib.load(f)

    raise ValueError(
        f"Unsupported config format: {suffix}. Use .json or .toml."
    )


# =========================
# Model config helper
# =========================
def get_lstm_config(
    vocab_size: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    lstm_config: Any,
) -> LSTMConfig:
    return LSTMConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        embedding_dim=cfg_require(lstm_config, "embedding_dim"),
        hidden_dim=cfg_require(lstm_config, "hidden_dim"),
        num_layers=cfg_require(lstm_config, "num_layers"),
        dropout=cfg_get(lstm_config, "dropout", 0.0),
        parameter_init=cfg_get(lstm_config, "parameter_init", None),
    )


# =========================
# Checkpoint helpers
# =========================
def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_loss: float,
    config: Optional[Any] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "val_loss": val_loss,
            "config": as_serializable(config),
        },
        path,
    )


def load_pretrained_for_finetune(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    device: torch.device,
    logger: logging.Logger,
) -> torch.nn.Module:
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    logger.info("Loading pretrained checkpoint from %s", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model_state = model.state_dict()
    loaded, skipped = 0, 0

    for name, weight in state_dict.items():
        if name not in model_state:
            skipped += 1
            continue

        if model_state[name].shape == weight.shape:
            model_state[name] = weight
            loaded += 1
        elif name in {"token_encoder.weight", "lm_head.weight"}:
            old_size = min(weight.size(0), model_state[name].size(0))
            model_state[name][:old_size] = weight[:old_size]
            loaded += 1
        elif name == "lm_head.bias":
            old_size = min(weight.size(0), model_state[name].size(0))
            model_state[name][:old_size] = weight[:old_size]
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_state)
    logger.info("Pretrained weights loaded: %d tensors; skipped: %d tensors", loaded, skipped)
    return model


# =========================
# Dataset loading
# =========================
def load_or_cache_dataset(training_config: Any, logger: logging.Logger) -> tuple[list[str], list[str]]:
    dataset_path = Path(cfg_require(training_config, "dataset_path")).expanduser().resolve()

    cached_data_path_value = cfg_get(training_config, "cached_data_path", None)
    if cached_data_path_value is None:
        cache_dir = Path(cfg_get(training_config, "cache_dir", "cache")).expanduser()
        cached_data_path = cache_dir / "smiles_split.pt"
    else:
        cached_data_path = Path(cached_data_path_value).expanduser().resolve()

    training_x_key = cfg_get(training_config, "training_X", "train_smiles")
    validation_x_key = cfg_get(training_config, "validation_X", "val_smiles")
    smiles_column = cfg_get(training_config, "smiles_column", "SMILES")

    if cached_data_path.exists():
        logger.info("Loading cached split from %s", cached_data_path)
        cached_data = torch.load(cached_data_path, map_location="cpu")
        train_smiles = cached_data[training_x_key]
        val_smiles = cached_data[validation_x_key]
        logger.info("Cached split size: train=%d, val=%d", len(train_smiles), len(val_smiles))
        return train_smiles, val_smiles

    logger.info("Loading dataset from %s", dataset_path)

    if dataset_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path)
    elif dataset_path.suffix.lower() == ".json":
        df = pd.read_json(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    if smiles_column not in df.columns:
        raise KeyError(
            f"SMILES column '{smiles_column}' not found. Available columns: {list(df.columns)}"
        )

    smiles = (
        df[smiles_column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .tolist()
    )

    if len(smiles) < 2:
        raise ValueError("Need at least 2 valid SMILES for train/validation split.")

    train_smiles, val_smiles = train_test_split(
        smiles,
        test_size=cfg_get(training_config, "val_train_split", 0.1),
        random_state=cfg_get(training_config, "seed", 42),
        shuffle=True,
    )

    cached_data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            training_x_key: train_smiles,
            validation_x_key: val_smiles,
        },
        cached_data_path,
    )
    logger.info("Saved cached split to %s", cached_data_path)
    logger.info("Dataset split size: train=%d, val=%d", len(train_smiles), len(val_smiles))

    return train_smiles, val_smiles


# =========================
# RL generation helpers
# =========================
def sample_next_token_with_logprob(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        log_probs = F.log_softmax(logits, dim=-1)
        next_log_prob = log_probs.gather(-1, next_token)
        return next_token, next_log_prob

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k, dim=-1)
        probs = F.softmax(values, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        next_token = indices.gather(-1, sampled_idx)
        next_log_prob = torch.log(probs.gather(-1, sampled_idx) + 1e-8)
    else:
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_log_prob = torch.log(probs.gather(-1, next_token) + 1e-8)

    return next_token, next_log_prob


def generate_for_rl(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.train()

    input_ids = prompt_ids
    batch_size = input_ids.size(0)
    device = input_ids.device

    hidden = None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    all_log_probs = []

    for _ in range(max_length):
        logits, hidden = model(input_ids[:, -1:], hidden)
        next_logits = logits[:, -1, :]

        next_token, next_log_prob = sample_next_token_with_logprob(
            next_logits,
            temperature=temperature,
            top_k=top_k,
        )

        next_token[finished] = pad_token_id
        next_log_prob[finished] = 0.0

        input_ids = torch.cat([input_ids, next_token], dim=1)
        all_log_probs.append(next_log_prob)
        finished |= next_token.squeeze(-1).eq(eos_token_id)

        if finished.all():
            break

    if not all_log_probs:
        sequence_log_probs = torch.zeros(batch_size, device=device)
    else:
        token_log_probs = torch.cat(all_log_probs, dim=1)
        sequence_log_probs = token_log_probs.sum(dim=1)

    return input_ids, sequence_log_probs


# =========================
# Train steps
# =========================
def train_step_ce_only(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_value: Optional[float] = 1.0,
) -> Dict[str, float]:
    model.train()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(input_ids)

    loss, metrics = compute_total_loss(logits=logits, labels=labels)
    loss.backward()

    if gradient_clip_value is not None and gradient_clip_value > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

    optimizer.step()
    return {k: float(v) for k, v in metrics.items()}


def train_step_with_reward(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    tokenizer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    rl_weight: float = 0.01,
    max_generation_length: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    gradient_clip_value: Optional[float] = 1.0,
) -> Dict[str, float]:
    model.train()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    batch_size = input_ids.size(0)

    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(input_ids)

    prompt_ids = torch.full(
        (batch_size, 1),
        bos_token_id,
        dtype=torch.long,
        device=device,
    )

    generated_ids, sequence_log_probs = generate_for_rl(
        model=model,
        prompt_ids=prompt_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_length=max_generation_length,
        temperature=temperature,
        top_k=top_k,
    )

    generated_smiles = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    loss, metrics = compute_total_loss(
        logits=logits,
        labels=labels,
        generated_smiles=generated_smiles,
        sequence_log_probs=sequence_log_probs,
        rl_weight=rl_weight,
    )

    loss.backward()

    if gradient_clip_value is not None and gradient_clip_value > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

    optimizer.step()
    return {k: float(v) for k, v in metrics.items()}


# =========================
# Scheduler helper
# =========================
def get_scheduler_name(training_config: Any) -> Optional[str]:
    return cfg_get(training_config, "scheduler", cfg_get(training_config, "schedular", None))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: Any,
    total_epochs: int,
):
    name = get_scheduler_name(training_config)

    if name in [None, "none", "None"]:
        return None

    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max(1, total_epochs),
        )

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
            eta_min=0.0,
        )

    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg_get(training_config, "scheduler_gamma", 0.95),
        )

    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg_get(training_config, "plateau_factor", 0.5),
            patience=cfg_get(training_config, "plateau_patience", 3),
        )

    raise ValueError(f"Unknown scheduler type: {name}")


def step_scheduler(scheduler, scheduler_name: Optional[str], val_loss: float) -> None:
    if scheduler is None:
        return

    if scheduler_name == "plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_history(history: Dict[str, list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


# =========================
# Training loops
# =========================
def run_ce_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    training_config: Any,
    checkpoint_dir: Path,
    log_dir: Path,
    device: torch.device,
    logger: logging.Logger,
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
        "learning_rate": [],
    }

    epochs_no_reward = int(cfg_get(training_config, "epochs_no_reward", 0))
    early_stop_patience = int(cfg_get(training_config, "early_stop_patience", 10))
    gradient_clip_value = cfg_get(training_config, "gradient_clip_value", 1.0)
    scheduler_name = get_scheduler_name(training_config)

    logger.info("Starting CE training for %d epochs", epochs_no_reward)

    for epoch in range(1, epochs_no_reward + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            metrics = train_step_ce_only(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                gradient_clip_value=gradient_clip_value,
            )
            running_loss += metrics["total_loss"]

            if step == 1 or step % int(cfg_get(training_config, "log_every", 50)) == 0:
                logger.info(
                    "[CE] epoch=%d step=%d/%d total_loss=%.4f lr=%.6g",
                    epoch,
                    step,
                    len(train_loader),
                    metrics["total_loss"],
                    get_lr(optimizer),
                )

        avg_train_loss = running_loss / max(1, len(train_loader))

        val_metrics = evaluate_ce_loss(model=model, val_loader=val_loader, device=device)
        val_loss = float(val_metrics["val_loss"])
        val_ppl = float(val_metrics.get("val_perplexity", float("nan")))

        step_scheduler(scheduler=scheduler, scheduler_name=scheduler_name, val_loss=val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["valid_loss"].append(val_loss)
        history["valid_perplexity"].append(val_ppl)
        history["learning_rate"].append(get_lr(optimizer))
        save_history(history, log_dir / "ce_history.csv")

        logger.info(
            "[CE] epoch=%d/%d train_loss=%.4f val_loss=%.4f val_ppl=%.4f lr=%.6g",
            epoch,
            epochs_no_reward,
            avg_train_loss,
            val_loss,
            val_ppl,
            get_lr(optimizer),
        )

        save_checkpoint(
            path=last_ce_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            config=training_config,
        )
        logger.info("Saved last CE checkpoint to %s", last_ce_path)

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
            logger.info("Saved best CE checkpoint to %s", best_ce_path)
        else:
            patience_counter += 1
            logger.info("No CE improvement: %d/%d", patience_counter, early_stop_patience)

        if patience_counter >= early_stop_patience:
            logger.info(
                "Early stopping CE at epoch %d. Best epoch=%d, best val_loss=%.4f",
                epoch,
                best_epoch,
                best_val,
            )
            break

    return history, best_ce_path


def run_reward_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    training_config: Any,
    generation_config: Any,
    checkpoint_dir: Path,
    log_dir: Path,
    device: torch.device,
    logger: logging.Logger,
    start_epoch: int = 1,
):
    best_val = float("inf")
    patience_counter = 0

    best_rl_path = checkpoint_dir / "best_rl_model.pt"
    last_rl_path = checkpoint_dir / "last_rl_model.pt"

    history = {
        "epoch": [],
        "train_loss": [],
        "ce_loss": [],
        "rl_loss": [],
        "reward_mean": [],
        "valid_loss": [],
        "learning_rate": [],
    }

    epochs_with_reward = int(cfg_get(training_config, "epochs_with_reward", 0))
    early_stop_patience = int(cfg_get(training_config, "early_stop_patience", 10))
    gradient_clip_value = cfg_get(training_config, "gradient_clip_value", 1.0)
    scheduler_name = get_scheduler_name(training_config)

    logger.info("Starting reward/RL training for %d epochs", epochs_with_reward)

    for epoch in range(1, epochs_with_reward + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        running_total = 0.0
        running_ce = 0.0
        running_rl = 0.0
        running_reward = 0.0

        for step, batch in enumerate(train_loader, start=1):
            metrics = train_step_with_reward(
                model=model,
                batch=batch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                device=device,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                rl_weight=cfg_get(training_config, "rl_weight", 0.01),
                max_generation_length=cfg_get(generation_config, "max_generation_length", cfg_get(generation_config, "max_length", 100)),
                temperature=cfg_get(generation_config, "temperature", 0.8),
                top_k=cfg_get(generation_config, "top_k", 50),
                gradient_clip_value=gradient_clip_value,
            )

            running_total += metrics.get("total_loss", 0.0)
            running_ce += metrics.get("ce_loss", 0.0)
            running_rl += metrics.get("rl_loss", 0.0)
            running_reward += metrics.get("reward_mean", 0.0)

            if step == 1 or step % int(cfg_get(training_config, "log_every", 50)) == 0:
                logger.info(
                    "[RL] epoch=%d step=%d/%d total=%.4f ce=%.4f rl=%.4f reward=%.4f lr=%.6g",
                    epoch,
                    step,
                    len(train_loader),
                    metrics.get("total_loss", 0.0),
                    metrics.get("ce_loss", 0.0),
                    metrics.get("rl_loss", 0.0),
                    metrics.get("reward_mean", 0.0),
                    get_lr(optimizer),
                )

        avg_total = running_total / max(1, len(train_loader))
        avg_ce = running_ce / max(1, len(train_loader))
        avg_rl = running_rl / max(1, len(train_loader))
        avg_reward = running_reward / max(1, len(train_loader))

        val_metrics = evaluate_ce_loss(model=model, val_loader=val_loader, device=device)
        val_loss = float(val_metrics["val_loss"])
        global_epoch = start_epoch + epoch - 1

        step_scheduler(scheduler=scheduler, scheduler_name=scheduler_name, val_loss=val_loss)

        history["epoch"].append(global_epoch)
        history["train_loss"].append(avg_total)
        history["ce_loss"].append(avg_ce)
        history["rl_loss"].append(avg_rl)
        history["reward_mean"].append(avg_reward)
        history["valid_loss"].append(val_loss)
        history["learning_rate"].append(get_lr(optimizer))
        save_history(history, log_dir / "rl_history.csv")

        logger.info(
            "[RL] epoch=%d/%d global_epoch=%d total=%.4f ce=%.4f rl=%.4f reward=%.4f val_loss=%.4f lr=%.6g",
            epoch,
            epochs_with_reward,
            global_epoch,
            avg_total,
            avg_ce,
            avg_rl,
            avg_reward,
            val_loss,
            get_lr(optimizer),
        )

        save_checkpoint(
            path=last_rl_path,
            model=model,
            optimizer=optimizer,
            epoch=global_epoch,
            val_loss=val_loss,
            config=training_config,
        )
        logger.info("Saved last RL checkpoint to %s", last_rl_path)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            save_checkpoint(
                path=best_rl_path,
                model=model,
                optimizer=optimizer,
                epoch=global_epoch,
                val_loss=val_loss,
                config=training_config,
            )
            logger.info("Saved best RL checkpoint to %s", best_rl_path)
        else:
            patience_counter += 1
            logger.info("No RL improvement: %d/%d", patience_counter, early_stop_patience)

        if patience_counter >= early_stop_patience:
            logger.info("Early stopping RL at epoch %d", epoch)
            break

    return history, best_rl_path


# =========================
# Main train function
# =========================
def train(config_path: str | Path) -> None:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)

    training_config = config["training_config"]
    tokenizer_config = config["tokenizer_config"]
    generation_config = config.get("generation_config", {})
    lstm_config = config["model_config"]

    seed = int(cfg_get(training_config, "seed", 42))
    set_seed(seed)

    work_dir = Path(cfg_get(training_config, "workdir", ".")).expanduser().resolve()
    cache_dir = work_dir / Path(cfg_get(training_config, "cache_dir", "cache"))
    checkpoint_dir = work_dir / Path(cfg_get(training_config, "checkpoint_dir", "checkpoints"))
    log_dir = work_dir / Path(cfg_get(training_config, "log_dir", "logs"))

    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir=log_dir)
    logger.info("Config path: %s", config_path)
    logger.info("Work dir: %s", work_dir)
    logger.info("Cache dir: %s", cache_dir)
    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info("Seed: %d", seed)

    train_smiles, val_smiles = load_or_cache_dataset(training_config, logger=logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tokenizer_name = cfg_require(tokenizer_config, "tokenizer_name")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info("Loaded tokenizer: %s", tokenizer_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.cls_token or tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.sep_token or tokenizer.bos_token

    condition_tokens = cfg_get(tokenizer_config, "condition_tokens", None)
    if condition_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": condition_tokens})
        logger.info("Added %d condition/special tokens", len(condition_tokens))

    logger.info(
        "Tokenizer size=%d pad=%s bos=%s eos=%s",
        len(tokenizer),
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    )

    max_length = int(cfg_get(tokenizer_config, "max_length", 128))

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

    batch_size = int(cfg_get(training_config, "batch_size", 32))
    num_workers = int(cfg_get(training_config, "num_workers", 0))

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

    logger.info(
        "DataLoaders ready: train_batches=%d val_batches=%d batch_size=%d",
        len(train_loader),
        len(val_loader),
        batch_size,
    )

    model_cfg = get_lstm_config(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        lstm_config=lstm_config,
    )

    model = SmilesLSTMGenerator.build_model(model_cfg)

    if bool(cfg_get(training_config, "fine_tune", False)):
        pretrained_ckpt_path = cfg_require(training_config, "pretrained_ckpt_path")
        model = load_pretrained_for_finetune(
            model=model,
            ckpt_path=pretrained_ckpt_path,
            device=device,
            logger=logger,
        )

    model.to(device)
    logger.info("Model moved to %s", device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_get(training_config, "learning_rate", 1e-3)),
        weight_decay=float(cfg_get(training_config, "weight_decay", 0.0)),
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        training_config=training_config,
        total_epochs=int(cfg_get(training_config, "epochs_no_reward", 0)),
    )

    ce_history, best_ce_path = run_ce_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        device=device,
        logger=logger,
    )

    epochs_with_reward = int(cfg_get(training_config, "epochs_with_reward", 0))
    if epochs_with_reward > 0:
        logger.info("Loading best CE checkpoint for reward training: %s", best_ce_path)
        ckpt = torch.load(best_ce_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        reward_lr = float(
            cfg_get(
                training_config,
                "reward_learning_rate",
                float(cfg_get(training_config, "learning_rate", 1e-3)) * 0.1,
            )
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=reward_lr,
            weight_decay=float(cfg_get(training_config, "weight_decay", 0.0)),
        )

        scheduler = build_scheduler(
            optimizer=optimizer,
            training_config=training_config,
            total_epochs=epochs_with_reward,
        )

        _, best_rl_path = run_reward_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=training_config,
            generation_config=generation_config,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            device=device,
            logger=logger,
            start_epoch=len(ce_history["epoch"]) + 1,
        )

        logger.info("Training complete. Best RL model: %s", best_rl_path)
    else:
        logger.info("Training complete. Best CE model: %s", best_ce_path)


# =========================
# CLI
# =========================
def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError(
            "Missing config path. Usage: python train_runner.py path/to/config.json"
        )

    config_path = Path(sys.argv[1]).expanduser().resolve()
    train(config_path=config_path)


if __name__ == "__main__":
    main()
