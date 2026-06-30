# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from src.gpt.dataset import (
    tokenize_and_cache_dataset,
    TokenizedSmilesCacheDataset,
)
from src.gpt.model import GPT


# ============================================================
# Distributed helpers
# ============================================================

def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_dist_available_and_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_dist_available_and_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def main_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed():
    """
    Supports:
    - Single GPU
    - Single CPU/MPS
    - Multi-GPU with torchrun
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA GPUs.")

        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

        device = torch.device("cuda", local_rank)
        distributed = True

    else:
        distributed = False

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return device, distributed


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


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


def reduce_mean(value: float, device: torch.device) -> float:
    if not is_dist_available_and_initialized():
        return value

    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()

    return tensor.item()


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
    if not is_main_process():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = unwrap_model(model)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "vocab_size": getattr(model_to_save, "vocab_size", None),
            "pad_token_id": getattr(model_to_save, "pad_token_id", None),
            "bos_token_id": getattr(model_to_save, "bos_token_id", None),
            "eos_token_id": getattr(model_to_save, "eos_token_id", None),
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

    input_ids = batch["input_ids"].to(device, non_blocking=True)

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
    total_batches = 0

    progress = tqdm(
        loader,
        desc="Validation",
        disable=not is_main_process(),
    )

    for batch in progress:
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        logits = model(x)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        total_loss += loss.item()
        total_batches += 1

    local_avg_loss = total_loss / max(total_batches, 1)
    avg_loss = reduce_mean(local_avg_loss, device)

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
# Training loop
# ============================================================

def run_training(
    model,
    train_loader,
    val_loader,
    train_sampler,
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

    model_for_config = unwrap_model(model)

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=model_for_config.pad_token_id,
    )

    epochs = getattr(
        training_config,
        "num_epochs",
        getattr(training_config, "epochs", 100),
    )

    early_stopping = getattr(training_config, "early_stopping", True)

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
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        num_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}/{epochs}",
            disable=not is_main_process(),
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
            num_batches += 1

            if is_main_process():
                progress.set_postfix(loss=metrics["loss"])

        local_train_loss = running_loss / max(num_batches, 1)
        avg_train_loss = reduce_mean(local_train_loss, device)

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

        if is_main_process():
            history["epoch"].append(epoch)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_perplexity"].append(val_ppl)
            history["learning_rate"].append(lr)

            print(
                f"[GPT-DDP] epoch={epoch}/{epochs} "
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

        # Broadcast early-stopping decision from rank 0
        stop_tensor = torch.tensor(0, device=device)

        if is_main_process():
            if early_stopping and patience_counter >= early_stop_patience:
                stop_tensor.fill_(1)

        if is_dist_available_and_initialized():
            dist.broadcast(stop_tensor, src=0)

        if stop_tensor.item() == 1:
            if is_main_process():
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch={best_epoch}, "
                    f"best val_loss={best_val_loss:.4f}"
                )
            break

    return history, best_path


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


# ============================================================
# Train entry
# ============================================================

def train(config_path: str | Path):
    import tomllib

    device, distributed = setup_distributed()

    try:
        if config_path is None:
            raise ValueError("Please provide a valid config path.")

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        full_config = config

        training_config = SimpleNamespace(**config["GPTTrainingConfig"])
        gpt_config = SimpleNamespace(**config["GPTConfig"])
        tokenizer_config = SimpleNamespace(**config["TokenizerConfig"])
        dataset_config = SimpleNamespace(**config["DatasetConfig"])

        seed = getattr(training_config, "seed", 42)
        set_seed(seed + get_rank())

        workdir = Path(training_config.workdir).expanduser().resolve()
        checkpoint_dir = workdir / "checkpoints"
        cache_dir = workdir / "cache"
        tokenizer_dir = workdir / "tokenizer"

        for directory in [
            workdir,
            checkpoint_dir,
            cache_dir,
            tokenizer_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        main_print(f"Using device: {device}")
        main_print(f"Distributed: {distributed}")
        main_print(f"World size: {get_world_size()}")
        main_print(f"Rank: {get_rank()}")

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

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
            main_print(f"Added {num_added} condition tokens")

        if is_main_process():
            tokenizer.save_pretrained(tokenizer_dir)

        barrier()

        pad_token_id = tokenizer.pad_token_id
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        main_print(f"vocab_size              : {tokenizer.vocab_size}")
        main_print(f"len(tokenizer)          : {len(tokenizer)}")
        main_print(f"pad_token_id            : {pad_token_id}")
        main_print(f"bos_token_id            : {bos_token_id}")
        main_print(f"eos_token_id            : {eos_token_id}")
        main_print(f"additional_special_tokens: "f"{condition_tokens}")

        # ----------------------------------------------------
        # Cache dataset
        # Rank 0 builds cache first. Other ranks wait.
        # ----------------------------------------------------

        max_length = tokenizer_config.max_length

        if is_main_process():
            manifest = tokenize_and_cache_dataset(
                dataset_config=dataset_config,
                tokenizer=tokenizer,
                max_length=max_length,
                cache_dir=cache_dir,
            )
        else:
            manifest = None

        barrier()

        if not is_main_process():
            manifest = tokenize_and_cache_dataset(
                dataset_config=dataset_config,
                tokenizer=tokenizer,
                max_length=max_length,
                cache_dir=cache_dir,
            )

        train_dataset = TokenizedSmilesCacheDataset(manifest["train"])
        val_dataset = TokenizedSmilesCacheDataset(manifest["val"])

        main_print(
            f"Loaded {len(train_dataset):,} training samples and "
            f"{len(val_dataset):,} validation samples"
        )

        # ----------------------------------------------------
        # DataLoader with DistributedSampler
        # ----------------------------------------------------

        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                drop_last=True,
            )

            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
                drop_last=False,
            )

            train_shuffle = False
            val_shuffle = False

        else:
            train_sampler = None
            val_sampler = None
            train_shuffle = True
            val_shuffle = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
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
            shuffle=val_shuffle,
            sampler=val_sampler,
            drop_last=False,
            num_workers=getattr(training_config, "num_workers", 0),
            pin_memory=(device.type == "cuda"),
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

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

        full_config["GPTConfig"] = model.get_config()

        resolved_config = {
            **full_config,
            "ResolvedConfig": {
                "workdir": str(workdir),
                "checkpoint_dir": str(checkpoint_dir),
                "cache_dir": str(cache_dir),
                "tokenizer_dir": str(tokenizer_dir),
                "device": str(device),
                "distributed": distributed,
                "world_size": get_world_size(),
                "rank": get_rank(),
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

        if is_main_process():
            save_json(resolved_config, workdir / "out_config.json")

        if distributed:
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False,
            )

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

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        history, best_path = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=training_config,
            checkpoint_dir=checkpoint_dir,
            device=device,
            full_config=resolved_config,
        )

        if is_main_process():
            history_path = checkpoint_dir / "history.pt"
            torch.save(history, history_path)

            save_json(history, checkpoint_dir / "history.json")

            print(f"Saved training history to {history_path}")
            print(f"Best model saved to {best_path}")

            if getattr(training_config, "plot_training_history", True):
                plot_output_dir = workdir / "plots"
                plot_training_history(
                    history=history,
                    output_dir=plot_output_dir,
                )

    finally:
        cleanup_distributed()


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Missing config path. Usage:\n"
            "Single GPU:\n"
            "  python train_gpt_smiles_ddp.py path/to/config.toml\n\n"
            "Multi GPU:\n"
            "  torchrun --nproc_per_node=4 train_gpt_smiles_ddp.py path/to/config.toml"
        )

    config_path = Path(sys.argv[1]).expanduser().resolve()
    train(config_path=config_path)


if __name__ == "__main__":
    main()