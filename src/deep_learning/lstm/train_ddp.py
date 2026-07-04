# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional


import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer


from src.deep_learning.lstm.dataset import (
    TokenizedSmilesCacheDataset,
    tokenize_and_cache_dataset,
)
from src.deep_learning.lstm.model import SmilesLSTM
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

from src.deep_learning.lstm.train_utils import (
    move_optimizer_state_to_device,
    reduce_mean,
    save_json,
    get_resume_path,
    set_seed,
    load_checkpoint_for_resume,
    save_checkpoint,
    append_history_csv,
    build_scheduler,
    step_scheduler,
    plot_training_history,
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

    logits, _= model(x)

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
        disable=disable_tqdm(),
    )

    for batch in progress:
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        logits, _ = model(x)

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
# Training loop
# ============================================================

def run_training(
    model,
    train_loader,
    val_loader,
    train_sampler,
    optimizer,
    scheduler,
    training_config,
    checkpoint_dir,
    device,
    full_config,
    start_epoch: int = 1,
    best_val_loss: float = float("inf"),
    best_val_perplexity: Optional[float] = None,
    best_epoch: int = 0,
    patience_counter: int = 0,
    history: Optional[dict] = None,
):
    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_model.pt"

    if history is None:
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

    if start_epoch > epochs:
        main_print(
            f"Checkpoint already reached epoch {start_epoch - 1}; "
            f"num_epochs={epochs}. Nothing to train."
        )
        return history, best_path

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        num_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}/{epochs}",
            disable=disable_tqdm(),
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

            if is_main_process() and not progress.disable:
                progress.set_postfix(loss=f"{metrics['loss']:.4f}")

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
            
            append_history_csv(
                checkpoint_dir / "history.csv",
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                val_perplexity=val_ppl,
                learning_rate=lr,
            )

            print(
                f"[LSTM-DDP] epoch={epoch}/{epochs} "
                f"train_loss={avg_train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_ppl={val_ppl:.4f} "
                f"lr={lr:.6g}",
                flush=True,
            )

            improved = val_loss < best_val_loss

            if improved:
                best_val_loss = val_loss
                best_val_perplexity = val_ppl
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                val_perplexity=val_ppl,
                best_val_loss=best_val_loss,
                best_val_perplexity=best_val_perplexity,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
                scheduler=scheduler,
                history=history,
                config=full_config,
            )

            if improved:
                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                    val_perplexity=val_ppl,
                    best_val_loss=best_val_loss,
                    best_val_perplexity=best_val_perplexity,
                    best_epoch=best_epoch,
                    patience_counter=patience_counter,
                    scheduler=scheduler,
                    history=history,
                    config=full_config,
                )

                print(f"Saved best checkpoint to {best_path}", flush=True)
            else:
                print(
                    f"No improvement: {patience_counter}/{early_stop_patience}",
                    flush=True,
                )

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
                    f"best val_loss={best_val_loss:.4f}",
                    flush=True,
                )
            break

    return history, best_path



# ============================================================
# Train entry
# ============================================================

def train(config_path: str | Path):
    import tomllib

    # Setup distributed training
    device, distributed = setup_distributed() 

    try:
        # Load configuration
        if config_path is None:
            raise ValueError("Please provide a valid config path.")

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        full_config = config

        training_config = SimpleNamespace(**config["LSTMTrainingConfig"])
        lstm_config = SimpleNamespace(**config["LSTMConfig"])
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

        barrier()  # Wait for rank 0 to save the tokenizer before other ranks load it

        pad_token_id = tokenizer.pad_token_id
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        main_print(f"vocab_size              : {tokenizer.vocab_size}")
        main_print(f"len(tokenizer)          : {len(tokenizer)}")
        main_print(f"pad_token_id            : {pad_token_id}")
        main_print(f"bos_token_id            : {bos_token_id}")
        main_print(f"eos_token_id            : {eos_token_id}")
        main_print(f"additional_special_tokens: {condition_tokens}")

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

        barrier()  # Wait for rank 0 to build the cache before other ranks load it

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

        model = SmilesLSTM(
            vocab_size=len(tokenizer),
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            embedding_dim=lstm_config.embedding_dim,
            hidden_dim=lstm_config.hidden_dim,
            num_layers=lstm_config.num_layers,
            dropout=lstm_config.dropout,
        ).to(device)


        full_config["LSTMConfig"] = model.get_config()

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
        # Resume
        # ----------------------------------------------------

        start_epoch = 1
        best_val_loss = float("inf")
        best_val_perplexity = None
        best_epoch = 0
        patience_counter = 0
        resume_history = None

        resume_path = get_resume_path(training_config, checkpoint_dir)

        if resume_path is not None:
            resume_state = load_checkpoint_for_resume(
                checkpoint_path=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

            start_epoch = resume_state["start_epoch"]
            best_val_loss = resume_state["best_val_loss"]
            best_val_perplexity = resume_state["best_val_perplexity"]
            best_epoch = resume_state["best_epoch"]
            patience_counter = resume_state["patience_counter"]
            resume_history = resume_state["history"]

            main_print(f"Resumed from: {resume_state['checkpoint_path']}")
            main_print(f"Starting from epoch: {start_epoch}")
            main_print(f"Best val loss so far: {best_val_loss:.6f}")
            main_print(f"Best epoch so far: {best_epoch}")

        barrier()

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
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            best_val_perplexity=best_val_perplexity,
            best_epoch=best_epoch,
            patience_counter=patience_counter,
            history=resume_history,
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
