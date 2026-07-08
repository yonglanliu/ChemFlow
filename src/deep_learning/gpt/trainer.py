# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from src.deep_learning.gpt.dataset import (
    TokenizedSmilesCacheDataset,
    tokenize_and_cache_dataset,
)
from src.deep_learning.gpt.model import GPT
from src.deep_learning.gpt.train_utils import (
    move_optimizer_state_to_device,
    reduce_mean,
    save_json,
    get_resume_path,
    set_seed,
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
    step_scheduler,
    build_scheduler,
)
from src.deep_learning.fine_tune.lora import LoRALinear

# ============================================================
# LoRA modules
# ============================================================
def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def add_lora_to_attention_layers(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    use_k_proj: bool = False,
) -> nn.Module:
    for block in model.blocks:
        block.attention_layer.q_proj = LoRALinear(
            block.attention_layer.q_proj,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

        block.attention_layer.v_proj = LoRALinear(
            block.attention_layer.v_proj,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

        if use_k_proj:
            block.attention_layer.k_proj = LoRALinear(
                block.attention_layer.k_proj,
                r=r,
                alpha=alpha,
                dropout=dropout,
            )

    return model


def add_lora_to_ffn_layers(
    model: nn.Module,
    r: int = 4,
    alpha: int = 16,
    dropout: float = 0.05,
    use_fc2: bool = False,
) -> nn.Module:
    for block in model.blocks:
        block.ffn[0] = LoRALinear(
            block.ffn[0],
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

        if use_fc2:
            block.ffn[2] = LoRALinear(
                block.ffn[2],
                r=r,
                alpha=alpha,
                dropout=dropout,
            )

    return model


def print_trainable_parameters(model: nn.Module) -> None:
    total = 0
    trainable = 0

    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    ratio = 100 * trainable / total if total > 0 else 0

    main_print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({ratio:.4f}%)"
    )

    main_print("Trainable parameter names:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            main_print(f"  {name}: {tuple(p.shape)}")


# ============================================================
# Trainer
# ============================================================

class GPTDDPTrainer:
    def __init__(
        self,
        model=GPT,
        config_path: str | Path | None = None,
    ):
        if config_path is None:
            raise ValueError("Config file is not specified.")

        config_path = Path(config_path)

        if not config_path.exists():
            raise ValueError(f"Config file {config_path} does not exist.")

        self.model = model
        self.config_path = config_path

    def train(self):
        device, distributed = setup_distributed()

        try:
            (
                full_config,
                training_config,
                gpt_config,
                tokenizer_config,
                dataset_config,
            ) = self.load_configs()

            seed = getattr(training_config, "seed", 42)
            set_seed(seed + get_rank())

            workdir = Path(training_config.workdir)
            checkpoint_dir = workdir / "checkpoints"
            cache_dir = workdir / "cache"
            tokenizer_dir = cache_dir / "tokenizer"

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

            main_print(f"vocab_size: {tokenizer.vocab_size}")
            main_print(f"len(tokenizer): {len(tokenizer)}")
            main_print(f"pad_token_id: {pad_token_id}")
            main_print(f"bos_token_id: {bos_token_id}")
            main_print(f"eos_token_id: {eos_token_id}")

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

            model = self.model(
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
            )

            # ============================================================
            # Fine-tuning logic
            # ============================================================
            fine_tune = training_config.Finetune["fine_tune"]

            if fine_tune:

                # Step 1: Load the base checkpoint for fine-tuning
                base_checkpoint = training_config.Finetune["base_checkpoint"]
                if base_checkpoint is None:
                    raise ValueError(
                        "Base checkpoint for fine-tuning is not specified."
                    )
                if not Path(base_checkpoint).exists():
                    raise ValueError(
                        f"Base checkpoint {base_checkpoint} does not exist."
                    )
                main_print(f"Loading base checkpoint for fine-tuning: {base_checkpoint}")
                checkpoint = torch.load(
                    base_checkpoint,
                    map_location=device,
                )

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]

                model.load_state_dict(checkpoint, strict=False)

                # Step 2: Freeze the model parameters before applying fine-tuning methods
                freeze_model(model)

                # step 3: Apply the specified fine-tuning method
                fine_tune_method = training_config.Finetune["fine_tune_method"].lower()

                if fine_tune_method == "lora":
                    main_print("Applying LoRA fine-tuning...")

                    lora_target = getattr(
                        training_config.Finetune,
                        "lora_target",
                        "attention",
                    )

                    if lora_target in ["attention", "all"]:
                        model = add_lora_to_attention_layers(
                            model,
                            r=training_config.Finetune["Lora"]["lora_r"],
                            alpha=training_config.Finetune["Lora"]["lora_alpha"],
                            dropout=training_config.Finetune["Lora"]["lora_dropout"],
                            use_k_proj=training_config.Finetune["Lora"]["lora_use_k_proj"],
                        )

                    if lora_target in ["ffn", "all"]:
                        model = add_lora_to_ffn_layers(
                            model,
                            r=training_config.Finetune["Lora"]["lora_ffn_r"],
                            alpha=training_config.Finetune["Lora"]["lora_ffn_alpha"],
                            dropout=training_config.Finetune["Lora"]["lora_dropout"],
                            use_fc2=training_config.Finetune["Lora"]["lora_use_fc2"],
                        )
                # To be finished
                elif fine_tune_method == "adapter":
                    main_print("Applying Adapter fine-tuning...")

                    # Implement adapter fine-tuning logic here
                # To be finished
                elif fine_tune_method == "prefix":
                    main_print("Applying Prefix-tuning fine-tuning...")

                    # Implement prefix-tuning fine-tuning logic here
                # To be finished
                elif fine_tune_method == "prompt":
                    main_print("Applying Prompt-tuning fine-tuning...")
                    freeze_model(model)
                    # Implement prompt-tuning fine-tuning logic here
                else:
                    raise ValueError(f"Unsupported fine-tuning method: {fine_tune_method}")

                # step 4: Unfreeze specific parameters based on fine-tuning configuration
                fine_tune_lm_head = training_config.Finetune["fine_tune_lm_head"]
                fine_tune_layer_norm = training_config.Finetune["fine_tune_layer_norm"]

                if fine_tune_lm_head and hasattr(model, "head"):
                    main_print("Fine-tuning LM head...")
                    for p in model.head.parameters():
                        p.requires_grad = True

                # just in case, if the model has lm_head attribute (for example, if you have modified the GPT class to include lm_head)
                if fine_tune_lm_head and hasattr(model, "lm_head"):
                    main_print("Fine-tuning LM head...")
                    for p in model.lm_head.parameters():
                        p.requires_grad = True

                if fine_tune_layer_norm:
                    main_print("Fine-tuning layer normalization...")
                    for module in model.modules():
                        if isinstance(module, nn.LayerNorm):
                            for p in module.parameters():
                                p.requires_grad = True

            # ============================================================
            model.to(device)

            print_trainable_parameters(model)

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
                save_json(resolved_config, workdir / "config.json")

            if distributed:
                model = DDP(
                    model,
                    device_ids=[device.index],
                    output_device=device.index,
                    find_unused_parameters=False,
                )

            trainable_params = [
                p for p in model.parameters()
                if p.requires_grad
            ]

            optimizer = torch.optim.AdamW(
                trainable_params,
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

            start_epoch = 1
            best_val_loss = float("inf")
            best_val_perplexity = None
            best_epoch = 0
            patience_counter = 0

            # ============================================================
            # Resume training logic
            # ============================================================
            resume_history = None

            resume = training_config.Resume["resume"]
            print(f"Resume training: {resume}")

            if resume:
                main_print("Resuming training from checkpoint...")
                resume_path = training_config.Resume["resume_checkpoint"]

                if resume_path is not None:
                    resume_state = self.load_checkpoint_for_resume(
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

            history, best_path = self.run_training(
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
                fine_tune=fine_tune,
            )

            if is_main_process():
                history_path = checkpoint_dir / "history.pt"
                torch.save(history, history_path)
                save_json(history, checkpoint_dir / "history.json")

                print(f"Saved training history to {history_path}")
                print(f"Best model saved to {best_path}")

                if getattr(training_config, "plot_training_history", True):
                    plot_output_dir = workdir / "plots"
                    self.plot_training_history(
                        history=history,
                        output_dir=plot_output_dir,
                    )

        finally:
            cleanup_distributed()

    def run_training(
        self,
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
        fine_tune: bool = False,
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
                metrics = self.train_step(
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

            val_metrics = self.evaluate(
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

                self.append_history_csv(
                    checkpoint_dir / "history.csv",
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                    val_perplexity=val_ppl,
                    learning_rate=lr,
                )

                print(
                    f"[GPT-LoRA-DDP] epoch={epoch}/{epochs} "
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

                if fine_tune:
                    self.save_adapter_checkpoint(
                        path=checkpoint_dir / "last_adapter.pt",
                        model=model,
                        config=full_config,
                    )
                else:
                    self.save_checkpoint(
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
                    if fine_tune:
                        self.save_adapter_checkpoint(
                            path=checkpoint_dir / "best_adapter.pt",
                            model=model,
                            config=full_config,
                        )
                        print(f"Saved best adapter checkpoint to {checkpoint_dir / 'best_adapter.pt'}", flush=True)
                    else:
                        self.save_checkpoint(
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
                        f"No improvement: "
                        f"{patience_counter}/{early_stop_patience}",
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

    def train_step(
        self,
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
                [
                    p for p in model.parameters()
                    if p.requires_grad
                ],
                gradient_clip_value,
            )

        optimizer.step()

        return {"loss": loss.item()}

    @torch.no_grad()
    def evaluate(
        self,
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

    def save_checkpoint(
        self,
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
                    scheduler.state_dict()
                    if scheduler is not None
                    else None
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

    def save_lora_only_checkpoint(
        self,
        path: Path,
        model,
        config: Optional[dict] = None,
    ) -> None:
        if not is_main_process():
            return

        model_to_save = unwrap_model(model)

        lora_state = {
            k: v.cpu()
            for k, v in model_to_save.state_dict().items()
            if "lora_" in k
        }

        torch.save(
            {
                "lora_state_dict": lora_state,
                "config": config,
            },
            path,
        )

    def save_adapter_checkpoint(
        self,
        path: Path,
        model,
        config: Optional[dict] = None,
    ) -> None:
        if not is_main_process():
            return

        model_to_save = unwrap_model(model)

        adapter_state = {
            k: v.cpu()
            for k, v in model_to_save.state_dict().items()
            if (
                "lora_" in k
                or "lm_head" in k
                or ".head" in k
                or "norm" in k.lower()
                or "layernorm" in k.lower()
            )
        }

        torch.save(
            {
                "adapter_state_dict": adapter_state,
                "config": config,
            },
            path,
        )

    def load_checkpoint_for_resume(
        self,
        checkpoint_path: str | Path,
        model,
        optimizer=None,
        scheduler=None,
        device: torch.device | str = "cpu",
    ):
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)

        unwrap_model(model).load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )

        if (
            optimizer is not None
            and checkpoint.get("optimizer_state_dict") is not None
        ):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, torch.device(device))

        if (
            scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
        ):
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
        self,
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
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_perplexity",
                        "learning_rate",
                    ]
                )

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    val_perplexity,
                    learning_rate,
                ]
            )

    def plot_training_history(
        self,
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

    def load_configs(self):
        import tomllib

        with open(self.config_path, "rb") as f:
            config = tomllib.load(f)

        full_config = config
        training_config = SimpleNamespace(**config["GPTTrainingConfig"])
        gpt_config = SimpleNamespace(**config["GPTConfig"])
        tokenizer_config = SimpleNamespace(**config["TokenizerConfig"])
        dataset_config = SimpleNamespace(**config["DatasetConfig"])

        return (
            full_config,
            training_config,
            gpt_config,
            tokenizer_config,
            dataset_config,
        )