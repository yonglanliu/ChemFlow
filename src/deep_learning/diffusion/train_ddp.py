# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from src.deep_learning.graphormer.utils.data_collator import graphormer_collate_fn

from src.deep_learning.utils import (
    is_dist_available_and_initialized,
    get_rank,
    get_world_size,
    is_main_process,
    main_print,
    disable_tqdm,
    setup_distributed,
    cleanup_distributed,
    barrier,
    step_scheduler,
    build_scheduler,
    reduce_mean,
    save_json,
    get_resume_path,
    set_seed,
    namespace_to_dict,
)

from src.deep_learning.diffusion.train_utils import (
    move_batch_to_device,
    plot_training_history,
    load_checkpoint_for_resume,
    save_checkpoint,
    append_history_csv,
    load_graphormer_backbone,
)

from src.deep_learning.graphormer import (
    GraphormerGraphEncoder,
    GraphormerFeaturizer,
)

from src.deep_learning.graphormer.modules.dataset import (
    GraphormerMoleculeDataset,
    featurize_and_cache_dataset,
)

from src.deep_learning.diffusion.modules.denoiser import GraphormerDenoiser
from src.deep_learning.diffusion.modules.diffuser import GraphormerDiffuser


# ============================================================
# Train / Eval
# ============================================================

def train_step(
    model,
    batch,
    optimizer,
    device,
    gradient_clip_value: Optional[float] = 1.0,
) -> Dict[str, float]:
    model.train()

    batch = move_batch_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)

    out = model(batch)

    loss = out["loss"]
    atom_loss = out["atom_loss"]
    bond_loss = out["bond_loss"]

    bond_acc = out["bond_acc"]
    real_bond_acc = out["real_bond_acc"]
    no_bond_ratio = out["no_bond_ratio"]
    pred_no_bond_ratio = out["pred_no_bond_ratio"]

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"NaN/Inf train loss: "
            f"loss={loss.item()}, "
            f"atom_loss={atom_loss.item()}, "
            f"bond_loss={bond_loss.item()}"
        )

    loss.backward()

    if gradient_clip_value is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_value,
        )

    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "atom_loss": float(atom_loss.detach().item()),
        "bond_loss": float(bond_loss.detach().item()),
        "bond_acc": float(bond_acc.detach().item()),
        "real_bond_acc": float(real_bond_acc.detach().item()),
        "no_bond_ratio": float(no_bond_ratio.detach().item()),
        "pred_no_bond_ratio": float(pred_no_bond_ratio.detach().item()),
        "real_bond_recall": float(out["real_bond_recall"].detach().item()),
        "real_bond_type_acc_when_pred_real": float(out["real_bond_type_acc_when_pred_real"].detach().item()),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
) -> Dict[str, float]:
    torch.manual_seed(1234)
    model.eval()

    total_loss = 0.0
    total_atom_loss = 0.0
    total_bond_loss = 0.0

    total_bond_acc = 0.0
    total_real_bond_acc = 0.0
    total_no_bond_ratio = 0.0
    total_pred_no_bond_ratio = 0.0
    total_real_bond_recall = 0.0
    total_real_bond_type_acc_when_pred_real = 0.0
    valid_steps = 0

    progress = tqdm(
        loader,
        desc="Validation",
        disable=disable_tqdm(),
    )

    for step, batch in enumerate(progress):
        batch = move_batch_to_device(batch, device)
        out = model(batch)

        loss = out["loss"]
        atom_loss = out["atom_loss"]
        bond_loss = out["bond_loss"]

        bond_acc = out["bond_acc"]
        real_bond_acc = out["real_bond_acc"]
        no_bond_ratio = out["no_bond_ratio"]
        pred_no_bond_ratio = out["pred_no_bond_ratio"]
        real_bond_recall = out["real_bond_recall"]
        real_bond_type_acc_when_pred_real = out["real_bond_type_acc_when_pred_real"]    

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"NaN/Inf validation loss at batch {step}: "
                f"loss={loss.item()}, "
                f"atom_loss={atom_loss.item()}, "
                f"bond_loss={bond_loss.item()}"
            )

        total_loss += float(loss.item())
        total_atom_loss += float(atom_loss.item())
        total_bond_loss += float(bond_loss.item())

        total_bond_acc += float(bond_acc.item())
        total_real_bond_acc += float(real_bond_acc.item())
        total_no_bond_ratio += float(no_bond_ratio.item())
        total_pred_no_bond_ratio += float(pred_no_bond_ratio.item())
        total_real_bond_recall += float(real_bond_recall.item())
        total_real_bond_type_acc_when_pred_real += float(real_bond_type_acc_when_pred_real.item())

        valid_steps += 1

    n = max(valid_steps, 1)

    local_metrics = {
        "loss": total_loss / n,
        "atom_loss": total_atom_loss / n,
        "bond_loss": total_bond_loss / n,
        "bond_acc": total_bond_acc / n,
        "real_bond_acc": total_real_bond_acc / n,
        "no_bond_ratio": total_no_bond_ratio / n,
        "pred_no_bond_ratio": total_pred_no_bond_ratio / n,
        "real_bond_recall": total_real_bond_recall / n,
        "real_bond_type_acc_when_pred_real": total_real_bond_type_acc_when_pred_real / n,
    }

    if is_dist_available_and_initialized():
        for key in local_metrics:
            local_metrics[key] = reduce_mean(
                local_metrics[key],
                device=device,
            )

    return local_metrics


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
    start_epoch: int = 1,
    # best_val_loss: float = float("inf"),
    best_pred_no_bond_ratio: float = float("inf"),
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
            "train_atom_loss": [],
            "train_bond_loss": [],
            "train_bond_acc": [],
            "train_real_bond_acc": [],
            "train_no_bond_ratio": [],
            "train_pred_no_bond_ratio": [],
            "train_real_bond_recall": [],
            "train_real_bond_type_acc_when_pred_real": [],
            "val_loss": [],
            "val_atom_loss": [],
            "val_bond_loss": [],
            "val_bond_acc": [],
            "val_real_bond_acc": [],
            "val_no_bond_ratio": [],
            "val_pred_no_bond_ratio": [],
            "val_real_bond_recall": [],
            "val_real_bond_type_acc_when_pred_real": [],
            "learning_rate": [],
        }

    epochs = getattr(
        training_config,
        "num_epochs",
        getattr(training_config, "epochs", 10),
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
        getattr(training_config, "max_grad_norm", 1.0),
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
        running_atom_loss = 0.0
        running_bond_loss = 0.0

        running_bond_acc = 0.0
        running_real_bond_acc = 0.0
        running_no_bond_ratio = 0.0
        running_pred_no_bond_ratio = 0.0
        running_real_bond_recall = 0.0
        running_real_bond_type_acc_when_pred_real = 0.0

        num_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            disable=disable_tqdm(),
        )

        for batch in progress:
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                gradient_clip_value=gradient_clip_value,
            )

            running_loss += metrics["loss"]
            running_atom_loss += metrics["atom_loss"]
            running_bond_loss += metrics["bond_loss"]

            running_bond_acc += metrics["bond_acc"]
            running_real_bond_acc += metrics["real_bond_acc"]
            running_no_bond_ratio += metrics["no_bond_ratio"]
            running_pred_no_bond_ratio += metrics["pred_no_bond_ratio"]
            running_real_bond_recall += metrics["real_bond_recall"]
            running_real_bond_type_acc_when_pred_real += metrics["real_bond_type_acc_when_pred_real"]

            num_batches += 1

            if is_main_process() and not progress.disable:
                progress.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "atom": f"{metrics['atom_loss']:.4f}",
                    "bond": f"{metrics['bond_loss']:.4f}",
                    "real_bond_acc": f"{metrics['real_bond_acc']:.4f}",
                    "pred_no_bond": f"{metrics['pred_no_bond_ratio']:.4f}",
                    "real_bond_recall": f"{metrics['real_bond_recall']:.4f}",
                    "real_bond_type_acc_when_pred_real": f"{metrics['real_bond_type_acc_when_pred_real']:.4f}",
                })

        n_train = max(num_batches, 1)

        local_train_loss = running_loss / n_train
        local_train_atom_loss = running_atom_loss / n_train
        local_train_bond_loss = running_bond_loss / n_train

        local_train_bond_acc = running_bond_acc / n_train
        local_train_real_bond_acc = running_real_bond_acc / n_train
        local_train_no_bond_ratio = running_no_bond_ratio / n_train
        local_train_pred_no_bond_ratio = running_pred_no_bond_ratio / n_train
        local_train_real_bond_recall = running_real_bond_recall / n_train
        local_train_real_bond_type_acc_when_pred_real = running_real_bond_type_acc_when_pred_real / n_train

        train_metrics = {
            "loss": local_train_loss,
            "atom_loss": local_train_atom_loss,
            "bond_loss": local_train_bond_loss,
            "bond_acc": local_train_bond_acc,
            "real_bond_acc": local_train_real_bond_acc,
            "no_bond_ratio": local_train_no_bond_ratio,
            "pred_no_bond_ratio": local_train_pred_no_bond_ratio,
            "real_bond_recall": local_train_real_bond_recall,
            "real_bond_type_acc_when_pred_real": local_train_real_bond_type_acc_when_pred_real,

        }

        if is_dist_available_and_initialized():
            for key in train_metrics:
                train_metrics[key] = reduce_mean(
                    train_metrics[key],
                    device=device,
                )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        val_loss = val_metrics["loss"]

        step_scheduler(
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            val_loss=val_loss,
        )

        lr = optimizer.param_groups[0]["lr"]

        if is_main_process():
            history["epoch"].append(epoch)

            history["train_loss"].append(train_metrics["loss"])
            history["train_atom_loss"].append(train_metrics["atom_loss"])
            history["train_bond_loss"].append(train_metrics["bond_loss"])
            history["train_bond_acc"].append(train_metrics["bond_acc"])
            history["train_real_bond_acc"].append(train_metrics["real_bond_acc"])
            history["train_real_bond_recall"].append(train_metrics["real_bond_recall"])
            history["train_real_bond_type_acc_when_pred_real"].append(train_metrics["real_bond_type_acc_when_pred_real"])
            history["train_no_bond_ratio"].append(train_metrics["no_bond_ratio"])
            history["train_pred_no_bond_ratio"].append(train_metrics["pred_no_bond_ratio"])

            history["val_loss"].append(val_metrics["loss"])
            history["val_atom_loss"].append(val_metrics["atom_loss"])
            history["val_bond_loss"].append(val_metrics["bond_loss"])
            history["val_bond_acc"].append(val_metrics["bond_acc"])
            history["val_real_bond_acc"].append(val_metrics["real_bond_acc"])
            history["val_real_bond_recall"].append(val_metrics["real_bond_recall"])
            history["val_real_bond_type_acc_when_pred_real"].append(val_metrics["real_bond_type_acc_when_pred_real"])
            history["val_no_bond_ratio"].append(val_metrics["no_bond_ratio"])
            history["val_pred_no_bond_ratio"].append(val_metrics["pred_no_bond_ratio"])

            history["learning_rate"].append(lr)

            append_history_csv(
                checkpoint_dir / "history.csv",
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_atom_loss=train_metrics["atom_loss"],
                train_bond_loss=train_metrics["bond_loss"],
                train_bond_acc=train_metrics["bond_acc"],
                train_real_bond_acc=train_metrics["real_bond_acc"],
                train_real_bond_recall=train_metrics["real_bond_recall"],
                train_real_bond_type_acc_when_pred_real=train_metrics["real_bond_type_acc_when_pred_real"],
                train_no_bond_ratio=train_metrics["no_bond_ratio"],
                train_pred_no_bond_ratio=train_metrics["pred_no_bond_ratio"],
                val_loss=val_metrics["loss"],
                val_atom_loss=val_metrics["atom_loss"],
                val_bond_loss=val_metrics["bond_loss"],
                val_bond_acc=val_metrics["bond_acc"],
                val_real_bond_acc=val_metrics["real_bond_acc"],
                val_real_bond_recall=val_metrics["real_bond_recall"],
                val_real_bond_type_acc_when_pred_real=val_metrics["real_bond_type_acc_when_pred_real"],
                val_no_bond_ratio=val_metrics["no_bond_ratio"],
                val_pred_no_bond_ratio=val_metrics["pred_no_bond_ratio"],
                learning_rate=lr,
            )

            print(
                f"[GraphormerDiffusion] epoch={epoch}/{epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_atom_loss={train_metrics['atom_loss']:.4f} "
                f"train_bond_loss={train_metrics['bond_loss']:.4f} "
                f"train_bond_acc={train_metrics['bond_acc']:.4f} "
                f"train_real_bond_acc={train_metrics['real_bond_acc']:.4f} "
                f"train_no_bond_ratio={train_metrics['no_bond_ratio']:.4f} "
                f"train_pred_no_bond_ratio={train_metrics['pred_no_bond_ratio']:.4f} "
                f"train_real_bond_recall={train_metrics['real_bond_recall']:.4f} "
                f"train_real_bond_type_acc_when_pred_real={train_metrics['real_bond_type_acc_when_pred_real']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_atom_loss={val_metrics['atom_loss']:.4f} "
                f"val_bond_loss={val_metrics['bond_loss']:.4f} "
                f"val_bond_acc={val_metrics['bond_acc']:.4f} "
                f"val_real_bond_acc={val_metrics['real_bond_acc']:.4f} "
                f"val_no_bond_ratio={val_metrics['no_bond_ratio']:.4f} "
                f"val_pred_no_bond_ratio={val_metrics['pred_no_bond_ratio']:.4f} "
                f"val_real_bond_recall={val_metrics['real_bond_recall']:.4f} "
                f"val_real_bond_type_acc_when_pred_real={val_metrics['real_bond_type_acc_when_pred_real']:.4f} "
                f"lr={lr:.6g}",
                flush=True,
            )

            # improved = val_loss < best_val_loss
            improved = val_metrics['pred_no_bond_ratio'] < best_pred_no_bond_ratio

            if improved:
                #best_val_loss = val_loss
                best_pred_no_bond_ratio = val_metrics['pred_no_bond_ratio']
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_atom_loss=train_metrics["atom_loss"],
                train_bond_loss=train_metrics["bond_loss"],
                val_loss=val_metrics["loss"],
                val_atom_loss=val_metrics["atom_loss"],
                val_bond_loss=val_metrics["bond_loss"],
                #best_val_loss=best_val_loss,
                best_pred_no_bond_ratio=best_pred_no_bond_ratio,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
                history=history,
                config=full_config,
            )

            if improved:
                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    train_atom_loss=train_metrics["atom_loss"],
                    train_bond_loss=train_metrics["bond_loss"],
                    val_loss=val_metrics["loss"],
                    val_atom_loss=val_metrics["atom_loss"],
                    val_bond_loss=val_metrics["bond_loss"],
                    #best_val_loss=best_val_loss,
                    best_pred_no_bond_ratio=best_pred_no_bond_ratio,
                    best_epoch=best_epoch,
                    patience_counter=patience_counter,
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
# Main training entry
# ============================================================

def train(config_path: str | Path):
    import tomllib

    device, distributed = setup_distributed()

    try:
        if config_path is None:
            raise ValueError(
                "Missing config path. Usage: "
                "python train_graphormer_diffusion.py path/to/config.toml"
            )

        config_path = Path(config_path).expanduser().resolve()

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        full_config = config

        training_config = SimpleNamespace(
            **full_config["GraphormerDiffusionTrainingConfig"]
        )
        encoder_config = SimpleNamespace(
            **full_config["GraphormerEncoderConfig"]
        )
        denoiser_config = SimpleNamespace(
            **full_config["GraphormerDenoiserConfig"]
        )
        diffuser_config = SimpleNamespace(
            **full_config["GraphormerDiffusionConfig"]
        )
        featurizer_config = SimpleNamespace(
            **full_config["FeaturizerConfig"]
        )
        dataset_config = SimpleNamespace(
            **full_config["DatasetConfig"]
        )

        seed = getattr(training_config, "seed", 42)
        set_seed(seed + get_rank())

        workdir = Path(training_config.workdir).expanduser().resolve()
        checkpoint_dir = workdir / "checkpoints"
        cache_dir = workdir / "cache"

        for directory in [workdir, checkpoint_dir, cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        main_print(f"Using device: {device}")
        main_print(f"Distributed: {distributed}")
        main_print(f"World size: {get_world_size()}")
        main_print(f"Rank: {get_rank()}")

        featurizer = GraphormerFeaturizer(
            **namespace_to_dict(featurizer_config)
        )

        atom_mask_token = featurizer.atom_mask_token
        bond_mask_token = featurizer.bond_mask_token
        atom_pad_token = featurizer.atom_pad_token
        bond_pad_token = featurizer.bond_pad_token

        main_print(f"Atom mask token: {atom_mask_token}")
        main_print(f"Bond mask token: {bond_mask_token}")
        main_print(f"Atom pad token: {atom_pad_token}")
        main_print(f"Bond pad token: {bond_pad_token}")

        if is_main_process():
            main_print("Featurizing and caching dataset...")
            manifest = featurize_and_cache_dataset(
                dataset_config=dataset_config,
                featurizer=featurizer,
                cache_dir=cache_dir,
            )

        barrier()

        if not is_main_process():
            main_print("Loading cached dataset manifest...")
            manifest = featurize_and_cache_dataset(
                dataset_config=dataset_config,
                featurizer=featurizer,
                cache_dir=cache_dir,
            )

        barrier()

        train_dataset = GraphormerMoleculeDataset(
            shard_paths=manifest["train"],
        )

        val_dataset = GraphormerMoleculeDataset(
            shard_paths=manifest["val"],
        )

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
            train_shuffle = True
            val_shuffle = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            drop_last=False,
            num_workers=getattr(training_config, "num_workers", 0),
            pin_memory=(device.type == "cuda"),
            collate_fn=graphormer_collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=getattr(
                training_config,
                "eval_batch_size",
                training_config.batch_size,
            ),
            shuffle=val_shuffle,
            sampler=None,
            drop_last=False,
            num_workers=getattr(training_config, "num_workers", 0),
            pin_memory=(device.type == "cuda"),
            collate_fn=graphormer_collate_fn,
        )

        if getattr(training_config, "use_pretrained_encoder", False):
            graphormer_encoder = GraphormerGraphEncoder(
                **namespace_to_dict(encoder_config)
            )

            checkpoint_path = getattr(
                training_config,
                "pretrained_encoder_checkpoint",
                None,
            )

            if checkpoint_path is None:
                raise ValueError(
                    "Missing pretrained encoder checkpoint path. "
                    "Please specify pretrained_encoder_checkpoint in config."
                )

            main_print(
                f"Loading pretrained Graphormer encoder from {checkpoint_path}..."
            )

            graphormer_encoder = load_graphormer_backbone(
                graphormer_encoder,
                ckpt_path=checkpoint_path,
            )

            main_print("Loaded pretrained Graphormer encoder.")
        else:
            graphormer_encoder = GraphormerGraphEncoder(
                **namespace_to_dict(encoder_config)
            )

        full_config["encoder_config"] = graphormer_encoder.get_config()

        if getattr(training_config, "freeze_encoder", False):
            main_print("Freezing Graphormer encoder.")
            for p in graphormer_encoder.parameters():
                p.requires_grad = False

        denoiser = GraphormerDenoiser(
            encoder=graphormer_encoder,
            hidden_dim=denoiser_config.hidden_dim,
            num_atom_types=featurizer.num_atom_types,
            num_bond_types=featurizer.num_bond_types,
            dropout=denoiser_config.dropout,
            bond_pair_mode=getattr(denoiser_config, "bond_pair_mode", "sum"),
            num_timesteps=diffuser_config.num_timesteps,
        )

        full_config["denoiser_config"] = denoiser.get_config()

        model = GraphormerDiffuser(
            denoiser=denoiser,
            num_timesteps=diffuser_config.num_timesteps,
            atom_mask_token=atom_mask_token,
            bond_mask_token=bond_mask_token,
            atom_pad_token=atom_pad_token,
            bond_pad_token=bond_pad_token,
            atom_loss_weight=diffuser_config.atom_loss_weight,
            bond_loss_weight=diffuser_config.bond_loss_weight,
            bond_binary_loss_weight = diffuser_config.bond_binary_loss_weight,
            bond_type_loss_weight = diffuser_config.bond_type_loss_weight,
        ).to(device)

        full_config["diffuser_config"] = model.get_config()

        resolved_config = {
            **full_config,
            "ResolvedConfig": {
                "workdir": str(workdir),
                "checkpoint_dir": str(checkpoint_dir),
                "cache_dir": str(cache_dir),
                "device": str(device),
                "distributed": distributed,
                "world_size": get_world_size(),
                "rank": get_rank(),
                "seed": seed,
            },
        }

        if is_main_process():
            save_json(resolved_config, workdir / "config.json")

        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        base_model = model.module if hasattr(model, "module") else model

        encoder_params = [
            p for p in base_model.denoiser.encoder.parameters()
            if p.requires_grad
        ]

        other_params = [
            p for n, p in base_model.named_parameters()
            if p.requires_grad and not n.startswith("denoiser.encoder.")
        ]

        param_groups = []

        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": training_config.encoder_learning_rate,
            })

        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": training_config.learning_rate,
            })

        optimizer = torch.optim.AdamW(
            param_groups,
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
        # best_val_loss = float("inf")
        best_pred_no_bond_ratio = float("inf")
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
                load_optimizer=not getattr(
                    training_config,
                    "resume_model_only",
                    False,
                ),
            )

            start_epoch = resume_state["start_epoch"]
            # best_val_loss = resume_state["best_val_loss"]
            best_pred_no_bond_ratio = resume_state["best_pred_no_bond_ratio"]
            best_epoch = resume_state["best_epoch"]
            patience_counter = resume_state["patience_counter"]
            resume_history = resume_state["history"]

            main_print(f"Resumed from: {resume_state['checkpoint_path']}")
            main_print(f"Starting from epoch: {start_epoch}")
            # main_print(f"Best val loss so far: {best_val_loss:.6f}")
            main_print(f"Best pred no bond ratio so far: {best_pred_no_bond_ratio:.6f}")
            main_print(f"Best epoch so far: {best_epoch}")

        barrier()

        history, best_model_path = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=training_config,
            checkpoint_dir=checkpoint_dir,
            device=device,
            full_config=full_config,
            start_epoch=start_epoch,
            #best_val_loss=best_val_loss,
            best_pred_no_bond_ratio=best_pred_no_bond_ratio,
            best_epoch=best_epoch,
            patience_counter=patience_counter,
            history=resume_history,
        )

        if is_main_process():
            history_path = checkpoint_dir / "history.pt"
            torch.save(history, history_path)
            save_json(history, checkpoint_dir / "history.json")

            print(f"Saved training history to {history_path}")
            print(f"Best model saved to {best_model_path}")

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
            "  python train_graphormer_diffusion.py path/to/config.toml"
        )

    config_path = Path(sys.argv[1]).expanduser().resolve()
    train(config_path=config_path)


if __name__ == "__main__":
    main()