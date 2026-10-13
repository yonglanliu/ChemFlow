"""Run Chemprop's official CLI with retained periodic Lightning checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from chemflow.deep_learning.test_evaluation import regression_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _independent_reference_loader(loader) -> DataLoader | None:
    """Clone a Lightning loader without its distributed sampler or workers."""
    if loader is None:
        return None
    batch_size = getattr(loader, "batch_size", None)
    if batch_size is None:
        raise ValueError(
            "Chemprop task inspection requires a data loader with batch_size."
        )
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=loader.collate_fn,
        pin_memory=bool(getattr(loader, "pin_memory", False)),
        drop_last=False,
    )


def _relation_codes_from_masks(targets, lt_mask, gt_mask):
    """Convert optional Chemprop censor masks to -1/0/+1 relation codes."""
    relation = torch.zeros_like(targets, dtype=torch.int8)
    for mask, code in ((lt_mask, -1), (gt_mask, 1)):
        if mask is None:
            continue
        if mask.ndim < targets.ndim:
            mask = mask.reshape(targets.shape)
        relation = torch.where(
            mask.to(dtype=torch.bool),
            torch.full_like(relation, code),
            relation,
        )
    return relation


def main() -> None:
    # Imports stay inside main so ChemFlow can be imported without the optional
    # Chemprop/Lightning stack installed.
    from chemflow.deep_learning.chemprop import losses as _losses
    _losses.patch_chemprop_bounded_metric_masks()
    _losses.patch_chemprop_bounded_tracking_validation()
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.callbacks import Callback
    from chemprop.cli import train as train_module

    interval = int(os.environ.get("CHEMFLOW_CHECKPOINT_EVERY_N_EPOCHS", "0"))
    if interval < 0:
        raise ValueError(
            "CHEMFLOW_CHECKPOINT_EVERY_N_EPOCHS cannot be negative in the "
            "periodic Chemprop launcher."
        )

    class PeriodicModelCheckpoint(ModelCheckpoint):
        """Keep Chemprop's best/last policy and add full-state snapshots."""

        def on_validation_end(self, trainer, pl_module) -> None:
            super().on_validation_end(trainer, pl_module)
            if trainer.sanity_checking:
                return
            completed_epoch = int(trainer.current_epoch) + 1
            if interval == 0 or completed_epoch % interval != 0:
                return
            periodic_path = (
                Path(self.dirpath) / f"epoch_{completed_epoch:04d}.ckpt"
            )
            # Lightning coordinates this call across distributed ranks and
            # writes only from the appropriate global process.
            trainer.save_checkpoint(periodic_path)

    train_module.ModelCheckpoint = PeriodicModelCheckpoint

    inspect_metrics = os.environ.get("CHEMFLOW_INSPECT_TASK_METRICS", "1") == "1"
    target_names = json.loads(os.environ.get("CHEMFLOW_TARGET_NAMES", "[]"))
    transfer_checkpoint_value = os.environ.get("CHEMFLOW_TRANSFER_CHECKPOINT")
    transfer_mode = os.environ.get("CHEMFLOW_TRANSFER_MODE")
    transfer_task = os.environ.get("CHEMFLOW_TRANSFER_TASK")
    freeze_transfer_encoder = (
        os.environ.get("CHEMFLOW_TRANSFER_FREEZE_ENCODER", "0") == "1"
    )
    original_trainer = train_module.pl.Trainer

    class EpochMetricsCallback(Callback):
        """Evaluate independent full-data loaders on global rank zero."""

        def __init__(self) -> None:
            super().__init__()
            self._reference_loaders = {}

        @staticmethod
        def _loader(value):
            if isinstance(value, (list, tuple)):
                return value[0]
            return value

        def _reference_loader(self, value):
            active_loader = self._loader(value)
            if active_loader is None:
                return None
            key = id(active_loader)
            if key not in self._reference_loaders:
                self._reference_loaders[key] = _independent_reference_loader(
                    active_loader
                )
            return self._reference_loaders[key]

        def _evaluate_loader(self, trainer, model, loader):
            truths, predictions, relations = [], [], []
            loader = self._reference_loader(loader)
            if loader is None:
                return {}
            was_training = model.training
            transforms = [
                model.message_passing.V_d_transform,
                model.message_passing.graph_transform,
                model.X_d_transform,
                model.predictor.output_transform,
            ]
            transform_states = [transform.training for transform in transforms]
            model.eval()
            # Chemprop's train/validation loaders already contain scaled graph,
            # atom, and molecule descriptors. Its validation hook therefore
            # keeps input transforms in training (identity) mode. Calling only
            # model.eval() here would scale descriptors a second time and can
            # produce absurd predictions when RDKit features are enabled.
            model.message_passing.V_d_transform.train()
            model.message_passing.graph_transform.train()
            model.X_d_transform.train()
            # Predictions and standardized batch targets both need conversion
            # back to their original units for the inspection metrics.
            model.predictor.output_transform.eval()
            with torch.inference_mode():
                for batch in loader:
                    batch = trainer.strategy.batch_to_device(batch, model.device)
                    bmg, v_d, x_d, targets, _, lt_mask, gt_mask = batch
                    predicted = model(bmg, v_d, x_d)
                    if model.predictor.n_targets > 1:
                        predicted = predicted[..., 0]
                    transform = model.predictor.output_transform
                    raw_targets = transform(targets)
                    truths.append(raw_targets.detach().cpu().numpy())
                    predictions.append(predicted.detach().cpu().numpy())
                    relation = _relation_codes_from_masks(
                        targets, lt_mask, gt_mask
                    )
                    relations.append(relation.detach().cpu().numpy())
            model.train(was_training)
            for transform, state in zip(transforms, transform_states):
                transform.train(state)
            truth = np.concatenate(truths, axis=0)
            prediction = np.concatenate(predictions, axis=0)
            relation = np.concatenate(relations, axis=0)
            names = target_names or [f"task_{index}" for index in range(truth.shape[1])]
            return {
                name: regression_metrics(
                    truth[:, index], prediction[:, index], relation[:, index]
                )
                for index, name in enumerate(names)
            }

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if trainer.sanity_checking:
                return
            # Every DDP rank must reach the same synchronization points. Keep
            # worker ranks parked while rank zero evaluates the independent
            # full-data loaders; otherwise they can enter the next epoch's
            # gradient all-reduce while rank zero is still inside this callback
            # and eventually trigger the NCCL watchdog.
            trainer.strategy.barrier("chemflow_task_metrics_start")
            if not trainer.is_global_zero:
                trainer.strategy.barrier("chemflow_task_metrics_end")
                return
            epoch = int(trainer.current_epoch) + 1
            try:
                split_metrics = {
                    "train": self._evaluate_loader(
                        trainer, pl_module, trainer.train_dataloader
                    ),
                    "validation": self._evaluate_loader(
                        trainer, pl_module, trainer.val_dataloaders
                    ),
                }
                rows = []
                logged = {"completed_epoch": epoch}
                for split, metrics_by_task in split_metrics.items():
                    print(
                        f"{split.title()} metrics by task — epoch {epoch}:",
                        flush=True,
                    )
                    for task, metrics in metrics_by_task.items():
                        row = {
                            "epoch": epoch,
                            "split": split,
                            "task": task,
                            **metrics,
                        }
                        rows.append(row)
                        shown = ", ".join(
                            f"{name}={value:.4f}"
                            for name, value in metrics.items()
                            if name
                            not in {
                                "n",
                                "n_total",
                                "n_exact",
                                "n_censored",
                                "n_upper_bound",
                                "n_lower_bound",
                                "n_bound_violations",
                            }
                            and value is not None
                            and np.isfinite(value)
                        )
                        print(
                            f"  {task} (n={metrics['n']}): {shown}",
                            flush=True,
                        )
                        for name, value in metrics.items():
                            if value is not None and isinstance(value, (int, float)):
                                logged[f"{split}_{task}_{name}"] = value
                log_dir = Path(trainer.logger.log_dir)
                model_dir = log_dir.parent.parent
                output = model_dir / "training_task_metrics.csv"
                previous = []
                if output.is_file():
                    previous = (
                        pd.read_csv(output)
                        .query("epoch != @epoch")
                        .to_dict("records")
                    )
                pd.DataFrame(previous + rows).to_csv(output, index=False)
                trainer.logger.log_metrics(logged, step=trainer.global_step)
            finally:
                trainer.strategy.barrier("chemflow_task_metrics_end")

    class TransferVerificationCallback(Callback):
        """Verify and report transfer state after the model has been built."""

        def __init__(self) -> None:
            super().__init__()
            self.reported = False

        def on_fit_start(self, trainer, pl_module) -> None:
            if self.reported or transfer_checkpoint_value is None:
                return
            checkpoint = Path(transfer_checkpoint_value).expanduser()
            checkpoint_hash = _sha256(checkpoint)
            expected_hash = os.environ.get(
                "CHEMFLOW_TRANSFER_CHECKPOINT_SHA256"
            )
            if expected_hash and checkpoint_hash != expected_hash:
                raise RuntimeError(
                    "Transfer checkpoint changed between ChemFlow validation "
                    f"and Chemprop loading: expected {expected_hash}, got "
                    f"{checkpoint_hash}."
                )

            encoder = pl_module.message_passing
            encoder_parameters = sum(
                parameter.numel() for parameter in encoder.parameters()
            )
            trainable_encoder_parameters = sum(
                parameter.numel()
                for parameter in encoder.parameters()
                if parameter.requires_grad
            )
            predictor_parameters = sum(
                parameter.numel() for parameter in pl_module.predictor.parameters()
            )
            encoder_state = (
                "trainable" if trainable_encoder_parameters else "frozen"
            )
            keep_head = transfer_mode == "full_model"
            predictor_targets = int(pl_module.predictor.n_targets)
            if (
                keep_head
                and target_names
                and predictor_targets != len(target_names)
            ):
                raise ValueError(
                    "A transferred prediction head must have the same number "
                    f"of targets as the new dataset: checkpoint="
                    f"{predictor_targets}, dataset={len(target_names)}. Set "
                    "ChempropConfig.transfer_keep_head = false to create a "
                    "compatible new head."
                )
            loaded_component = (
                "complete_model_including_prediction_head"
                if keep_head
                else "message_passing_encoder_and_aggregation"
            )
            verification = {
                "load_succeeded": True,
                "transfer_mode": transfer_mode,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "checksum_verified": bool(
                    expected_hash and checkpoint_hash == expected_hash
                ),
                "loaded_component": loaded_component,
                "encoder_class": type(encoder).__name__,
                "encoder_parameters": encoder_parameters,
                "encoder_trainable_parameters": trainable_encoder_parameters,
                "encoder_state": encoder_state,
                "predictor_parameters": predictor_parameters,
                "predictor_targets": predictor_targets,
                "configured_task": transfer_task,
                "prediction_head_transferred": keep_head,
                "prediction_head_initialized_fresh": not keep_head,
            }
            if trainer.is_global_zero:
                model_dir = Path(trainer.logger.log_dir).parent.parent
                verification_path = model_dir / "transfer_load_verification.json"
                verification_path.write_text(
                    json.dumps(verification, indent=2) + "\n",
                    encoding="utf-8",
                )
                print("", flush=True)
                print("=" * 72, flush=True)
                print("CHEMFLOW TRANSFER WEIGHTS LOADED SUCCESSFULLY", flush=True)
                print(f"  Transfer mode: {transfer_mode}", flush=True)
                print(f"  Checkpoint: {checkpoint.resolve()}", flush=True)
                print(f"  SHA-256: {checkpoint_hash}", flush=True)
                print("  Checksum verified: yes", flush=True)
                print(f"  Loaded component: {loaded_component}", flush=True)
                print(f"  Encoder class: {type(encoder).__name__}", flush=True)
                print(f"  Encoder parameters: {encoder_parameters:,}", flush=True)
                print(f"  Encoder state: {encoder_state}", flush=True)
                print(
                    "  Prediction head: "
                    + ("transferred" if keep_head else "initialized fresh"),
                    flush=True,
                )
                print(f"  Verification: {verification_path}", flush=True)
                print("=" * 72, flush=True)
                print("", flush=True)
            self.reported = True

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            if freeze_transfer_encoder:
                pl_module.message_passing.eval()

    class MetricsTrainer(original_trainer):
        def __init__(self, *args, callbacks=None, **kwargs):
            callbacks = list(callbacks or [])
            if inspect_metrics:
                callbacks.append(EpochMetricsCallback())
            if transfer_checkpoint_value is not None:
                callbacks.append(TransferVerificationCallback())
            super().__init__(*args, callbacks=callbacks, **kwargs)

    train_module.pl.Trainer = MetricsTrainer

    if transfer_mode == "encoder_only" and freeze_transfer_encoder:
        # Chemprop 2.2 only accepts its public --freeze-encoder switch with
        # --checkpoint, not --from-foundation. Freeze the imported foundation
        # encoder immediately after model construction and before Lightning
        # configures its optimizer.
        original_build_model = train_module.build_model

        def build_model_with_frozen_encoder(*args, **kwargs):
            model = original_build_model(*args, **kwargs)
            model.message_passing.requires_grad_(False)
            model.message_passing.eval()
            return model

        train_module.build_model = build_model_with_frozen_encoder

    from chemprop.cli.main import main as chemprop_main

    chemprop_main()


if __name__ == "__main__":
    main()
