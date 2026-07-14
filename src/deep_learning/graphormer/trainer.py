# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import csv
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.deep_learning.graphormer.config import (
    GraphormerFinetuneClassificationConfig,
    GraphormerFinetuneRegressionConfig,
)
from src.deep_learning.graphormer.evaluation.classification import ClassificationEvaluator
from src.deep_learning.graphormer.evaluation.regression import RegressionEvaluator
from src.deep_learning.graphormer.models.graphormer_finetune_model import (
    GraphormerFineTuneClassificationModel,
    GraphormerFineTuneRegressionModel,
)
from src.deep_learning.graphormer.modules.dataset import (
    GraphormerMoleculeDataset,
    featurize_and_cache_dataset,
)
from src.deep_learning.graphormer.modules.graphormer_featurizer import GraphormerFeaturizer
from src.deep_learning.graphormer.utils.data_collator import graphormer_collate_fn
from src.deep_learning.utils import (
    barrier,
    build_scheduler,
    cleanup_distributed,
    disable_tqdm,
    get_rank,
    get_world_size,
    is_dist_available_and_initialized,
    is_main_process,
    main_print,
    namespace_to_dict,
    save_json,
    set_seed,
    setup_distributed,
    step_scheduler,
    unwrap_model,
)
from functools import partial
import numpy as np

def update_dataclass_from_config(target: Any, source: Any, *, strict: bool = False) -> Any:
    if not is_dataclass(target):
        raise TypeError(f"target must be a dataclass instance, got {type(target).__name__}")

    target_fields = {item.name for item in fields(target)}

    if isinstance(source, dict):
        values = source
    elif is_dataclass(source):
        values = {item.name: getattr(source, item.name) for item in fields(source)}
    elif isinstance(source, SimpleNamespace) or hasattr(source, "__dict__"):
        values = vars(source)
    else:
        raise TypeError(f"Unsupported source config type: {type(source).__name__}")

    unknown = []
    for name, value in values.items():
        if name in target_fields:
            setattr(target, name, value)
        else:
            unknown.append(name)

    if strict and unknown:
        raise ValueError(f"Unknown fields for {type(target).__name__}: {unknown}")

    return target


def config_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return config_to_dict(asdict(value))
    if isinstance(value, SimpleNamespace):
        return {k: config_to_dict(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: config_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_to_dict(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def dict_to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [dict_to_namespace(v) for v in value]
    return value


def config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


# def unwrap_model(model: nn.Module) -> nn.Module:
#     return model.module if isinstance(model, DDP) else model


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    return batch


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_optimizer_parameter_groups(
    model: nn.Module,
    training_config,
) -> list[dict[str, Any]]:
    model = unwrap_model(model)

    encoder_parameters = []
    lora_parameters = []
    head_parameters = []

    encoder_names = []
    lora_names = []
    head_names = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        lower_name = name.lower()

        # LoRA is inside encoder, so check it first.
        if "lora_" in lower_name:
            lora_parameters.append(parameter)
            lora_names.append(name)

        elif (
            name.startswith("regression_head.")
            or name.startswith("classification_head.")
        ):
            head_parameters.append(parameter)
            head_names.append(name)

        elif name.startswith("encoder."):
            encoder_parameters.append(parameter)
            encoder_names.append(name)

        else:
            raise ValueError(
                f"Unclassified trainable parameter: {name}"
            )

    base_lr = float(
        training_config.learning_rate
    )

    parameter_groups = []

    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": float(
                    config_get(
                        training_config,
                        "encoder_learning_rate",
                        base_lr,
                    )
                ),
                "name": "encoder",
            }
        )

    if lora_parameters:
        parameter_groups.append(
            {
                "params": lora_parameters,
                "lr": float(
                    config_get(
                        training_config,
                        "lora_learning_rate",
                        base_lr,
                    )
                ),
                "name": "lora",
            }
        )

    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": float(
                    config_get(
                        training_config,
                        "head_learning_rate",
                        base_lr,
                    )
                ),
                "name": "head",
            }
        )

    print(
        f"Encoder: {len(encoder_names)} tensors"
    )
    print(
        f"LoRA: {len(lora_names)} tensors"
    )
    print(
        f"Head: {len(head_names)} tensors"
    )

    return parameter_groups


def get_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    return {
        group.get("name", f"group_{index}"): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }

class GraphormerDDPTrainer:
    def __init__(self, config_path: str | Path | None = None) -> None:

        # -------------------------------------------------------------
        # Setup distributed training and device
        # -------------------------------------------------------------
        self.device, self.distributed = setup_distributed()

        # -------------------------------------------------------------
        # Load configurations
        # -------------------------------------------------------------
        if config_path is None:
            raise ValueError("Config file is not specified.")

        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {self.config_path}")

        (self.base_config, self.training_config, raw_model_config, self.featurizer_config, self.dataset_config) = self.load_configs()

        set_seed(int(config_get(self.training_config, "seed", 42)) + get_rank())

        self.workdir = Path(self.base_config.workdir).expanduser().resolve()
        self.checkpoint_dir = self.workdir / "checkpoints"
        self.cache_dir = self.workdir / "cache"

        for directory in (self.workdir, self.checkpoint_dir, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)

        main_print(f"Using device: {self.device}")
        main_print(f"Distributed: {self.distributed}")
        main_print(f"World size: {get_world_size()}")
        main_print(f"Rank: {get_rank()}")


        # -------------------------------------------------------------
        # Setup model
        # -------------------------------------------------------------
        self.task = str(self.base_config.task).lower()
        if self.task == "regression":
            config = GraphormerFinetuneRegressionConfig()
            # Intentionally update only from GraphormerConfig.
            self.model_config = update_dataclass_from_config(config, raw_model_config)
            self.model = GraphormerFineTuneRegressionModel(cfg=self.model_config)
            self.evaluator = RegressionEvaluator()
        elif self.task == "classification":
            config = GraphormerFinetuneClassificationConfig()
            # Intentionally update only from GraphormerConfig.
            self.model_config = update_dataclass_from_config(config, raw_model_config)
            self.model = GraphormerFineTuneClassificationModel(cfg=self.model_config)
            loss_type = self.model_config.loss_type.lower()
            num_classes = int(self.model_config.num_classes)

            if loss_type == "bce":
                evaluator_loss_type = "binary"

            elif loss_type == "cross_entropy":
                if num_classes == 2:
                    evaluator_loss_type = "binary"
                elif num_classes > 2:
                    evaluator_loss_type = "multiclass"
                else:
                    raise ValueError(
                        "cross_entropy requires num_classes >= 2, "
                        f"got num_classes={num_classes}."
                    )

            else:
                raise ValueError(
                    f"Unsupported classification loss_type: {loss_type!r}. "
                    "Expected 'bce' or 'cross_entropy'."
                )

            main_print(
                f"Using evaluator task type: {evaluator_loss_type}"
            )
            self.evaluator = ClassificationEvaluator(
                loss_type=evaluator_loss_type,
                num_classes=self.model_config.num_classes,
            )
        else:
            raise ValueError(
                f"Unsupported task '{self.task}'. Expected regression or classification."
            )

        if hasattr(self.model_config, "multi_hop_max_dist"):
            self.featurizer_config.multi_hop_max_dist = self.model_config.multi_hop_max_dist
            self.dataset_config.multi_hop_max_dist = self.model_config.multi_hop_max_dist
        if hasattr(self.model_config, "max_nodes"):
            self.dataset_config.max_nodes = self.model_config.max_nodes
        if hasattr(self.model_config, "spatial_pos_max"):
            self.dataset_config.spatial_pos_max = self.model_config.spatial_pos_max

        # -------------------------------------------------------------
        # Setup featurizer
        # -------------------------------------------------------------
        self.featurizer = GraphormerFeaturizer(**namespace_to_dict(self.featurizer_config))
        self._print_featurizer_tokens()


        # -------------------------------------------------------------
        # Load dataset and create DataLoaders
        # -------------------------------------------------------------
        (self.train_loader, self.val_loader, self.test_loader, self.train_sampler) = self.load_dataset(
            dataset_config=self.dataset_config,
            featurizer=self.featurizer,
            cache_dir=self.cache_dir,
            device=self.device,
            training_config=self.training_config,
            distributed=self.distributed,
        )

    def train(self) -> None:
        try:
            self.model.to(self.device)

            if self.distributed:
                if self.device.type != "cuda" or self.device.index is None:
                    raise RuntimeError("CUDA DDP requires a CUDA device index.")
                self.model = DDP(
                    self.model,
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                    find_unused_parameters=False,
                )

            # self.print_trainable_parameters(self.model)

            # Assign trainable parameters to optimizer
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable:
                raise RuntimeError("No trainable parameters remain.")

            parameter_groups = build_optimizer_parameter_groups(
                    model=self.model,
                    training_config=self.training_config,
                )

            optimizer = torch.optim.AdamW(
                parameter_groups,
                weight_decay=float(
                    self.training_config.weight_decay
                ),
            )

            epochs = int(config_get(self.training_config, "num_epochs", config_get(self.training_config, "epochs", 100),))
            scheduler = build_scheduler(optimizer=optimizer, training_config=self.training_config, total_epochs=epochs,)

            resolved_config = {
                "BaseConfig": config_to_dict(self.base_config),
                "GraphormerTrainingConfig": config_to_dict(self.training_config),
                "GraphormerConfig": config_to_dict(self.model_config),
                "DatasetConfig": config_to_dict(self.dataset_config),
                "FeaturizerConfig": config_to_dict(
                    self.featurizer_config
                ),
                "ResolvedConfig": {
                    "workdir": str(self.workdir),
                    "checkpoint_dir": str(self.checkpoint_dir),
                    "cache_dir": str(self.cache_dir),
                    "device": str(self.device),
                    "distributed": self.distributed,
                    "world_size": get_world_size(),
                    "rank": get_rank(),
                },
            }
            if is_main_process():
                save_json(resolved_config, self.workdir / "config.json")

            # -------------------------------------------------------------
            # !!! Select monitor metric and mode for early stopping
            # -------------------------------------------------------------
            monitor_mode = str(config_get(self.training_config, "monitor_mode", "min")).lower()
            initial_best = float("inf") if monitor_mode == "min" else float("-inf")

            resume_state = {
                "start_epoch": 1,
                "best_metric": initial_best,
                "best_epoch": 0,
                "patience_counter": 0,
                "history": None,
            }

            if bool(config_get(self.training_config, "resume", False)):
                resume_path = config_get(self.training_config, "resume_checkpoint", None)
                if not resume_path:
                    raise ValueError("resume=True but resume_checkpoint is missing.")
                resume_state = self.load_checkpoint_for_resume(
                    checkpoint_path=resume_path,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=self.device,
                    fallback_best_metric=initial_best,
                )

            barrier()  # Ensure all processes have loaded the checkpoint before training

            history, best_path = self.run_training(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                train_sampler=self.train_sampler,
                optimizer=optimizer,
                scheduler=scheduler,
                training_config=self.training_config,
                checkpoint_dir=self.checkpoint_dir,
                device=self.device,
                full_config=resolved_config,
                **{
                    key: resume_state[key]
                    for key in (
                        "start_epoch",
                        "best_metric",
                        "best_epoch",
                        "patience_counter",
                        "history",
                    )
                },
            )

            # Save final training history and best model checkpoint
            if is_main_process():
                torch.save(history, self.checkpoint_dir / "history.pt")
                save_json(history, self.checkpoint_dir / "history.json")
                print(f"Best model saved to {best_path}")
                if bool(config_get(self.training_config, "plot_training_history", True)):
                    self.plot_training_history(history, self.workdir / "plots")

            # -------------------------------------------------------------
            # Evaluate on hold-out test set if available
            # -------------------------------------------------------------
            if bool(config_get(self.training_config, "evaluate_test", True)) and self.test_loader is not None:
                barrier()

                best_checkpoint = self.load_model_checkpoint(checkpoint_path=best_path, model=self.model, device=self.device)

                barrier()

                test_metrics, curve_data = self.evaluate(model=self.model, loader=self.test_loader, device=self.device, prefix="test", return_curve_data=True)

                if is_main_process():
                    
                    save_json(
                        {
                            "checkpoint": str(best_path),
                            "best_epoch": int(
                                best_checkpoint.get("best_epoch", 0)
                            ),
                            **test_metrics,
                        },
                        self.workdir / "test_metrics.json",
                    )

                    if curve_data is not None:
                        save_json(curve_data, self.workdir / "test_curve_data.json")
                        self.plot_classification_curves(
                            curve_data=curve_data,
                            output_dir=self.workdir / "plots",
                            prefix="test",
                        )


                    print(f"Best model loaded from {best_path}")

                    print("[Hold-out Test] " + " ".join(f"{name}={value:.4f}" for name, value in test_metrics.items()))

                    # if bool(config_get(self.training_config, "plot_evaluation_metrics", True)):
                    #     self.plot_training_history(history, self.workdir / "plots")
        finally:
            cleanup_distributed()

    def run_training(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_sampler: Optional[DistributedSampler],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        training_config: Any,
        checkpoint_dir: Path,
        device: torch.device,
        full_config: dict,
        start_epoch: int = 1,
        best_metric: float = float("inf"),
        best_epoch: int = 0,
        patience_counter: int = 0,
        history: Optional[dict] = None,
    ) -> tuple[dict, Path]:
        best_path = checkpoint_dir / "best_model.pt"
        last_path = checkpoint_dir / "last_model.pt"
        best_adapter_path = checkpoint_dir / "best_adapter.pt"
        last_adapter_path = checkpoint_dir / "last_adapter.pt"

        monitor_metric = str(config_get(training_config, "monitor_metric", "val_loss"))
        monitor_mode = str(config_get(training_config, "monitor_mode", "min")).lower()
        if monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'.")

        learning_rates = get_learning_rates(optimizer)

        if history is None:
            history = {
                "epoch": [],
                "train_loss": [],
            }

            for group_name in learning_rates:
                history[f"{group_name}_lr"] = []

        epochs = int(
            config_get(
                training_config,
                "num_epochs",
                config_get(training_config, "epochs", 100),
            )
        )
        early_stopping = bool(config_get(training_config, "early_stopping", True))
        patience = int(config_get(training_config, "early_stopping_patience", 10))
        gradient_clip = config_get(training_config, "gradient_clip_value", 1.0)
        scheduler_name = config_get(
            training_config,
            "scheduler",
            config_get(training_config, "schedular", None),
        )
        save_adapter = bool(getattr(self.model_config, "use_lora", False))

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
                value = self.train_step(
                    model,
                    batch,
                    optimizer,
                    device,
                    gradient_clip,
                )
                running_loss += value
                num_batches += 1
                if is_main_process() and not progress.disable:
                    progress.set_postfix(loss=f"{value:.4f}")
                    
            training_dtype = next(model.parameters()).dtype
            stats = torch.tensor(
                [running_loss, num_batches],
                dtype=training_dtype,
                device=device,
            )
            if is_dist_available_and_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            avg_train_loss = float((stats[0] / stats[1].clamp_min(1)).item())

            val_metrics = self.evaluate(model, val_loader, device)
            if monitor_metric not in val_metrics:
                raise KeyError(
                    f"Metric '{monitor_metric}' missing. "
                    f"Available: {sorted(val_metrics)}"
                )

            val_loss = float(val_metrics["val_loss"])
            monitor_value = float(val_metrics[monitor_metric])
            if not torch.isfinite(torch.tensor(monitor_value)):
                raise FloatingPointError(
                    f"{monitor_metric} is non-finite: {monitor_value}"
                )

            step_scheduler(
                scheduler=scheduler,
                scheduler_name=scheduler_name,
                val_loss=val_loss,
            )
            learning_rates = get_learning_rates(optimizer)
            improved = (
                monitor_value < best_metric
                if monitor_mode == "min"
                else monitor_value > best_metric
            )

            if is_main_process():
                if improved:
                    best_metric = monitor_value
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                history["epoch"].append(epoch)
                history["train_loss"].append(avg_train_loss)
                learning_rates = get_learning_rates(optimizer)

                for group_name, lr in learning_rates.items():
                    history[f"{group_name}_lr"].append(lr)

                for name, value in val_metrics.items():
                    history.setdefault(name, []).append(float(value))

                self.append_history_csv(
                    checkpoint_dir / "history.csv",
                    epoch,
                    avg_train_loss,
                    learning_rates,
                    val_metrics,
                )

                metric_text = " ".join(
                    f"{name}={float(value):.4f}"
                    for name, value in val_metrics.items()
                )
                print(
                    f"[Graphormer-DDP] epoch={epoch}/{epochs} "
                    f"train_loss={avg_train_loss:.4f} {metric_text} "
                    f"{' '.join(f'{name}_lr={lr:.4e}' for name, lr in learning_rates.items())}",
                    flush=True,
                )

                common_checkpoint_args = dict(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_metrics=val_metrics,
                    best_metric=best_metric,
                    monitor_metric=monitor_metric,
                    monitor_mode=monitor_mode,
                    monitor_value=monitor_value,
                    best_epoch=best_epoch,
                    patience_counter=patience_counter,
                    scheduler=scheduler,
                    history=history,
                    config=full_config,
                )
                self.save_checkpoint(path=last_path, **common_checkpoint_args)
                if save_adapter:
                    self.save_adapter_checkpoint(
                        last_adapter_path,
                        model,
                        full_config,
                    )
                if improved:
                    self.save_checkpoint(path=best_path, **common_checkpoint_args)
                    if save_adapter:
                        self.save_adapter_checkpoint(
                            best_adapter_path,
                            model,
                            full_config,
                        )
                    print(f"Saved best checkpoint to {best_path}")

            state = torch.tensor(
                [best_metric, best_epoch, patience_counter],
                dtype=training_dtype,
                device=device,
            )
            if is_dist_available_and_initialized():
                dist.broadcast(state, src=0)
            best_metric = float(state[0].item())
            best_epoch = int(state[1].item())
            patience_counter = int(state[2].item())

            stop = torch.zeros((), dtype=torch.int32, device=device)
            if is_main_process() and early_stopping and patience_counter >= patience:
                stop.fill_(1)
            if is_dist_available_and_initialized():
                dist.broadcast(stop, src=0)
            if stop.item() == 1:
                if is_main_process():
                    print(
                        f"Early stopping at epoch {epoch}. "
                        f"Best epoch={best_epoch}; "
                        f"best {monitor_metric}={best_metric:.4f}"
                    )
                break

        return history, best_path

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gradient_clip_value: Optional[float] = 1.0,
    ) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = self.forward_batch(model, move_batch_to_device(batch, device))
        loss = self.extract_loss(outputs)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        if gradient_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(gradient_clip_value),
            )
        optimizer.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, device: torch.device, prefix: str = "val", return_curve_data: bool = False) -> dict[str, float]:

        model.eval()

        predictions_list = []
        targets_list = []

        total_loss = 0.0
        total_samples = 0

        description = ("Validation" if prefix == "val" else "Hold-out test")

        for batch in tqdm(loader, desc=description, disable=disable_tqdm()):

            batch = move_batch_to_device(batch, device)
            outputs = self.forward_batch(model, batch)
            loss = self.extract_loss(outputs)  
            predictions = outputs.get("predictions", outputs.get("logits"))

            if predictions is None:
                raise KeyError("Output must contain predictions or logits.")

            # if batch is a dict, get targets from 'y'; if it's an object, get targets from attribute 'y'
            targets = (batch.get("y") if isinstance(batch, dict) else getattr(batch, "y", None))

            if targets is None:
                raise KeyError(f"{description} batch does not contain y.")

            batch_size = int(targets.size(0))

            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            predictions_list.append(predictions.detach())
            targets_list.append(targets.detach())

        if not predictions_list:
            raise RuntimeError(f"{description} loader produced no batches.")

        # Combine predictions and targets across all batches
        local_predictions = torch.cat(predictions_list, dim=0)
        local_targets = torch.cat(targets_list, dim=0)

        training_dtype = next(model.parameters()).dtype

        # Combine loss and sample counts across all processes
        loss_stats = torch.tensor([total_loss, total_samples], dtype=training_dtype, device=device)

        # Use all_reduce to sum the loss and sample counts across all processes
        if is_dist_available_and_initialized():
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)

        average_loss = float((loss_stats[0] / loss_stats[1].clamp_min(1)).item())

        predictions = self.gather_variable_tensors(local_predictions).cpu()
        targets = self.gather_variable_tensors(local_targets).cpu()

        metrics = self.evaluator.compute(predictions=predictions, targets=targets, loss=average_loss, prefix=prefix)
        metrics = {name: float(value) for name, value in metrics.items()}

        if not return_curve_data:
            return metrics

        if not isinstance(
            self.evaluator,
            ClassificationEvaluator,
        ):
            return metrics, {}

        curve_data = self.evaluator.compute_curve_data(
            predictions=predictions,
            targets=targets,
        )

        return metrics, curve_data

    @staticmethod
    def gather_variable_tensors(tensor: torch.Tensor) -> torch.Tensor:
        if not is_dist_available_and_initialized():
            return tensor

        world_size = dist.get_world_size()
        size = torch.tensor([tensor.size(0)], dtype=torch.long, device=tensor.device)
        sizes = [torch.zeros_like(size) for _ in range(world_size)]
        dist.all_gather(sizes, size)
        sizes_int = [int(item.item()) for item in sizes]
        max_size = max(sizes_int)

        if tensor.size(0) < max_size:
            padding = torch.zeros(
                (max_size - tensor.size(0), *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat([tensor, padding], dim=0)

        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)

        return torch.cat([item[:size] for item, size in zip(gathered, sizes_int)], dim=0)

    @staticmethod
    def forward_batch(model: nn.Module, batch: Any) -> Any:
        if isinstance(batch, dict):
            return model(batched_data=batch)
        if isinstance(batch, (tuple, list)):
            return model(*batch)
        return model(batch)

    @staticmethod
    def extract_loss(outputs: Any) -> torch.Tensor:
        if torch.is_tensor(outputs):
            loss = outputs
        elif isinstance(outputs, Mapping) and "loss" in outputs:
            loss = outputs["loss"]
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, (tuple, list)) and outputs:
            loss = outputs[0]
        else:
            raise TypeError("Could not extract loss from model output.")

        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise ValueError("Extracted loss must be a scalar tensor.")
        return loss

    def save_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_loss: float,
        val_metrics: Mapping[str, float],
        best_metric: float,
        monitor_metric: str,
        monitor_mode: str,
        monitor_value: float,
        best_epoch: int,
        patience_counter: int,
        scheduler: Any = None,
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
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "train_loss": train_loss,
                "val_metrics": dict(val_metrics),
                "best_metric": best_metric,
                "monitor_metric": monitor_metric,
                "monitor_mode": monitor_mode,
                "monitor_value": monitor_value,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "history": history,
                "config": config,
            },
            path,
        )

    def save_adapter_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        config: Optional[dict] = None,
    ) -> None:
        if not is_main_process():
            return

        model_to_save = unwrap_model(model)
        trainable_names = {
            name for name, parameter in model_to_save.named_parameters()
            if parameter.requires_grad
        }
        adapter_state = {
            name: value.detach().cpu()
            for name, value in model_to_save.state_dict().items()
            if name in trainable_names
        }
        if not adapter_state:
            raise RuntimeError("No trainable parameters found for adapter checkpoint.")

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"adapter_state_dict": adapter_state, "config": config},
            path,
        )

    def load_checkpoint_for_resume(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        device: torch.device | str = "cpu",
        fallback_best_metric: float = float("inf"),
    ) -> dict:
        checkpoint_path = Path(
            checkpoint_path
        ).expanduser().resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model_to_load = unwrap_model(model)

        # ============================================================
        # Full training checkpoint
        # ============================================================
        if "model_state_dict" in checkpoint:
            model_to_load.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

            if (
                optimizer is not None
                and checkpoint.get("optimizer_state_dict")
            ):
                optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )
                move_optimizer_state_to_device(
                    optimizer,
                    torch.device(device),
                )

            if (
                scheduler is not None
                and checkpoint.get("scheduler_state_dict")
            ):
                scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"]
                )

            return {
                "start_epoch": int(
                    checkpoint.get("epoch", 0)
                ) + 1,
                "best_metric": float(
                    checkpoint.get(
                        "best_metric",
                        fallback_best_metric,
                    )
                ),
                "best_epoch": int(
                    checkpoint.get(
                        "best_epoch",
                        checkpoint.get("epoch", 0),
                    )
                ),
                "patience_counter": int(
                    checkpoint.get(
                        "patience_counter",
                        0,
                    )
                ),
                "history": checkpoint.get("history"),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_type": "full",
            }

        # ============================================================
        # Adapter-only checkpoint
        # ============================================================
        if "adapter_state_dict" in checkpoint:
            incompatible = model_to_load.load_state_dict(
                checkpoint["adapter_state_dict"],
                strict=False,
            )

            if incompatible.unexpected_keys:
                raise RuntimeError(
                    "Unexpected adapter checkpoint keys: "
                    f"{incompatible.unexpected_keys}"
                )

            print(
                f"Loaded adapter checkpoint from "
                f"{checkpoint_path}"
            )
            print(
                f"Missing keys are expected for the frozen "
                f"base encoder: {len(incompatible.missing_keys)}"
            )

            # Adapter loading is not a strict resume:
            # optimizer/scheduler/history start fresh.
            return {
                "start_epoch": int(
                    checkpoint.get("epoch", 0)
                ) + 1,
                "best_metric": fallback_best_metric,
                "best_epoch": 0,
                "patience_counter": 0,
                "history": None,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_type": "adapter",
            }

        raise KeyError(
            "Unsupported checkpoint format. Expected either "
            "'model_state_dict' for a full checkpoint or "
            "'adapter_state_dict' for an adapter checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    def append_history_csv(
        self,
        path: str | Path,
        epoch: int,
        train_loss: float,
        learning_rates: Mapping[str, float],
        metrics: Mapping[str, float],
    ) -> None:
        path = Path(path)
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            **{name: float(value) for name, value in metrics.items()},
            **{f"{name}_lr": float(value) for name, value in learning_rates.items()},
        }
        fieldnames = list(row)

        if path.exists():
            with path.open("r", newline="") as file:
                header = next(csv.reader(file), None)
            if header != fieldnames:
                raise ValueError(
                    f"History columns changed. Existing={header}; current={fieldnames}"
                )

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def plot_training_history(self, history: dict, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dpi = int(config_get(self.training_config, "plot_dpi", 300))
        epochs = history.get("epoch", [])
        if not epochs:
            return

        if "train_loss" in history and "val_loss" in history:
            plt.figure(figsize=(7, 5))
            plt.plot(epochs, history["train_loss"], label="Train")
            plt.plot(epochs, history["val_loss"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "loss_curve.png", dpi=dpi)
            plt.close()

        for name, values in history.items():
            if (name.startswith("val_") or name.endswith("_lr")) and name != "val_loss" and len(values) == len(epochs):
                plt.figure(figsize=(7, 5))
                plt.plot(epochs, values)
                plt.xlabel("Epoch")
                plt.ylabel(name)
                plt.tight_layout()
                plt.savefig(output_dir / f"{name}.png", dpi=dpi)
                plt.close()


    def load_dataset(
        self,
        dataset_config: Any,
        featurizer: GraphormerFeaturizer,
        cache_dir: Path,
        device: torch.device,
        training_config: Any,
        distributed: bool,
    ) -> tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
        if is_main_process():
            manifest = featurize_and_cache_dataset(
                dataset_config=dataset_config,
                featurizer=featurizer,
                cache_dir=cache_dir,
            )
        else:
            manifest = None

        barrier()
        if manifest is None:
            manifest = featurize_and_cache_dataset(
                dataset_config=dataset_config,
                featurizer=featurizer,
                cache_dir=cache_dir,
            )

        train_dataset = GraphormerMoleculeDataset(manifest["train"])
        val_dataset = GraphormerMoleculeDataset(manifest["val"])

        if bool(config_get(self.training_config, "evaluate_test", True)) and config_get(dataset_config, "test_fraction", 0.0) > 0.0:
            test_dataset = GraphormerMoleculeDataset(manifest["test"])
        else:
            test_dataset = None

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
            if test_dataset is not None:
                test_sampler = DistributedSampler(
                    test_dataset,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                    shuffle=False,
                    drop_last=False,
                )
            else:
                test_sampler = None

            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None
            shuffle = True

        collate_fn = partial(
            graphormer_collate_fn,
            max_nodes=int(config_get(dataset_config, "max_nodes", 128,)),
            multi_hop_max_dist=int(config_get(dataset_config, "multi_hop_max_dist", 5,)),
            spatial_pos_max=int(config_get(dataset_config, "spatial_pos_max", 1024,)),
            )
        
        common = {
            "num_workers": int(config_get(training_config, "num_workers", 0)),
            "pin_memory": device.type == "cuda",
            "collate_fn": collate_fn,
        }

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(training_config.batch_size),
            shuffle=shuffle,
            sampler=train_sampler,
            drop_last=True,
            **common,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config_get(training_config,"eval_batch_size",training_config.batch_size,)),
            shuffle=False,
            sampler=val_sampler,
            drop_last=False,
            **common,
        )

        test_loader = None

        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=int(config_get(training_config,"eval_batch_size",training_config.batch_size,)),
                shuffle=False,
                sampler=test_sampler,
                drop_last=False,
                **common,
            )
        return (train_loader, val_loader, test_loader, train_sampler)

    @staticmethod
    def print_trainable_parameters(model: nn.Module) -> None:
        model = unwrap_model(model)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = 100.0 * trainable / total if total else 0.0
        main_print(f"Trainable params: {trainable:,}/{total:,} ({ratio:.4f}%)")
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                main_print(f"  {name}: {tuple(parameter.shape)}")

    def _print_featurizer_tokens(self) -> None:
        for name in (
            "atom_mask_token",
            "bond_mask_token",
            "atom_pad_token",
            "bond_pad_token",
        ):
            if hasattr(self.featurizer, name):
                main_print(f"{name}: {getattr(self.featurizer, name)}")

    def load_configs(self) -> tuple[Any, Any, Any, Any, Any]:
        import tomllib

        with self.config_path.open("rb") as file:
            config = tomllib.load(file)

        sections = (
            "BaseConfig",
            "GraphormerTrainingConfig",
            "GraphormerConfig",
            "FeaturizerConfig",
            "DatasetConfig",
        )
        missing = [name for name in sections if name not in config]
        if missing:
            raise KeyError(f"Missing TOML sections: {missing}")

        return tuple(dict_to_namespace(config[name]) for name in sections)

    def load_model_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        device: torch.device | str = "cpu",
    ) -> dict:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_to_load = unwrap_model(model)

        if "model_state_dict" in checkpoint:
            model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=True)
            print(f"Loaded full model checkpoint from {checkpoint_path}")
            return checkpoint

        if "adapter_state_dict" in checkpoint:
            incompatible = model_to_load.load_state_dict(
                checkpoint["adapter_state_dict"], strict=False
            )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected adapter checkpoint keys: {incompatible.unexpected_keys}"
                )
            print(f"Loaded adapter checkpoint from {checkpoint_path}")
            print(
                f"Missing keys are expected for the frozen base encoder: "
                f"{len(incompatible.missing_keys)}"
            )
            return checkpoint

        raise KeyError(
            "Unsupported checkpoint format. Expected either 'model_state_dict' for a full "
            "checkpoint or 'adapter_state_dict' for an adapter checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )
    def plot_classification_curves(
        self,
        curve_data: dict[str, Any],
        output_dir: str | Path,
        prefix: str = "test",
    ) -> None:
        """
        Plot ROC curve, precision-recall curve, and confusion matrix.

        Parameters
        ----------
        curve_data
            Curve data returned by ClassificationEvaluator.compute_curve_data().

        output_dir
            Directory in which plots will be saved.

        prefix
            Filename prefix, such as ``"test"``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dpi = int(
            config_get(
                self.training_config,
                "plot_dpi",
                300,
            )
        )

        prefix = prefix.rstrip("_")

        if "roc" in curve_data:
            self._plot_roc_curve(
                roc_data=curve_data["roc"],
                output_path=output_dir / f"{prefix}_roc_curve.png",
                dpi=dpi,
            )

        if "pr" in curve_data:
            self._plot_precision_recall_curve(
                pr_data=curve_data["pr"],
                output_path=output_dir / f"{prefix}_pr_curve.png",
                dpi=dpi,
            )

        if "confusion_matrix" in curve_data:
            self._plot_confusion_matrix(
                matrix=curve_data["confusion_matrix"],
                output_path=output_dir / f"{prefix}_confusion_matrix.png",
                dpi=dpi,
                class_names=curve_data.get("class_names"),
            )


    @staticmethod
    def _plot_roc_curve(
        roc_data: dict[str, Any],
        output_path: str | Path,
        dpi: int = 300,
    ) -> None:
        """
        Plot a binary ROC curve.
        """
        fpr = np.asarray(roc_data["fpr"], dtype=np.float64)
        tpr = np.asarray(roc_data["tpr"], dtype=np.float64)

        if fpr.shape != tpr.shape:
            raise ValueError(
                "ROC fpr and tpr must have the same shape, "
                f"got {fpr.shape} and {tpr.shape}."
            )

        auc_value = roc_data.get("auc")

        plt.figure(figsize=(7, 6))

        if auc_value is None:
            plt.plot(
                fpr,
                tpr,
                linewidth=2,
                label="ROC curve",
            )
        else:
            plt.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"ROC curve (AUC = {float(auc_value):.4f})",
            )

        # Random-classifier baseline.
        plt.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1.5,
            label="Random classifier",
        )

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()


    @staticmethod
    def _plot_precision_recall_curve(
        pr_data: dict[str, Any],
        output_path: str | Path,
        dpi: int = 300,
    ) -> None:
        """
        Plot a binary precision-recall curve.
        """
        precision = np.asarray(
            pr_data["precision"],
            dtype=np.float64,
        )
        recall = np.asarray(
            pr_data["recall"],
            dtype=np.float64,
        )

        if precision.shape != recall.shape:
            raise ValueError(
                "PR precision and recall must have the same shape, "
                f"got {precision.shape} and {recall.shape}."
            )

        average_precision = pr_data.get("average_precision")
        positive_prevalence = pr_data.get("positive_prevalence")

        plt.figure(figsize=(7, 6))

        if average_precision is None:
            plt.plot(
                recall,
                precision,
                linewidth=2,
                label="Precision–recall curve",
            )
        else:
            plt.plot(
                recall,
                precision,
                linewidth=2,
                label=(
                    "Precision–recall curve "
                    f"(AP = {float(average_precision):.4f})"
                ),
            )

        # For PR curves, the random baseline is the positive-class prevalence.
        if positive_prevalence is not None:
            prevalence = float(positive_prevalence)

            plt.axhline(
                y=prevalence,
                linestyle="--",
                linewidth=1.5,
                label=f"Baseline = {prevalence:.4f}",
            )

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()


    @staticmethod
    def _plot_confusion_matrix(
        matrix: Any,
        output_path: str | Path,
        dpi: int = 300,
        class_names: list[str] | None = None,
    ) -> None:
        """
        Plot a confusion matrix.
        """
        matrix = np.asarray(matrix)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Confusion matrix must be a square 2D array, "
                f"got shape {matrix.shape}."
            )

        num_classes = matrix.shape[0]

        if class_names is None:
            class_names = [
                str(index)
                for index in range(num_classes)
            ]

        if len(class_names) != num_classes:
            raise ValueError(
                "class_names length must match confusion-matrix size: "
                f"{len(class_names)} versus {num_classes}."
            )

        figure_size = max(6.0, num_classes * 1.2)

        figure, axis = plt.subplots(
            figsize=(figure_size, figure_size),
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=class_names,
        )

        display.plot(
            ax=axis,
            values_format="d",
            colorbar=False,
        )

        axis.set_title("Confusion Matrix")
        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)