# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import csv
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.deep_learning.graphormer.config import (
    GraphormerFinetuneClassificationConfig,
    GraphormerFinetuneRegressionConfig,
    GraphormerFinetuneMultitaskConfig,
    calculate_task_weights,
)
from src.deep_learning.graphormer.evaluation.classification import ClassificationEvaluator
from src.deep_learning.graphormer.evaluation.regression import RegressionEvaluator
from src.deep_learning.graphormer.evaluation.multitask import MultiTaskEvaluator
from src.deep_learning.graphormer.models.graphormer_finetune_model import (
    GraphormerFineTuneClassificationModel,
    GraphormerFineTuneRegressionModel,
)
from src.deep_learning.graphormer.models.graphormer_multitask_model import (
    GraphormerMultiTaskModel,
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
    is_distributed,
    is_main_process,
    main_print,
    namespace_to_dict,
    save_json,
    set_seed,
    setup_distributed,
    step_scheduler,
    unwrap_model,
)
from src.deep_learning.plotter.training_plotter import ClassificationPlotter
from functools import partial
import numpy as np
import pandas as pd
from src.deep_learning.cross_validation import (
    CrossValidation,
    DeepLearningGridSearchCV,
)
import copy

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
    adaptor_parameters = []

    encoder_names = []
    lora_names = []
    head_names = []
    adaptor_names = []
    loss_names = []
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
            or "heads." in lower_name
        ):
            head_parameters.append(parameter)
            head_names.append(name)

        elif name.startswith("encoder."):
            encoder_parameters.append(parameter)
            encoder_names.append(name)
        elif "adaptors" in lower_name:
            adaptor_parameters.append(parameter)
            adaptor_names.append(name)
        elif "loss" in lower_name or "log_vars" in lower_name:
            loss_names.append(name)
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
                "lr": float(config_get(training_config, "encoder_learning_rate", base_lr)),
                "name": "encoder",
            }
        )

    if lora_parameters:
        parameter_groups.append(
            {
                "params": lora_parameters,
                "lr": float(config_get(training_config, "lora_learning_rate", base_lr)),
                "name": "lora",
            }
        )

    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": float(config_get(training_config, "head_learning_rate", base_lr)),
                "name": "head",
            }
        )

    print(f"Encoder: {len(encoder_names)} tensors")
    print(f"LoRA: {len(lora_names)} tensors")
    print(f"Head: {len(head_names)} tensors")

    return parameter_groups


def get_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {group.get("name", f"group_{index}"): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}



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
                    raise ValueError(f"cross_entropy requires num_classes >= 2, got num_classes={num_classes}.")
            else:
                raise ValueError(f"Unsupported classification loss_type: {loss_type!r}. "
                                 "Expected 'bce' or 'cross_entropy'.")

            main_print(f"Using evaluator task type: {evaluator_loss_type}")
            self.evaluator = ClassificationEvaluator(
                loss_type=evaluator_loss_type,
                num_classes=self.model_config.num_classes,
            )
        # To support multi-task learning, we can add a new task type "multitask" here.
        elif self.task == "multitask":
            self.model_config = update_dataclass_from_config(GraphormerFinetuneMultitaskConfig(), raw_model_config)

            if self.model_config.task_weights is None:
                dataset_df = pd.read_csv(self.dataset_config.dataset_path)
                task_names = list(getattr(self.dataset_config, "task_names", []) or [])
                if not task_names:
                    task_names = list(getattr(self.dataset_config, "target_column", []) or [])
                if not task_names:
                    raise ValueError("No task names available for automatic task weight calculation.")

                split_column = getattr(self.dataset_config, "split_column", None)
                if split_column and split_column in dataset_df.columns:
                    split_values = dataset_df[split_column].astype(str).str.strip().str.lower()
                    train_df = dataset_df[split_values.isin({"train", "training"})].copy()
                    if train_df.empty:
                        train_df = dataset_df.copy()
                else:
                    train_df = dataset_df.copy()

                weight_dict = calculate_task_weights(
                    train_df=train_df,
                    endpoints=task_names,
                    method=getattr(self.model_config, "task_weight_method", "sqrt_inverse"),
                    custom_task_weights=getattr(self.model_config, "task_weights", None),
                )
                self.model_config.task_weights = [weight_dict.get(name, 1.0) for name in task_names]

            self.model = GraphormerMultiTaskModel(cfg=self.model_config)
            self.evaluator = MultiTaskEvaluator()  # or a custom evaluator for multi-task
        else:
            raise ValueError(
                f"Unsupported task '{self.task}'. Expected regression, classification, or multitask."
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


        self.cross_validation = bool(
            config_get(self.training_config, "cross_validation", False)
        )
        self.grid_search = bool(
            config_get(self.training_config, "grid_search", False)
        )

        # Grid search already performs K-fold CV internally, so it takes
        # precedence over standalone cross-validation.
        if self.grid_search:
            self.cv_splits = int(config_get(self.training_config, "cv_splits", 5))
            main_print(
                f"Grid-search CV enabled with {self.cv_splits} folds per parameter configuration."
            )
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            self.train_sampler = None

        elif self.cross_validation:
            self.cv_splits = int(config_get(self.training_config, "cv_splits", 5))
            main_print(f"Cross-validation enabled with {self.cv_splits} splits.")
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            self.train_sampler = None

        else:
            main_print("Cross-validation disabled. Using a single train/validation split.")
            # -------------------------------------------------------------
            # Load dataset and create DataLoaders
            # -------------------------------------------------------------
            (
                self.train_loader,
                self.val_loader,
                self.test_loader,
                self.train_sampler,
            ) = self.load_dataset(
                dataset_config=self.dataset_config,
                featurizer=self.featurizer,
                cache_dir=self.cache_dir,
                device=self.device,
                training_config=self.training_config,
                distributed=self.distributed,
            )

    def train(self) -> None:
        """
        Entry point for Graphormer training.

        Modes
        -----
        grid_search=True
            Grid-search hyperparameters using K-fold CV.

        cross_validation=True
            Run K-fold CV once using the configured hyperparameters.

        otherwise
            Run the standard train/validation/test workflow.
        """
        try:
            if self.grid_search:
                self.train_grid_search()
            elif self.cross_validation:
                self.train_cross_validation()
            else:
                self.train_regular()
        finally:
            cleanup_distributed()

    def train_regular(self) -> None:
        """Run the standard single train/validation/test workflow."""
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

        # -------------------------------------------------------------
        # Assign trainable parameters to optimizer
        # -------------------------------------------------------------
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters remain.")

        parameter_groups = build_optimizer_parameter_groups(
            model=self.model,
            training_config=self.training_config,
        )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=float(self.training_config.weight_decay),
        )

        epochs = int(
            config_get(
                self.training_config,
                "num_epochs",
                config_get(self.training_config, "epochs", 100),
            )
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            training_config=self.training_config,
            total_epochs=epochs,
        )

        resolved_config = self.build_resolved_config(
            workdir=self.workdir,
            checkpoint_dir=self.checkpoint_dir,
        )
        if is_main_process():
            save_json(resolved_config, self.workdir / "config.json")

        # -------------------------------------------------------------
        # Select monitor metric and mode for early stopping
        # -------------------------------------------------------------
        monitor_mode = str(
            config_get(self.training_config, "monitor_mode", "min")
        ).lower()
        if monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'.")

        initial_best = (
            float("inf") if monitor_mode == "min" else float("-inf")
        )

        resume_state = {
            "start_epoch": 1,
            "best_metric": initial_best,
            "best_epoch": 0,
            "patience_counter": 0,
            "history": None,
        }

        if bool(config_get(self.training_config, "resume", False)):
            resume_path = config_get(
                self.training_config,
                "resume_checkpoint",
                None,
            )
            if not resume_path:
                raise ValueError(
                    "resume=True but resume_checkpoint is missing."
                )

            resume_state = self.load_checkpoint_for_resume(
                checkpoint_path=resume_path,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                fallback_best_metric=initial_best,
            )

        barrier()

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

        # -------------------------------------------------------------
        # Save final training history
        # -------------------------------------------------------------
        if is_main_process():
            torch.save(
                history,
                self.checkpoint_dir / "history.pt",
            )
            save_json(
                history,
                self.checkpoint_dir / "history.json",
            )
            print(f"Best model saved to {best_path}")

            if bool(
                config_get(
                    self.training_config,
                    "plot_training_history",
                    True,
                )
            ):
                self.plot_training_history(
                    history,
                    self.workdir / "plots",
                )

        # -------------------------------------------------------------
        # Evaluate on protected hold-out test set
        # -------------------------------------------------------------
        if (
            bool(
                config_get(
                    self.training_config,
                    "evaluate_test",
                    True,
                )
            )
            and self.test_loader is not None
        ):
            barrier()

            best_checkpoint = self.load_model_checkpoint(
                checkpoint_path=best_path,
                model=self.model,
                device=self.device,
            )

            barrier()

            test_metrics, curve_data = self.evaluate(
                model=self.model,
                loader=self.test_loader,
                device=self.device,
                prefix="test",
                return_curve_data=True,
            )

            if is_main_process():
                save_json(
                    {
                        "checkpoint": str(best_path),
                        "best_epoch": int(
                            best_checkpoint.get(
                                "best_epoch",
                                0,
                            )
                        ),
                        **test_metrics,
                    },
                    self.workdir / "test_metrics.json",
                )

                if curve_data is not None:
                    save_json(
                        curve_data,
                        self.workdir / "test_curve_data.json",
                    )

                    ClassificationPlotter(
                        config=self.training_config
                    ).plot_classification_curves(
                        data=curve_data,
                        output_dir=self.workdir / "plots",
                        prefix="test",
                    )

                print(
                    f"Best model loaded from {best_path}"
                )
                print(
                    "[Hold-out Test] "
                    + " ".join(
                        f"{name}={value:.4f}"
                        for name, value in test_metrics.items()
                    )
                )

    def build_fresh_model(
        self,
        model_config: Any | None = None,
    ) -> nn.Module:
        """
        Construct a fresh model.

        For CV/grid search, every fold must start from the same pretrained
        initialization. ``model_config`` can be supplied by grid search so
        model-level hyperparameters (for example LoRA rank or dropout) are
        applied without mutating ``self.model_config``.
        """
        cfg = self.model_config if model_config is None else model_config

        if self.task == "regression":
            return GraphormerFineTuneRegressionModel(cfg=cfg)

        if self.task == "classification":
            return GraphormerFineTuneClassificationModel(cfg=cfg)

        if self.task == "multitask":
            return GraphormerMultiTaskModel(cfg=cfg)

        raise ValueError(
            f"Unsupported task '{self.task}'. "
            "Expected regression, classification, or multitask."
        )

    def build_resolved_config(
        self,
        workdir: Path,
        checkpoint_dir: Path,
    ) -> dict:
        """Build a serializable resolved configuration."""
        return {
            "BaseConfig": config_to_dict(self.base_config),
            "GraphormerTrainingConfig": config_to_dict(
                self.training_config
            ),
            "GraphormerConfig": config_to_dict(
                self.model_config
            ),
            "DatasetConfig": config_to_dict(
                self.dataset_config
            ),
            "FeaturizerConfig": config_to_dict(
                self.featurizer_config
            ),
            "ResolvedConfig": {
                "workdir": str(workdir),
                "checkpoint_dir": str(checkpoint_dir),
                "cache_dir": str(self.cache_dir),
                "device": str(self.device),
                "distributed": self.distributed,
                "world_size": get_world_size(),
                "rank": get_rank(),
            },
        }

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

        # -------------------------------------------------------------
        # Setup checkpoint paths
        # -------------------------------------------------------------
        best_path = checkpoint_dir / "best_model.pt"
        last_path = checkpoint_dir / "last_model.pt"
        best_adapter_path = checkpoint_dir / "best_adapter.pt"
        last_adapter_path = checkpoint_dir / "last_adapter.pt"

        #-------------------------------------------------------------
        # Get monitor metric and mode
        #-------------------------------------------------------------
        monitor_metric = str(config_get(training_config, "monitor_metric", "val_loss"))
        monitor_mode = str(config_get(training_config, "monitor_mode", "min")).lower()
        if monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'.")

        learning_rates = get_learning_rates(optimizer)

        if history is None:
            history = {"epoch": [], "train_loss": []}
            for group_name in learning_rates:
                history[f"{group_name}_lr"] = []

        # -------------------------------------------------------------
        # Get training parameters
        # -------------------------------------------------------------
        epochs = int(config_get(training_config, "num_epochs", config_get(training_config, "epochs", 100)))
        early_stopping = bool(config_get(training_config, "early_stopping", True))
        patience = int(config_get(training_config, "early_stopping_patience", 10))
        gradient_clip = config_get(training_config, "gradient_clip_value", 1.0)
        scheduler_name = config_get(training_config, "scheduler", config_get(training_config, "schedular", None))
        save_adapter = bool(getattr(self.model_config, "use_lora", False))

        # -------------------------------------------------------------
        ## Training loop
        # -------------------------------------------------------------
        for epoch in range(start_epoch, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            running_loss = 0.0
            num_batches = 0
            progress = tqdm(train_loader, desc=f"Train epoch {epoch}/{epochs}", disable=disable_tqdm())

            for batch in progress:
                loss_value, task_losses = self.train_step(model, batch, optimizer, device, gradient_clip)
                running_loss += loss_value
                num_batches += 1
                if is_main_process() and not progress.disable:
                    postfix = {"loss": f"{loss_value:.4f}"}
                    if task_losses is not None:
                        postfix.update(
                            {
                                f"task_{i}_loss": f"{task_loss:.4f}"
                                for i, task_loss in enumerate(
                                    task_losses.detach().cpu().tolist()
                                )
                            }
                        )
                    progress.set_postfix(**postfix)
                    
            training_dtype = next(model.parameters()).dtype
            stats = torch.tensor(
                [running_loss, num_batches],
                dtype=training_dtype,
                device=device,
            )
            if is_distributed():
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
                raise FloatingPointError(f"{monitor_metric} is non-finite: {monitor_value}")

            # Step the scheduler based on the validation loss
            step_scheduler(scheduler=scheduler, scheduler_name=scheduler_name, val_loss=val_loss)

            # Get current learning rates for all parameter groups
            learning_rates = get_learning_rates(optimizer)

            # check if the current monitor value is better than the best metric so far
            improved = (monitor_value < best_metric if monitor_mode == "min" else monitor_value > best_metric)

            # If metric is improved, reset patience counter; otherwise, increment it. Save checkpoints accordingly.
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

                # Save training history to CSV for easier inspection
                self.append_history_csv(
                    checkpoint_dir / "history.csv",
                    epoch,
                    avg_train_loss,
                    learning_rates,
                    val_metrics,
                )

                metric_text = " ".join(f"{name}={float(value):.4f}" for name, value in val_metrics.items())
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

                # Save adapter checkpoint if applicable
                if save_adapter:
                    self.save_adapter_checkpoint(last_adapter_path, model, full_config)
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
            if is_distributed():
                dist.broadcast(state, src=0)
            best_metric = float(state[0].item())
            best_epoch = int(state[1].item())
            patience_counter = int(state[2].item())

            stop = torch.zeros((), dtype=torch.int32, device=device)
            if is_main_process() and early_stopping and patience_counter >= patience:
                stop.fill_(1)
            if is_distributed():
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
    ) -> tuple[float, torch.Tensor | None]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = self.forward_batch(model, move_batch_to_device(batch, device))

        loss = self.extract_loss(outputs)
        task_losses = None
        if self.task == "multitask":
            task_losses = self.extract_task_losses(outputs)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        if gradient_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(gradient_clip_value),
            )
        optimizer.step()
        return float(loss.detach().item()), task_losses.detach() if task_losses is not None else None

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
        if is_distributed():
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)

        average_loss = float((loss_stats[0] / loss_stats[1].clamp_min(1)).item())

        predictions = self.gather_variable_tensors(local_predictions).cpu()
        targets = self.gather_variable_tensors(local_targets).cpu()

        if self.task == "multitask":
            task_names = self.dataset_config.task_names
            metrics = self.evaluator.compute(predictions=predictions, targets=targets, loss=average_loss, task_names=task_names, prefix=prefix)
        else:
            metrics = self.evaluator.compute(predictions=predictions, targets=targets, loss=average_loss, prefix=prefix)

        metrics = {name: float(value) for name, value in metrics.items()}

        if not return_curve_data:
            return metrics

        if not isinstance(self.evaluator, ClassificationEvaluator):
            return metrics, {}

        curve_data = self.evaluator.compute_curve_data(
            predictions=predictions,
            targets=targets,
        )

        return metrics, curve_data

    def train_grid_search(self) -> None:
        """
        Tune hyperparameters with K-fold cross-validation.

        The protected test set is never used for selecting hyperparameters.
        Each parameter configuration is evaluated on the SAME CV folds.
        Every fold starts from a fresh pretrained model and uses the existing
        early-stopping implementation in ``run_training``.
        """
        if bool(config_get(self.training_config, "resume", False)):
            raise ValueError(
                "resume=True is not supported during grid-search CV. "
                "Every fold/configuration must start from a fresh model."
            )

        # -------------------------------------------------------------
        # Load development pool + protected test set
        # -------------------------------------------------------------
        cv_dataset, test_dataset = self.load_cv_dataset(
            dataset_config=self.dataset_config,
            featurizer=self.featurizer,
            cache_dir=self.cache_dir,
        )

        # The callback used by DeepLearningGridSearchCV needs access to
        # the same CV pool. Keep it only for the duration of grid search.
        self._grid_search_cv_dataset = cv_dataset

        n_splits = int(
            config_get(
                self.training_config,
                "cv_splits",
                config_get(self.training_config, "cv_folds", 5),
            )
        )
        cv_seed = int(
            config_get(
                self.training_config,
                "cv_seed",
                config_get(self.training_config, "seed", 42),
            )
        )

        monitor_metric = str(
            config_get(
                self.training_config,
                "grid_search_monitor_metric",
                config_get(self.training_config, "monitor_metric", "val_rmse"),
            )
        )
        monitor_mode = str(
            config_get(
                self.training_config,
                "grid_search_monitor_mode",
                config_get(self.training_config, "monitor_mode", "min"),
            )
        ).lower()

        if monitor_mode not in {"min", "max"}:
            raise ValueError(
                "grid_search_monitor_mode/monitor_mode must be 'min' or 'max'."
            )

        # -------------------------------------------------------------
        # Read the parameter grid from GraphormerTrainingConfig
        # -------------------------------------------------------------
        raw_param_grid = config_get(
            self.training_config,
            "grid_search_param_grid",
            None,
        )

        if raw_param_grid is None:
            raise ValueError(
                "grid_search=True but grid_search_param_grid is missing. "
                "Example TOML:\n"
                'grid_search_param_grid = { learning_rate = [5e-5, 1e-4], '
                'lora_r = [4, 8], head_dropout = [0.1, 0.2] }'
            )

        param_grid = config_to_dict(raw_param_grid)

        if not isinstance(param_grid, dict) or not param_grid:
            raise TypeError(
                "grid_search_param_grid must resolve to a non-empty mapping."
            )

        for name, values in param_grid.items():
            if not isinstance(values, list) or not values:
                raise TypeError(
                    f"Grid-search parameter '{name}' must contain a non-empty list; "
                    f"got {type(values).__name__}: {values!r}"
                )

        grid_root = self.workdir / "grid_search"

        if is_main_process():
            grid_root.mkdir(parents=True, exist_ok=True)
            save_json(
                {
                    "param_grid": param_grid,
                    "n_splits": n_splits,
                    "cv_seed": cv_seed,
                    "monitor_metric": monitor_metric,
                    "monitor_mode": monitor_mode,
                    "cv_pool_size": len(cv_dataset),
                    "protected_test_size": (
                        len(test_dataset) if test_dataset is not None else 0
                    ),
                },
                grid_root / "grid_search_config.json",
            )

        barrier()

        grid_search = DeepLearningGridSearchCV(
            param_grid=param_grid,
            n_splits=n_splits,
            cv_seed=cv_seed,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            output_dir=grid_root,
        )

        grid_search.fit(
            cv_dataset=cv_dataset,
            cv_class=CrossValidation,
            run_single_fold=self.run_single_cv_fold,
        )

        if is_main_process():
            print("\n[Grid-search CV complete]")
            print(f"Best {monitor_metric}: {grid_search.best_score_}")
            print(f"Best parameters: {grid_search.best_params_}")

            save_json(
                {
                    "best_score": grid_search.best_score_,
                    "best_params": grid_search.best_params_,
                    "monitor_metric": monitor_metric,
                    "monitor_mode": monitor_mode,
                },
                grid_root / "selected_hyperparameters.json",
            )

            if (
                bool(config_get(self.training_config, "evaluate_test", True))
                and test_dataset is not None
            ):
                print(
                    f"The protected test set contains {len(test_dataset)} samples "
                    "and was NOT used during grid search. "
                    "Fix the selected hyperparameters first, then train/evaluate "
                    "the final model separately."
                )

        barrier()

        del self._grid_search_cv_dataset

    def train_cross_validation(self) -> None:
        """
        Run K-fold cross-validation.

        The original train and validation splits are combined into one
        model-development pool. The protected test split remains outside CV.

        Each fold:
            1. builds fold-specific train/validation loaders,
            2. constructs a fresh model,
            3. trains with the existing early-stopping logic,
            4. restores the best checkpoint,
            5. evaluates the fold validation subset.

        The protected test set is not evaluated automatically after CV.
        A final model should first be retrained on the complete CV pool
        (or the fold models should be ensembled).
        """
        # -------------------------------------------------------------
        # CV does not support regular-training resume semantics.
        # -------------------------------------------------------------
        if bool(config_get(self.training_config, "resume", False)):
            raise ValueError(
                "resume=True is not supported for cross-validation. "
                "Resume individual fold checkpoints explicitly if needed."
            )

        # -------------------------------------------------------------
        # Load development pool + protected test dataset
        # -------------------------------------------------------------
        cv_dataset, test_dataset = self.load_cv_dataset(
            dataset_config=self.dataset_config,
            featurizer=self.featurizer,
            cache_dir=self.cache_dir,
        )

        n_splits = int(
            config_get(
                self.training_config,
                "cv_splits",
                config_get(
                    self.training_config,
                    "cv_folds",
                    getattr(self, "cv_splits", 5),
                ),
            )
        )
        cv_seed = int(
            config_get(
                self.training_config,
                "cv_seed",
                config_get(self.training_config, "seed", 42),
            )
        )

        cv = CrossValidation(
            dataset=cv_dataset,
            n_splits=n_splits,
            seed=cv_seed,
        )

        cv_root = self.workdir / "cross_validation"
        if is_main_process():
            cv_root.mkdir(parents=True, exist_ok=True)
        barrier()

        fold_results: list[dict] = []
        best_epochs: list[int] = []

        # =============================================================
        # CV fold loop
        # =============================================================
        for fold, train_idx, val_idx in cv.split_indices():
            if is_main_process():
                print(
                    f"\n{'=' * 70}\n"
                    f"Cross-validation fold {fold}/{n_splits}\n"
                    f"Train size: {len(train_idx)}\n"
                    f"Validation size: {len(val_idx)}\n"
                    f"{'=' * 70}",
                    flush=True,
                )

            # ---------------------------------------------------------
            # Fold-specific random seed
            # ---------------------------------------------------------
            fold_seed = cv_seed + fold - 1
            set_seed(fold_seed + get_rank())

            # ---------------------------------------------------------
            # Build fold-specific DataLoaders
            # ---------------------------------------------------------
            (
                train_loader,
                val_loader,
                train_sampler,
            ) = self.build_cv_fold_loaders(
                cv_dataset=cv_dataset,
                train_indices=train_idx,
                val_indices=val_idx,
                dataset_config=self.dataset_config,
                training_config=self.training_config,
                device=self.device,
                distributed=self.distributed,
            )

            # ---------------------------------------------------------
            # Fresh model for this fold
            # ---------------------------------------------------------
            model = self.build_fresh_model()
            model.to(self.device)

            if self.distributed:
                if (
                    self.device.type != "cuda"
                    or self.device.index is None
                ):
                    raise RuntimeError(
                        "CUDA DDP requires a CUDA device index."
                    )

                model = DDP(
                    model,
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                    find_unused_parameters=False,
                )

            # ---------------------------------------------------------
            # Verify trainable parameters
            # ---------------------------------------------------------
            trainable = [
                p
                for p in model.parameters()
                if p.requires_grad
            ]
            if not trainable:
                raise RuntimeError(
                    f"No trainable parameters remain in fold {fold}."
                )

            # ---------------------------------------------------------
            # Build a NEW optimizer and scheduler for this fold
            # ---------------------------------------------------------
            parameter_groups = build_optimizer_parameter_groups(
                model=model,
                training_config=self.training_config,
            )

            optimizer = torch.optim.AdamW(
                parameter_groups,
                weight_decay=float(
                    self.training_config.weight_decay
                ),
            )

            epochs = int(
                config_get(
                    self.training_config,
                    "num_epochs",
                    config_get(
                        self.training_config,
                        "epochs",
                        100,
                    ),
                )
            )

            scheduler = build_scheduler(
                optimizer=optimizer,
                training_config=self.training_config,
                total_epochs=epochs,
            )

            # ---------------------------------------------------------
            # Fold-specific output directories
            # ---------------------------------------------------------
            fold_workdir = cv_root / f"fold_{fold}"
            fold_checkpoint_dir = (
                fold_workdir / "checkpoints"
            )

            if is_main_process():
                fold_checkpoint_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
            barrier()

            # ---------------------------------------------------------
            # Save fold configuration
            # ---------------------------------------------------------
            resolved_config = self.build_resolved_config(
                workdir=fold_workdir,
                checkpoint_dir=fold_checkpoint_dir,
            )
            resolved_config["CrossValidation"] = {
                "enabled": True,
                "fold": fold,
                "n_splits": n_splits,
                "cv_seed": cv_seed,
                "fold_seed": fold_seed,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "cv_pool_size": len(cv_dataset),
                "protected_test_size": (
                    len(test_dataset)
                    if test_dataset is not None
                    else 0
                ),
            }

            if is_main_process():
                save_json(
                    resolved_config,
                    fold_workdir / "config.json",
                )

            # ---------------------------------------------------------
            # Early-stopping initial state
            # ---------------------------------------------------------
            monitor_mode = str(
                config_get(
                    self.training_config,
                    "monitor_mode",
                    "min",
                )
            ).lower()

            if monitor_mode not in {"min", "max"}:
                raise ValueError(
                    "monitor_mode must be 'min' or 'max'."
                )

            initial_best = (
                float("inf")
                if monitor_mode == "min"
                else float("-inf")
            )

            # ---------------------------------------------------------
            # Train this fold using existing run_training()
            # ---------------------------------------------------------
            history, best_path = self.run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                train_sampler=train_sampler,
                optimizer=optimizer,
                scheduler=scheduler,
                training_config=self.training_config,
                checkpoint_dir=fold_checkpoint_dir,
                device=self.device,
                full_config=resolved_config,
                start_epoch=1,
                best_metric=initial_best,
                best_epoch=0,
                patience_counter=0,
                history=None,
            )

            # ---------------------------------------------------------
            # Restore best fold checkpoint
            # ---------------------------------------------------------
            barrier()

            best_checkpoint = self.load_model_checkpoint(
                checkpoint_path=best_path,
                model=model,
                device=self.device,
            )

            barrier()

            # ---------------------------------------------------------
            # Evaluate best model on this fold's validation subset
            # ---------------------------------------------------------
            val_metrics, curve_data = self.evaluate(
                model=model,
                loader=val_loader,
                device=self.device,
                prefix="val",
                return_curve_data=True,
            )

            if is_main_process():
                best_epoch = int(
                    best_checkpoint.get(
                        "best_epoch",
                        best_checkpoint.get("epoch", 0),
                    )
                )

                best_epochs.append(best_epoch)

                fold_result = {
                    "fold": fold,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "best_epoch": best_epoch,
                    "best_checkpoint": str(best_path),
                    **val_metrics,
                }
                fold_results.append(fold_result)

                save_json(
                    fold_result,
                    fold_workdir / "fold_metrics.json",
                )
                torch.save(
                    history,
                    fold_workdir / "history.pt",
                )
                save_json(
                    history,
                    fold_workdir / "history.json",
                )

                if curve_data is not None:
                    save_json(
                        curve_data,
                        fold_workdir / "val_curve_data.json",
                    )

                    if isinstance(
                        self.evaluator,
                        ClassificationEvaluator,
                    ):
                        ClassificationPlotter(
                            config=self.training_config
                        ).plot_classification_curves(
                            data=curve_data,
                            output_dir=fold_workdir / "plots",
                            prefix="val",
                        )

                if bool(
                    config_get(
                        self.training_config,
                        "plot_training_history",
                        True,
                    )
                ):
                    self.plot_training_history(
                        history,
                        fold_workdir / "plots",
                    )

                print(
                    f"[CV Fold {fold}] "
                    + " ".join(
                        f"{name}={value:.4f}"
                        for name, value in val_metrics.items()
                    ),
                    flush=True,
                )

            barrier()

            # ---------------------------------------------------------
            # Release fold resources before starting the next fold
            # ---------------------------------------------------------
            del model
            del optimizer
            del scheduler
            del train_loader
            del val_loader
            del train_sampler

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            barrier()

        # =============================================================
        # Summarize CV results
        # =============================================================
        if is_main_process():
            cv_summary = self.summarize_cv_results(
                fold_results
            )
            cv_summary["n_splits"] = n_splits
            cv_summary["cv_seed"] = cv_seed
            cv_summary["cv_pool_size"] = len(cv_dataset)
            cv_summary["best_epochs"] = best_epochs

            if best_epochs:
                cv_summary["mean_best_epoch"] = float(
                    np.mean(best_epochs)
                )
                cv_summary["median_best_epoch"] = int(
                    np.median(best_epochs)
                )
                cv_summary["std_best_epoch"] = float(
                    np.std(best_epochs, ddof=1)
                    if len(best_epochs) > 1
                    else 0.0
                )

            save_json(
                {
                    "fold_results": fold_results,
                    "summary": cv_summary,
                },
                cv_root / "cv_summary.json",
            )

            pd.DataFrame(fold_results).to_csv(
                cv_root / "cv_fold_results.csv",
                index=False,
            )

            print("\n[Cross-validation summary]")
            for name, value in cv_summary.items():
                print(f"{name}: {value}")

        barrier()

        # -------------------------------------------------------------
        # Keep the protected test set untouched.
        # -------------------------------------------------------------
        if (
            bool(
                config_get(
                    self.training_config,
                    "evaluate_test",
                    True,
                )
            )
            and test_dataset is not None
            and is_main_process()
        ):
            print(
                "\nCross-validation complete. "
                f"The protected test set contains {len(test_dataset)} samples "
                "and has NOT been evaluated. "
                "Retrain a final model on the complete CV pool "
                "(for example using the median best epoch) or ensemble "
                "the fold models before final test evaluation."
            )

    def summarize_cv_results(self, fold_results: list[dict]) -> dict:

        if not fold_results:
            return {}

        excluded = {"fold", "best_epoch"}
        metric_names = [key for key in fold_results[0] if key not in excluded]
        summary = {}

        for metric in metric_names:

            values = np.asarray([result[metric] for result in fold_results if metric in result], dtype=float)
            if values.size == 0:
                continue

            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        return summary
    
    @staticmethod
    def gather_variable_tensors(tensor: torch.Tensor) -> torch.Tensor:
        if not is_distributed():
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

        return torch.cat([item[:size.item()] for item, size in zip(gathered, sizes)], dim=0)

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

    @staticmethod
    def extract_task_losses(outputs: Any) -> torch.Tensor:
        task_losses = None
        if torch.is_tensor(outputs):
            task_losses = outputs
        elif isinstance(outputs, Mapping) and "task_losses" in outputs:
            task_losses = outputs["task_losses"]
        elif hasattr(outputs, "task_losses"):
            task_losses = outputs.task_losses
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 1:
            task_losses = outputs[1]
        else:
            raise TypeError("Could not extract task losses from model output.")

        if not torch.is_tensor(task_losses):
            raise ValueError("Extracted task losses must be a tensor.")
        return task_losses
    
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
        # Load a checkpoint for resuming training. This can be either a full training checkpoint or an adapter-only checkpoint.
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Unwrap the model in case it's wrapped in DDP or other wrappers
        model_to_load = unwrap_model(model)

        # ============================================================
        # Full training checkpoint
        # ============================================================
        if "model_state_dict" in checkpoint:
            model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=True)

            if (optimizer is not None and checkpoint.get("optimizer_state_dict")):
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                move_optimizer_state_to_device(optimizer, torch.device(device))

            if (scheduler is not None and checkpoint.get("scheduler_state_dict")):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

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
            incompatible = model_to_load.load_state_dict(checkpoint["adapter_state_dict"], strict=False)

            if incompatible.unexpected_keys:
                raise RuntimeError(f"Unexpected adapter checkpoint keys: {incompatible.unexpected_keys}")

            print(f"Loaded adapter checkpoint from {checkpoint_path}")
            print(f"Missing keys are expected for the frozen base encoder: {len(incompatible.missing_keys)}")

            # Adapter loading is not a strict resume:
            # optimizer/scheduler/history start fresh.
            return {
                "start_epoch": int(checkpoint.get("epoch", 0)) + 1,
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
                raise ValueError(f"History columns changed. Existing={header}; current={fieldnames}")

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

    def load_manifest(self, dataset_config, featurizer, cache_dir):
        """
        Load the dataset manifest, generating and caching it if necessary.
        This function ensures that only the main process generates the manifest, while other processes wait for it to be available.
        """
        if is_main_process():
            manifest = featurize_and_cache_dataset(dataset_config=dataset_config, featurizer=featurizer, cache_dir=cache_dir)
        else:
            manifest = None
        barrier()

        if manifest is None:
            manifest = featurize_and_cache_dataset(dataset_config=dataset_config, featurizer=featurizer, cache_dir=cache_dir)

        dataset_config.split_task_counts = manifest.get("split_task_counts")
        dataset_config.task_names = manifest.get("task_names", getattr(dataset_config, "task_names", None))
        return manifest

    def load_dataset(
        self,
        dataset_config: Any,
        featurizer: GraphormerFeaturizer,
        cache_dir: Path,
        device: torch.device,
        training_config: Any,
        distributed: bool,
    ) -> tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
        manifest = self.load_manifest(dataset_config, featurizer, cache_dir)
        # Add cached statistics to the dataset configuration.
        dataset_config.split_task_counts = manifest.get("split_task_counts")
        dataset_config.task_names = manifest.get("task_names", getattr(dataset_config, "task_names", None))

        train_manifest = manifest["train"]
        val_manifest = manifest["val"]

        train_dataset = GraphormerMoleculeDataset(train_manifest)
        val_dataset = GraphormerMoleculeDataset(val_manifest)

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

    def load_cv_dataset(self, dataset_config, featurizer, cache_dir):
        """
        Load the cross-validation dataset and the test dataset.
        Returns a tuple of (cv_dataset, test_dataset).
        """
        manifest = self.load_manifest(dataset_config, featurizer, cache_dir)

        train_dataset = GraphormerMoleculeDataset(manifest["train"])
        val_manifest = manifest.get("val")

        # Build a concatenated dataset for cross-validation if a validation set is available.
        if val_manifest:
            val_dataset = GraphormerMoleculeDataset(val_manifest)
            cv_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset,])
        else:
            cv_dataset = train_dataset

        test_dataset = None
        test_manifest = manifest.get("test")

        if (bool(config_get( self.training_config, "evaluate_test", True)) and test_manifest):
            test_dataset = GraphormerMoleculeDataset(test_manifest)

        return cv_dataset, test_dataset
    def build_cv_fold_loaders(
        self,
        cv_dataset,
        train_indices,
        val_indices,
        dataset_config,
        training_config,
        device,
        distributed,
    ):
        """
        Build train/validation DataLoaders for one cross-validation fold.

        Parameters
        ----------
        cv_dataset
            Full dataset used for cross-validation.

        train_indices
            Training indices for the current fold.

        val_indices
            Validation indices for the current fold.

        dataset_config
            Dataset configuration.

        training_config
            Training configuration.

        device
            Current training device.

        distributed
            Whether DDP is enabled.

        Returns
        -------
        train_loader
            DataLoader for the current fold's training subset.

        val_loader
            DataLoader for the current fold's validation subset.

        train_sampler
            DistributedSampler for training, or None.
        """

        # ============================================================
        # Create fold subsets
        # ============================================================

        train_dataset = Subset(
            cv_dataset,
            train_indices.tolist(),
        )

        val_dataset = Subset(
            cv_dataset,
            val_indices.tolist(),
        )

        # ============================================================
        # Distributed samplers
        # ============================================================

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

        else:
            train_sampler = None
            val_sampler = None
            train_shuffle = True

        # ============================================================
        # Graphormer collator
        # ============================================================

        collate_fn = partial(
            graphormer_collate_fn,
            max_nodes=int(
                config_get(
                    dataset_config,
                    "max_nodes",
                    128,
                )
            ),
            multi_hop_max_dist=int(
                config_get(
                    dataset_config,
                    "multi_hop_max_dist",
                    5,
                )
            ),
            spatial_pos_max=int(
                config_get(
                    dataset_config,
                    "spatial_pos_max",
                    1024,
                )
            ),
        )

        # ============================================================
        # Common DataLoader arguments
        # ============================================================

        common = {
            "num_workers": int(
                config_get(
                    training_config,
                    "num_workers",
                    0,
                )
            ),
            "pin_memory": device.type == "cuda",
            "collate_fn": collate_fn,
        }

        # ============================================================
        # Train DataLoader
        # ============================================================

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(training_config.batch_size),
            shuffle=train_shuffle,
            sampler=train_sampler,
            drop_last=True,
            **common,
        )

        # ============================================================
        # Validation DataLoader
        # ============================================================

        val_loader = DataLoader(
            val_dataset,
            batch_size=int(
                config_get(
                    training_config,
                    "eval_batch_size",
                    training_config.batch_size,
                )
            ),
            shuffle=False,
            sampler=val_sampler,
            drop_last=False,
            **common,
        )

        return (
            train_loader,
            val_loader,
            train_sampler,
        )
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

    def run_single_cv_fold(
        self,
        *,
        fold,
        train_idx,
        val_idx,
        params,
        config_index,
        config_output_dir,
    ):
        """
        Train/evaluate one fold for one grid-search parameter configuration.

        This method is called on every DDP rank. It returns the same metric
        dictionary on every rank so the grid-search controller can make the
        same decision everywhere.
        """
        if not hasattr(self, "_grid_search_cv_dataset"):
            raise RuntimeError(
                "Grid-search CV dataset is not initialized. "
                "Call train_grid_search() rather than run_single_cv_fold() directly."
            )

        cv_dataset = self._grid_search_cv_dataset

        # ============================================================
        # Clone configs and apply this parameter configuration
        # ============================================================
        model_config = copy.deepcopy(self.model_config)
        training_config = copy.deepcopy(self.training_config)

        for name, value in params.items():
            if hasattr(training_config, name):
                setattr(training_config, name, value)
            elif hasattr(model_config, name):
                setattr(model_config, name, value)
            else:
                raise KeyError(
                    f"Unknown grid-search parameter '{name}'. "
                    "It must exist in GraphormerTrainingConfig or GraphormerConfig."
                )

        # Same fold seed for the same fold across all configurations.
        cv_seed = int(
            config_get(
                self.training_config,
                "cv_seed",
                config_get(self.training_config, "seed", 42),
            )
        )
        fold_seed = cv_seed + int(fold) - 1
        set_seed(fold_seed + get_rank())

        # ============================================================
        # Build fold DataLoaders
        # ============================================================
        (
            train_loader,
            val_loader,
            train_sampler,
        ) = self.build_cv_fold_loaders(
            cv_dataset=cv_dataset,
            train_indices=train_idx,
            val_indices=val_idx,
            dataset_config=self.dataset_config,
            training_config=training_config,
            device=self.device,
            distributed=self.distributed,
        )

        # ============================================================
        # Fresh model for this parameter configuration + fold
        # ============================================================
        model = self.build_fresh_model(model_config=model_config)
        model.to(self.device)

        if self.distributed:
            if self.device.type != "cuda" or self.device.index is None:
                raise RuntimeError(
                    "CUDA DDP requires a CUDA device index."
                )

            model = DDP(
                model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=False,
            )

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                f"No trainable parameters remain for config {config_index}, fold {fold}."
            )

        # ============================================================
        # Fresh optimizer + scheduler
        # ============================================================
        parameter_groups = build_optimizer_parameter_groups(
            model=model,
            training_config=training_config,
        )

        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=float(training_config.weight_decay),
        )

        epochs = int(
            config_get(
                training_config,
                "num_epochs",
                config_get(training_config, "epochs", 100),
            )
        )

        scheduler = build_scheduler(
            optimizer=optimizer,
            training_config=training_config,
            total_epochs=epochs,
        )

        # ============================================================
        # Output directories
        # ============================================================
        fold_output_dir = Path(config_output_dir) / f"fold_{fold}"
        checkpoint_dir = fold_output_dir / "checkpoints"

        if is_main_process():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        barrier()

        resolved_config = self.build_resolved_config(
            workdir=fold_output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        resolved_config["GridSearch"] = {
            "config_index": int(config_index),
            "params": config_to_dict(params),
            "fold": int(fold),
            "fold_seed": int(fold_seed),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
        }
        resolved_config["GraphormerTrainingConfig"] = config_to_dict(
            training_config
        )
        resolved_config["GraphormerConfig"] = config_to_dict(model_config)

        if is_main_process():
            save_json(
                resolved_config,
                fold_output_dir / "config.json",
            )

        # ============================================================
        # Early stopping
        # ============================================================
        monitor_mode = str(
            config_get(training_config, "monitor_mode", "min")
        ).lower()

        if monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'.")

        initial_best = (
            float("inf")
            if monitor_mode == "min"
            else float("-inf")
        )

        # ============================================================
        # Train
        # ============================================================
        history, best_path = self.run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=training_config,
            checkpoint_dir=checkpoint_dir,
            device=self.device,
            full_config=resolved_config,
            start_epoch=1,
            best_metric=initial_best,
            best_epoch=0,
            patience_counter=0,
            history=None,
        )

        barrier()

        # ============================================================
        # Restore best fold checkpoint and evaluate
        # ============================================================
        checkpoint = self.load_model_checkpoint(
            checkpoint_path=best_path,
            model=model,
            device=self.device,
        )

        barrier()

        val_metrics = self.evaluate(
            model=model,
            loader=val_loader,
            device=self.device,
            prefix="val",
        )

        # evaluate() gathers across ranks, so all ranks have the same metrics.
        result = {
            "fold": int(fold),
            "best_epoch": int(
                checkpoint.get(
                    "best_epoch",
                    checkpoint.get("epoch", 0),
                )
            ),
            **{name: float(value) for name, value in val_metrics.items()},
        }

        if is_main_process():
            save_json(
                result,
                fold_output_dir / "fold_metrics.json",
            )
            torch.save(
                history,
                fold_output_dir / "history.pt",
            )
            save_json(
                history,
                fold_output_dir / "history.json",
            )

            if bool(
                config_get(
                    training_config,
                    "plot_training_history",
                    True,
                )
            ):
                self.plot_training_history(
                    history,
                    fold_output_dir / "plots",
                )

            print(
                f"[Grid config {config_index} | fold {fold}] "
                + " ".join(
                    f"{name}={float(value):.4f}"
                    for name, value in val_metrics.items()
                ),
                flush=True,
            )

        barrier()

        # ============================================================
        # Cleanup
        # ============================================================
        del model
        del optimizer
        del scheduler
        del train_loader
        del val_loader
        del train_sampler

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        barrier()

        return result

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
            incompatible = model_to_load.load_state_dict(checkpoint["adapter_state_dict"], strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError( f"Unexpected adapter checkpoint keys: {incompatible.unexpected_keys}")
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
    
