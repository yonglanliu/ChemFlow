"""ChemFlow-style fine-tuning for the pretrained CheMeleon architecture.

The model construction follows the official CheMeleon reference workflow:
ChemProp's ``SimpleMoleculeMolGraphFeaturizer``, the published pretrained
``BondMessagePassing`` state, and mean molecular aggregation. ChemFlow owns
the surrounding dataset validation, splitting, evaluation, and artifacts.
"""

from __future__ import annotations

import json
import math
import shutil
import tomllib
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from chemflow.deep_learning.chemeleon.applicability import (
    APPLICABILITY_VERSION,
    DEFAULT_CALIBRATION_CONFIDENCE,
    DEFAULT_LOCAL_MAX_SAMPLES,
    DEFAULT_LOCAL_MIN_EMBEDDING_SIMILARITY,
    DEFAULT_LOCAL_MIN_FP_SIMILARITY,
    DEFAULT_LOCAL_MIN_SAMPLES,
    DEFAULT_OOD_CONFIDENCE,
    DEFAULT_SIMILARITY_BITS,
    DEFAULT_SIMILARITY_RADIUS,
    MAX_EMBEDDING_DIMENSIONS,
    applicability_diagnostics,
    extract_model_embeddings,
    fit_embedding_projection,
    fit_multitask_validation_calibration,
    fit_ood_calibration,
    packed_morgan_fingerprints,
)
from chemflow.deep_learning.chemeleon.pretrained import (
    CHEMELEON_WEIGHTS_SHA256,
    CHEMELEON_WEIGHTS_URL,
    ensure_pretrained_weights,
)
from chemflow.deep_learning.chemeleon.mixed_predictor import (
    MixedTaskFFN,
    MixedTaskMetric,
)
from chemflow.deep_learning.utils.train_utils import save_json, set_seed
from chemflow.deep_learning.task_weighting import (
    resolve_task_loss_weights,
    resolve_task_types,
)


@dataclass
class BaseConfig:
    workdir: str
    task: str = "regression"
    seed: int = 42


@dataclass
class DatasetConfig:
    dataset_path: str
    test_dataset_path: str | None = None
    smiles_column: str = "SMILES"
    target_column: str | list[str] = "target"
    task_types: list[str] | dict[str, str] | None = None
    split_column: str | None = None
    split_type: str = "random"
    val_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_workers: int = 0
    num_epochs: int = 30
    accelerator: str = "auto"
    devices: int | str = 1
    strategy: str = "auto"
    num_nodes: int = 1
    sync_batchnorm: bool = False
    precision: str = "32-true"
    deterministic: bool = True
    gradient_clip_value: float = 1.0
    early_stopping: bool = True
    early_stopping_patience: int = 10
    class_balance: bool = False
    evaluate_test: bool = True
    plot_training_history: bool = True
    inspect_task_metrics: bool = True
    tensorboard: bool = True
    tensorboard_dir: str = "tensorboard"
    task_loss_weighting: str = "uniform"
    task_loss_weights: list[float] | dict[str, float] | None = None
    resume: bool = False
    resume_checkpoint: str | None = None
    verbose: bool = True


@dataclass
class ModelConfig:
    pretrained_path: str | None = None
    pretrained_url: str = CHEMELEON_WEIGHTS_URL
    pretrained_sha256: str | None = CHEMELEON_WEIGHTS_SHA256
    transfer_checkpoint: str | None = None
    transfer_encoder_only: bool = True
    freeze_epochs: int = 0
    dropout: float = 0.0
    ffn_hidden_dim: int = 256
    ffn_num_layers: int = 1
    warmup_epochs: int = 2
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4


@dataclass
class CheMeleonRunConfig:
    base: BaseConfig
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig


class FreezeMessagePassingCallback(Callback):
    """Freeze the pretrained encoder for an initial number of epochs."""

    def __init__(self, freeze_epochs: int):
        super().__init__()
        self.freeze_epochs = int(freeze_epochs)

    def on_fit_start(self, trainer, pl_module) -> None:
        for parameter in pl_module.message_passing.parameters():
            parameter.requires_grad = False

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch == self.freeze_epochs:
            for parameter in pl_module.message_passing.parameters():
                parameter.requires_grad = True


class TrainingTaskMetricsCallback(Callback):
    """Record per-task training and validation metrics after every epoch."""

    def __init__(
        self,
        *,
        train_loader,
        val_loader,
        train_records: list[dict],
        val_records: list[dict],
        target_names: list[str],
        task: str,
        task_types: list[str],
        output_path: Path,
    ) -> None:
        super().__init__()
        self.split_loaders = {
            "train": train_loader,
            "validation": val_loader,
        }
        self.split_targets = {
            "train": np.asarray(
                [item["targets"] for item in train_records], dtype=np.float32
            ),
            "validation": np.asarray(
                [item["targets"] for item in val_records], dtype=np.float32
            ),
        }
        self.target_names = list(target_names)
        self.task = str(task)
        self.task_types = list(task_types)
        self.output_path = Path(output_path)
        self.rows: list[dict[str, Any]] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        epoch = int(trainer.current_epoch) + 1
        metric_names = (
            ("loss", "mae", "rmse", "median_ae", "r2", "pearson", "spearman")
            if self.task == "regression"
            else (
                "loss", "accuracy", "balanced_accuracy", "precision", "recall",
                "f1", "mcc", "roc_auc", "pr_auc",
            )
        )
        if self.task == "mixed":
            metric_names = (
                "loss", "mae", "rmse", "median_ae", "r2", "pearson", "spearman",
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
                "roc_auc", "pr_auc",
            )
        logged: dict[str, float] = {}
        # Replace an epoch if Lightning reruns it after resuming.
        self.rows = [row for row in self.rows if int(row["epoch"]) != epoch]
        for split_name, loader in self.split_loaders.items():
            targets = self.split_targets[split_name]
            predictions = _predict_checkpoint_predictions(pl_module, loader)
            metrics = _evaluate(
                self.task, predictions, targets, self.target_names, self.task_types
            )
            log_prefix = "train" if split_name == "train" else "val"
            print(f"{split_name.title()} metrics by task — epoch {epoch}:")
            for task_index, target_name in enumerate(self.target_names):
                row: dict[str, Any] = {
                    "epoch": epoch,
                    "split": split_name,
                    "task": target_name,
                    "num_labeled": int(
                        np.isfinite(targets[:, task_index]).sum()
                    ),
                }
                for metric_name in metric_names:
                    value = float(
                        metrics.get(
                            f"test_{target_name}_{metric_name}", float("nan")
                        )
                    )
                    row[metric_name] = value
                    logged[f"{log_prefix}_{target_name}_{metric_name}"] = value
                self.rows.append(row)
                shown = ", ".join(
                    f"{name}={row[name]:.4f}"
                    for name in metric_names
                    if name != "loss" and math.isfinite(float(row[name]))
                )
                print(f"  {target_name} (n={row['num_labeled']}): {shown}")

            overall: dict[str, Any] = {
                "epoch": epoch,
                "split": split_name,
                "task": "overall_macro",
                "num_labeled": int(np.isfinite(targets).sum()),
            }
            for metric_name in metric_names:
                value = float(metrics.get(f"test_{metric_name}", float("nan")))
                overall[metric_name] = value
                logged[f"{log_prefix}_macro_{metric_name}"] = value
            self.rows.append(overall)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).sort_values(
            ["epoch", "split", "task"]
        ).to_csv(self.output_path, index=False)
        if trainer.loggers:
            logged["epoch"] = float(trainer.current_epoch)
            for logger in trainer.loggers:
                logger.log_metrics(logged, step=trainer.global_step)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"TOML section [{name}] must be a table.")
    return value


def load_config(path: str | Path) -> CheMeleonRunConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    missing = [name for name in ("BaseConfig", "DatasetConfig") if name not in raw]
    if missing:
        raise KeyError(f"Missing TOML sections: {missing}")

    base = BaseConfig(**_section(raw, "BaseConfig"))
    dataset = DatasetConfig(**_section(raw, "DatasetConfig"))
    training = TrainingConfig(**_section(raw, "CheMeleonTrainingConfig"))
    model = ModelConfig(**_section(raw, "CheMeleonConfig"))

    base.task = str(base.task).strip().lower()
    resolve_task_types(base.task, _target_columns(dataset), dataset.task_types)

    dataset.split_type = str(dataset.split_type).strip().lower()
    allowed_splits = {
        "random",
        "random_with_repeated_smiles",
        "scaffold_balanced",
        "kennard_stone",
        "kmeans",
        "predefined",
    }
    if dataset.split_type not in allowed_splits:
        raise ValueError(
            f"Unsupported split_type {dataset.split_type!r}; "
            f"expected one of {sorted(allowed_splits)}."
        )
    if dataset.split_type == "predefined" and not dataset.split_column:
        raise ValueError(
            "split_type='predefined' requires DatasetConfig.split_column."
        )

    if not 0.0 < float(dataset.val_fraction) < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if not 0.0 <= float(dataset.test_fraction) < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if dataset.test_dataset_path and float(dataset.test_fraction) != 0.0:
        raise ValueError(
            "test_fraction must be 0 when test_dataset_path is provided."
        )
    if dataset.val_fraction + dataset.test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")
    if int(training.num_epochs) < 1:
        raise ValueError("num_epochs must be at least 1.")
    if int(training.num_nodes) < 1:
        raise ValueError("num_nodes must be at least 1.")
    accelerator = str(training.accelerator).strip().lower()
    strategy = str(training.strategy).strip().lower()
    if accelerator == "mps" and (
        strategy.startswith("ddp")
        or training.devices not in (1, "1", "auto")
    ):
        raise ValueError(
            "MPS supports only single-device training. Use accelerator='gpu' "
            "with multiple CUDA devices for DDP."
        )
    if int(model.freeze_epochs) < 0:
        raise ValueError("freeze_epochs cannot be negative.")
    if model.transfer_checkpoint and not bool(model.transfer_encoder_only):
        raise ValueError(
            "Only encoder-only CheMeleon transfer is supported. "
            "Set transfer_encoder_only=true."
        )
    if (
        base.task in {"classification", "mixed"}
        and bool(training.class_balance)
        and len(_target_columns(dataset)) > 1
    ):
        raise ValueError(
            "class_balance is supported only for single-task classification. "
            "Set class_balance=false for multitask classification."
        )

    return CheMeleonRunConfig(base, dataset, training, model)


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_resume_checkpoint(
    training: TrainingConfig,
    checkpoint_dir: Path,
) -> Path | None:
    """Resolve and validate a full-state Lightning resume checkpoint."""
    configured = str(training.resume_checkpoint or "").strip()
    if not bool(training.resume) and not configured:
        return None

    path = (
        _resolved_path(configured)
        if configured
        else (checkpoint_dir / "last.ckpt").resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"CheMeleon resume checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not checkpoint.get("optimizer_states"):
        raise ValueError(
            f"CheMeleon checkpoint {path} contains model weights only and cannot "
            "resume optimizer-level training. Use it as transfer_checkpoint, or "
            "resume from a last.ckpt created by the full-state checkpoint format."
        )
    return path


def _complete_batch_size(dataset_size: int, requested_batch_size: int) -> int:
    """Choose a ChemProp batch size that cannot trigger its drop-last guard."""
    if dataset_size < 1:
        raise ValueError("Cannot build a dataloader for an empty dataset.")
    batch_size = min(int(requested_batch_size), int(dataset_size))
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    while batch_size > 1 and dataset_size % batch_size == 1:
        batch_size -= 1
    return batch_size


def _target_columns(config: DatasetConfig) -> list[str]:
    value = config.target_column
    columns = [value] if isinstance(value, str) else list(value)
    columns = [str(column).strip() for column in columns]
    if not columns or any(not column for column in columns):
        raise ValueError("target_column must contain at least one non-empty column name.")
    if len(set(columns)) != len(columns):
        raise ValueError(f"target_column contains duplicate names: {columns}")
    return columns


def _valid_records(frame: pd.DataFrame, config: DatasetConfig) -> tuple[list[dict], list[dict]]:
    target_columns = _target_columns(config)
    smiles_column = config.smiles_column
    if (
        smiles_column == "SMILES"
        and smiles_column not in frame.columns
        and "smiles" in frame.columns
    ):
        # Backward compatibility for data_splits.csv written by older releases.
        smiles_column = "smiles"
    required = [smiles_column, *target_columns]
    if config.split_column:
        required.append(config.split_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}")

    records: list[dict] = []
    rejected: list[dict] = []
    for row_index, row in frame.iterrows():
        smiles = str(row[smiles_column]).strip()
        targets = pd.to_numeric(row[target_columns], errors="coerce").to_numpy(
            dtype=np.float32
        )
        reason = None
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            reason = "invalid_smiles"
        elif not np.isfinite(targets).any():
            reason = "all_targets_missing_or_non_numeric"

        if reason:
            rejected.append(
                {"original_index": row_index, "smiles": smiles, "reason": reason}
            )
            continue

        records.append(
            {
                "original_index": row_index,
                "smiles": smiles,
                "targets": targets,
                "mol": molecule,
                "split": (
                    str(row[config.split_column]).strip().lower()
                    if config.split_column
                    else None
                ),
            }
        )

    if not records:
        raise ValueError("No valid labeled molecules remain after dataset validation.")
    return records, rejected


def _split_records(
    records: list[dict],
    config: DatasetConfig,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if config.split_column:
        train = [item for item in records if item["split"] in {"train", "training"}]
        val = [item for item in records if item["split"] in {"val", "valid", "validation"}]
        test = [item for item in records if item["split"] in {"test", "testing"}]
        unknown = [
            item for item in records
            if item["split"] not in {
                "train", "training", "val", "valid", "validation", "test", "testing"
            }
        ]
        if unknown:
            values = sorted({item["split"] for item in unknown})
            raise ValueError(f"Unknown values in split column: {values}")
        if not train or not val:
            raise ValueError("Explicit splits must contain non-empty train and validation sets.")
        return train, val, test

    sizes = (
        1.0 - float(config.val_fraction) - float(config.test_fraction),
        float(config.val_fraction),
        float(config.test_fraction),
    )
    molecules = _molecules_for_split(records, config.split_type)
    train_indices, val_indices, test_indices = data.make_split_indices(
        molecules,
        split=config.split_type,
        sizes=sizes,
        seed=int(seed),
    )
    train_idx = list(train_indices[0])
    val_idx = list(val_indices[0])
    test_idx = list(test_indices[0])
    return (
        [records[int(index)] for index in train_idx],
        [records[int(index)] for index in val_idx],
        [records[int(index)] for index in test_idx],
    )


def _molecules_for_split(records: list[dict], split_type: str) -> list[Chem.Mol]:
    """Return molecules safe for the requested Chemprop split operation.

    Some RDKit releases can retain invalid double-bond stereo metadata after
    constructing a Bemis-Murcko scaffold. Canonicalizing that scaffold then
    raises ``Pre-condition Violation: bad bond stereo``. Scaffold splitting
    does not use stereochemistry, so remove it from molecule copies supplied
    only to the splitter. The original molecules and SMILES remain unchanged
    for featurization, training, prediction, and saved split artifacts.
    """
    if str(split_type).strip().lower() != "scaffold_balanced":
        return [item["mol"] for item in records]

    split_molecules: list[Chem.Mol] = []
    for item in records:
        molecule = Chem.Mol(item["mol"])
        Chem.RemoveStereochemistry(molecule)
        split_molecules.append(molecule)
    return split_molecules


def _datapoints(records: list[dict]) -> list[data.MoleculeDatapoint]:
    return [
        data.MoleculeDatapoint.from_smi(
            item["smiles"],
            np.asarray(item["targets"], dtype=np.float32),
        )
        for item in records
    ]


def _save_split_manifest(
    workdir: Path,
    train: list[dict],
    val: list[dict],
    test: list[dict],
    target_columns: list[str],
) -> None:
    rows = []
    for split_name, records in (("train", train), ("validation", val), ("test", test)):
        rows.extend(
            {
                "original_index": item["original_index"],
                "SMILES": item["smiles"],
                **{
                    target_name: float(item["targets"][task_index])
                    for task_index, target_name in enumerate(target_columns)
                },
                "split": split_name,
            }
            for item in records
        )
    pd.DataFrame(rows).to_csv(workdir / "data_splits.csv", index=False)


def _make_model(
    config: CheMeleonRunConfig,
    pretrained_path: Path,
    train_dataset: data.MoleculeDataset,
    val_dataset: data.MoleculeDataset,
    task_loss_weights: np.ndarray | None = None,
) -> models.MPNN:
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    if not {"hyper_parameters", "state_dict"}.issubset(checkpoint):
        raise ValueError("Invalid CheMeleon checkpoint: expected hyper_parameters and state_dict.")

    parameters = dict(checkpoint["hyper_parameters"])
    parameters["dropout"] = float(config.model.dropout)
    message_passing = nn.BondMessagePassing(**parameters)
    message_passing.load_state_dict(checkpoint["state_dict"], strict=True)
    if config.model.transfer_checkpoint:
        _load_transfer_encoder(
            message_passing,
            _resolved_path(config.model.transfer_checkpoint),
        )

    target_columns = _target_columns(config.dataset)
    task_types = resolve_task_types(
        config.base.task, target_columns, config.dataset.task_types
    )
    n_tasks = len(target_columns)
    output_transform = None
    if config.base.task == "regression":
        scaler = train_dataset.normalize_targets()
        val_dataset.normalize_targets(scaler)
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        predictor = nn.RegressionFFN(
            n_tasks=n_tasks,
            input_dim=message_passing.output_dim,
            hidden_dim=int(config.model.ffn_hidden_dim),
            n_layers=int(config.model.ffn_num_layers),
            dropout=float(config.model.dropout),
            output_transform=output_transform,
            task_weights=torch.as_tensor(task_loss_weights, dtype=torch.float32),
        )
        metrics = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()]
    elif config.base.task == "classification":
        predictor = nn.BinaryClassificationFFN(
            n_tasks=n_tasks,
            input_dim=message_passing.output_dim,
            hidden_dim=int(config.model.ffn_hidden_dim),
            n_layers=int(config.model.ffn_num_layers),
            dropout=float(config.model.dropout),
            task_weights=torch.as_tensor(task_loss_weights, dtype=torch.float32),
        )
        metrics = [
            nn.metrics.BinaryAUROC(),
            nn.metrics.BinaryAUPRC(),
            nn.metrics.BinaryAccuracy(),
            nn.metrics.BinaryF1Score(),
        ]
    else:
        scaler = StandardScaler().fit(train_dataset.Y)
        for index, task_type in enumerate(task_types):
            if task_type == "classification":
                scaler.mean_[index] = 0.0
                scaler.scale_[index] = 1.0
                scaler.var_[index] = 1.0
        train_dataset.normalize_targets(scaler)
        val_dataset.normalize_targets(scaler)
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        predictor = MixedTaskFFN(
            task_types=task_types,
            input_dim=message_passing.output_dim,
            hidden_dim=int(config.model.ffn_hidden_dim),
            n_layers=int(config.model.ffn_num_layers),
            dropout=float(config.model.dropout),
            output_transform=output_transform,
            task_weights=torch.as_tensor(task_loss_weights, dtype=torch.float32),
        )
        metrics = [MixedTaskMetric(task_types)]

    return models.MPNN(
        message_passing=message_passing,
        agg=nn.MeanAggregation(),
        predictor=predictor,
        batch_norm=False,
        metrics=metrics,
        warmup_epochs=int(config.model.warmup_epochs),
        init_lr=float(config.model.init_lr),
        max_lr=float(config.model.max_lr),
        final_lr=float(config.model.final_lr),
    )


def _load_transfer_encoder(
    message_passing: torch.nn.Module,
    checkpoint_path: str | Path,
) -> int:
    """Load only a fine-tuned checkpoint's message-passing encoder.

    Fine-tuned Lightning checkpoints contain task-specific predictor tensors,
    target scaling statistics, and criterion weights. Those tensors cannot be
    reused when the destination has a different number of tasks, so this
    loader explicitly selects only ``message_passing.*`` parameters.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Transfer checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Invalid transfer checkpoint {path}: expected a Lightning state_dict."
        )

    encoder_state: dict[str, torch.Tensor] = {}
    prefixes = (
        "message_passing.",
        "model.message_passing.",
        "module.message_passing.",
    )
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                encoder_state[key.removeprefix(prefix)] = value
                break

    if not encoder_state:
        raise ValueError(
            f"Transfer checkpoint {path} contains no message_passing encoder weights."
        )

    expected_state = message_passing.state_dict()
    missing = sorted(set(expected_state) - set(encoder_state))
    unexpected = sorted(set(encoder_state) - set(expected_state))
    incompatible_shapes = sorted(
        key
        for key in set(expected_state) & set(encoder_state)
        if tuple(expected_state[key].shape) != tuple(encoder_state[key].shape)
    )
    if missing or unexpected or incompatible_shapes:
        raise ValueError(
            "Transfer encoder is incompatible with the configured CheMeleon "
            f"architecture. Missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={incompatible_shapes}."
        )

    message_passing.load_state_dict(encoder_state, strict=True)
    print("CheMeleon transfer-weight audit:")
    print(f"  Checkpoint: {path}")
    print(f"  File size: {path.stat().st_size:,} bytes")
    print("  Loading mode: encoder only")
    print(f"  Encoder tensors loaded: {len(encoder_state):,}")
    print("  Encoder compatibility: passed (strict=True)")
    print("  Task-specific prediction head: not loaded")
    return len(encoder_state)


def _evaluate(
    task: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: list[str] | None = None,
    task_types: list[str] | None = None,
) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if target_names is None:
        target_names = [f"task_{index}" for index in range(predictions.shape[1])]
    task_types = task_types or [task] * len(target_names)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {predictions.shape} and {targets.shape}."
        )
    if predictions.shape[1] != len(target_names):
        raise ValueError(
            "Prediction task count does not match target names: "
            f"{predictions.shape[1]} and {len(target_names)}."
        )

    metrics: dict[str, float] = {}
    aggregate: dict[str, list[float]] = {}
    all_valid = np.isfinite(predictions) & np.isfinite(targets)

    for task_index, target_name in enumerate(target_names):
        valid = all_valid[:, task_index]
        if not valid.any():
            continue
        prediction_tensor = torch.as_tensor(
            predictions[valid, task_index], dtype=torch.float32
        ).reshape(-1, 1)
        target_tensor = torch.as_tensor(
            targets[valid, task_index], dtype=torch.float32
        ).reshape(-1, 1)

        if task_types[task_index] == "regression":
            loss = functional.mse_loss(prediction_tensor, target_tensor).item()
            prediction_values = prediction_tensor.numpy().reshape(-1)
            target_values = target_tensor.numpy().reshape(-1)
            pearson = (
                float(pearsonr(target_values, prediction_values).statistic)
                if target_values.size > 1
                and np.unique(target_values).size > 1
                and np.unique(prediction_values).size > 1
                else float("nan")
            )
            spearman = (
                float(spearmanr(target_values, prediction_values).statistic)
                if target_values.size > 1
                and np.unique(target_values).size > 1
                and np.unique(prediction_values).size > 1
                else float("nan")
            )
            task_metrics = {
                "test_loss": float(loss),
                "test_mae": float(mean_absolute_error(target_values, prediction_values)),
                "test_rmse": float(mean_squared_error(target_values, prediction_values) ** 0.5),
                "test_median_ae": float(median_absolute_error(target_values, prediction_values)),
                "test_r2": (
                    float(r2_score(target_values, prediction_values))
                    if target_values.size > 1 else float("nan")
                ),
                "test_pearson": pearson,
                "test_spearman": spearman,
            }
        else:
            probabilities = prediction_tensor.clamp(1e-7, 1.0 - 1e-7)
            loss = functional.binary_cross_entropy(probabilities, target_tensor).item()
            probability_values = probabilities.numpy().reshape(-1)
            target_values = target_tensor.numpy().reshape(-1).astype(np.int64)
            predicted_values = (probability_values >= 0.5).astype(np.int64)
            has_both_classes = np.unique(target_values).size > 1
            task_metrics = {
                "test_loss": float(loss),
                "test_accuracy": float(accuracy_score(target_values, predicted_values)),
                "test_balanced_accuracy": float(
                    balanced_accuracy_score(target_values, predicted_values)
                ),
                "test_precision": float(
                    precision_score(target_values, predicted_values, zero_division=0)
                ),
                "test_recall": float(
                    recall_score(target_values, predicted_values, zero_division=0)
                ),
                "test_f1": float(f1_score(target_values, predicted_values, zero_division=0)),
                "test_mcc": float(matthews_corrcoef(target_values, predicted_values)),
                "test_roc_auc": (
                    float(roc_auc_score(target_values, probability_values))
                    if has_both_classes else float("nan")
                ),
                "test_pr_auc": (
                    float(average_precision_score(target_values, probability_values))
                    if has_both_classes else float("nan")
                ),
            }

        for metric_name, value in task_metrics.items():
            suffix = metric_name.removeprefix("test_")
            metrics[f"test_{target_name}_{suffix}"] = float(value)
            if math.isfinite(float(value)):
                aggregate.setdefault(suffix, []).append(float(value))

    for metric_name, values in aggregate.items():
        metrics[f"test_{metric_name}"] = float(np.mean(values))
    return metrics


def _predict_checkpoint(
    model: models.MPNN,
    dataloader,
    checkpoint_path: str | Path,
) -> np.ndarray:
    """Predict a complete hold-out set on one rank in dataloader order."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    return _predict_checkpoint_predictions(model, dataloader)


def _predict_checkpoint_predictions(model: models.MPNN, dataloader) -> np.ndarray:
    """Predict using the weights currently loaded into ``model``."""

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            batch = model.transfer_batch_to_device(batch, device, 0)
            prediction = model.predict_step(batch, batch_index)
            prediction = prediction.detach().cpu()
            if prediction.ndim == 1:
                prediction = prediction.reshape(-1, 1)
            batches.append(prediction)

    if not batches:
        return np.empty((0, model.predictor.n_tasks), dtype=np.float32)
    return torch.cat(batches, dim=0).numpy()


def _validate_task_coverage(
    records: list[dict],
    target_columns: list[str],
    split_name: str,
) -> None:
    target_matrix = np.asarray([item["targets"] for item in records], dtype=np.float32)
    missing = [
        target_name
        for task_index, target_name in enumerate(target_columns)
        if not np.isfinite(target_matrix[:, task_index]).any()
    ]
    if missing:
        raise ValueError(
            f"The {split_name} split has no finite labels for tasks: {missing}."
        )


def _embed_run_config(checkpoint_path: str | Path, resolved: dict[str, Any]) -> None:
    path = Path(checkpoint_path)
    if not path.is_file():
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["chemflow_config"] = resolved
    torch.save(checkpoint, path)


def _embed_applicability_payload(
    checkpoint_path: str | Path,
    payload: dict[str, Any],
) -> None:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["chemflow_applicability"] = payload
    torch.save(checkpoint, path)


def _reference_partition(
    records: list[dict],
    embeddings: np.ndarray,
    target_names: list[str],
    *,
    predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    if len(records) != len(embeddings):
        raise RuntimeError(
            "Reference embedding count does not match the corresponding dataset."
        )
    partition: dict[str, Any] = {
        "original_index": [item["original_index"] for item in records],
        "smiles": [item["smiles"] for item in records],
        "target_names": list(target_names),
        "targets": torch.as_tensor(
            np.asarray([item["targets"] for item in records], dtype=np.float32)
        ),
        "embeddings": torch.as_tensor(embeddings),
    }
    if predictions is not None:
        partition["predictions"] = torch.as_tensor(
            np.asarray(predictions, dtype=np.float32)
        )
    return partition


def _build_applicability_payload(
    *,
    model: models.MPNN,
    train_loader,
    val_loader,
    train_records: list[dict],
    val_records: list[dict],
    target_names: list[str],
    task: str,
    task_types: list[str],
    seed: int,
) -> dict[str, Any]:
    train_embeddings = extract_model_embeddings(model, train_loader)
    val_embeddings = extract_model_embeddings(model, val_loader)
    val_predictions = _predict_checkpoint_predictions(model, val_loader)
    projection, train_projected, val_projected = fit_embedding_projection(
        train_embeddings,
        val_embeddings,
        dimensions=MAX_EMBEDDING_DIMENSIONS,
        seed=int(seed),
    )
    val_targets = np.asarray(
        [item["targets"] for item in val_records], dtype=np.float32
    )
    calibration = fit_multitask_validation_calibration(
        task_types,
        val_targets,
        val_predictions,
        target_names,
        confidence=DEFAULT_CALIBRATION_CONFIDENCE,
    )
    payload = {
        "version": APPLICABILITY_VERSION,
        "task": task,
        "task_types": dict(zip(target_names, task_types)),
        "embedding_space": "fine_tuned_mean_mpn_pca",
        "projection": projection,
        "similarity": {
            "method": "morgan_tanimoto",
            "radius": DEFAULT_SIMILARITY_RADIUS,
            "bits": DEFAULT_SIMILARITY_BITS,
        },
        "training": _reference_partition(
            train_records, train_projected, target_names
        ),
        "validation": _reference_partition(
            val_records,
            val_projected,
            target_names,
            predictions=val_predictions,
        ),
        "calibration": calibration,
    }
    payload["training"]["fingerprints"] = torch.from_numpy(
        packed_morgan_fingerprints(
            payload["training"]["smiles"],
            radius=DEFAULT_SIMILARITY_RADIUS,
            bits=DEFAULT_SIMILARITY_BITS,
        )
    )
    payload["training_by_task"] = {}
    target_matrix = np.asarray(
        [item["targets"] for item in train_records], dtype=np.float32
    )
    for task_index, target_name in enumerate(target_names):
        selected = np.flatnonzero(np.isfinite(target_matrix[:, task_index]))
        task_records = [
            {
                **train_records[int(index)],
                "targets": np.asarray(
                    [train_records[int(index)]["targets"][task_index]],
                    dtype=np.float32,
                ),
            }
            for index in selected
        ]
        task_partition = _reference_partition(
            task_records,
            train_projected[selected],
            [target_name],
        )
        task_partition["fingerprints"] = torch.from_numpy(
            packed_morgan_fingerprints(
                task_partition["smiles"],
                radius=DEFAULT_SIMILARITY_RADIUS,
                bits=DEFAULT_SIMILARITY_BITS,
            )
        )
        payload["training_by_task"][target_name] = task_partition
    payload["ood_calibration"] = {}
    payload["validation_by_task"] = {}
    payload["local_calibration"] = {
        "method": "fingerprint_embedding_nearest_validation_residuals",
        "min_samples": DEFAULT_LOCAL_MIN_SAMPLES,
        "max_samples": DEFAULT_LOCAL_MAX_SAMPLES,
        "min_fp_similarity": DEFAULT_LOCAL_MIN_FP_SIMILARITY,
        "min_embedding_similarity": DEFAULT_LOCAL_MIN_EMBEDDING_SIMILARITY,
    }
    validation_matrix = np.asarray(
        [item["targets"] for item in val_records], dtype=np.float32
    )
    for task_index, target_name in enumerate(target_names):
        selected = np.flatnonzero(np.isfinite(validation_matrix[:, task_index]))
        task_validation_records = [
            {
                **val_records[int(index)],
                "targets": np.asarray(
                    [val_records[int(index)]["targets"][task_index]],
                    dtype=np.float32,
                ),
            }
            for index in selected
        ]
        validation_partition = _reference_partition(
            task_validation_records,
            val_projected[selected],
            [target_name],
            predictions=val_predictions[selected, task_index : task_index + 1],
        )
        validation_partition["fingerprints"] = torch.from_numpy(
            packed_morgan_fingerprints(
                validation_partition["smiles"],
                radius=DEFAULT_SIMILARITY_RADIUS,
                bits=DEFAULT_SIMILARITY_BITS,
            )
        )
        payload["validation_by_task"][target_name] = validation_partition
        validation_smiles = [val_records[int(index)]["smiles"] for index in selected]
        diagnostics = applicability_diagnostics(
            validation_smiles,
            val_embeddings[selected],
            list(range(len(selected))),
            payload,
            reference_task=target_name,
        )
        payload["ood_calibration"][target_name] = fit_ood_calibration(
            diagnostics["train_max_tanimoto"],
            diagnostics["train_embedding_cosine_distance"],
            confidence=DEFAULT_OOD_CONFIDENCE,
        )
    return payload


def _save_task_metrics_csv(
    path: str | Path,
    task: str,
    metrics: dict[str, float],
    target_names: list[str],
    targets: np.ndarray,
    task_types: list[str] | None = None,
) -> None:
    metric_names = (
        ["loss", "mae", "rmse", "median_ae", "r2", "pearson", "spearman"]
        if task == "regression"
        else [
            "loss",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "roc_auc",
            "pr_auc",
        ]
    )
    if task == "mixed":
        metric_names = [
            "loss", "mae", "rmse", "median_ae", "r2", "pearson", "spearman",
            "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
            "roc_auc", "pr_auc",
        ]
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    rows: list[dict[str, Any]] = []
    for task_index, target_name in enumerate(target_names):
        row: dict[str, Any] = {
            "split": "test",
            "task": target_name,
            "num_labeled": int(np.isfinite(targets[:, task_index]).sum()),
        }
        row.update(
            {
                metric_name: metrics.get(
                    f"test_{target_name}_{metric_name}", float("nan")
                )
                for metric_name in metric_names
            }
        )
        rows.append(row)

    overall: dict[str, Any] = {
        "split": "test",
        "task": "overall_macro",
        "num_labeled": int(np.isfinite(targets).sum()),
    }
    overall.update(
        {
            metric_name: metrics.get(f"test_{metric_name}", float("nan"))
            for metric_name in metric_names
        }
    )
    rows.append(overall)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _plot_history(metrics_path: Path, output_path: Path) -> None:
    if not metrics_path.is_file():
        return
    import matplotlib.pyplot as plt

    history = pd.read_csv(metrics_path)
    figure, axis = plt.subplots(figsize=(7, 5))
    plotted = False
    for column, label in (("train_loss", "Train"), ("val_loss", "Validation")):
        if column not in history:
            continue
        values = history.loc[history[column].notna(), ["epoch", column]]
        if values.empty:
            continue
        values = values.groupby("epoch", as_index=False)[column].last()
        axis.plot(values["epoch"], values[column], label=label)
        plotted = True
    if plotted:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("CheMeleon fine-tuning")
        axis.legend()
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300)
    plt.close(figure)


class CheMeleonTrainer:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {self.config_path}")
        self.config = load_config(self.config_path)
        self.workdir = _resolved_path(self.config.base.workdir)
        self.checkpoint_dir = self.workdir / "checkpoints"

    def train(self) -> dict[str, float] | None:
        set_seed(int(self.config.base.seed))
        pl.seed_everything(int(self.config.base.seed), workers=True)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = _resolved_path(self.config.dataset.dataset_path)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
        frame = pd.read_csv(dataset_path)
        records, rejected = _valid_records(frame, self.config.dataset)
        target_columns = _target_columns(self.config.dataset)
        task_types = resolve_task_types(
            self.config.base.task,
            target_columns,
            self.config.dataset.task_types,
        )

        train_records, val_records, test_records = _split_records(
            records,
            self.config.dataset,
            seed=int(self.config.base.seed),
        )

        if self.config.dataset.test_dataset_path:
            if test_records:
                raise ValueError(
                    "The training dataset contains test rows while "
                    "test_dataset_path is also configured. Use only one test source."
                )
            test_dataset_path = _resolved_path(
                self.config.dataset.test_dataset_path
            )
            if not test_dataset_path.is_file():
                raise FileNotFoundError(
                    f"Test dataset does not exist: {test_dataset_path}"
                )
            test_frame = pd.read_csv(test_dataset_path)
            test_config = replace(self.config.dataset, split_column=None)
            test_records, test_rejected = _valid_records(test_frame, test_config)
            for item in rejected:
                item["dataset"] = "training"
            for item in test_rejected:
                item["dataset"] = "external_test"
            rejected.extend(test_rejected)

        _validate_task_coverage(train_records, target_columns, "training")
        _validate_task_coverage(val_records, target_columns, "validation")
        train_target_matrix = np.asarray(
            [item["targets"] for item in train_records], dtype=np.float32
        )
        task_label_counts, task_loss_weights = resolve_task_loss_weights(
            train_target_matrix,
            target_columns,
            strategy=self.config.training.task_loss_weighting,
            configured=self.config.training.task_loss_weights,
        )
        print("CheMeleon task loss weighting:")
        for target_name, count, weight in zip(
            target_columns, task_label_counts, task_loss_weights
        ):
            print(f"  {target_name}: labels={int(count)}, weight={float(weight):.6f}")

        if "classification" in task_types:
            labels = {
                float(item["targets"][index])
                for item in (*records, *test_records)
                for index, task_type in enumerate(task_types)
                if task_type == "classification"
                and np.isfinite(item["targets"][index])
            }
            if not labels.issubset({0.0, 1.0}):
                raise ValueError(
                    "Binary classification targets must contain only 0 and 1; "
                    f"found {sorted(labels)}."
                )

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dataset = data.MoleculeDataset(_datapoints(train_records), featurizer)
        val_dataset = data.MoleculeDataset(_datapoints(val_records), featurizer)
        test_dataset = (
            data.MoleculeDataset(_datapoints(test_records), featurizer)
            if test_records
            else None
        )

        configured_path = self.config.model.pretrained_path
        pretrained_path = ensure_pretrained_weights(
            _resolved_path(configured_path) if configured_path else None,
            url=self.config.model.pretrained_url,
            sha256=self.config.model.pretrained_sha256,
        )
        pretrained_source = (
            "configured pretrained_path"
            if configured_path
            else "official CheMeleon default/cache"
        )
        checksum = self.config.model.pretrained_sha256
        print("CheMeleon pretrained-weight audit:")
        print(f"  Source: {pretrained_source}")
        print(f"  Resolved path: {pretrained_path}")
        print(f"  File size: {pretrained_path.stat().st_size:,} bytes")
        if checksum:
            print(f"  SHA-256 verification: passed ({checksum})")
        else:
            print("  SHA-256 verification: disabled by configuration")
        if not configured_path:
            print(f"  Official source URL: {self.config.model.pretrained_url}")

        model = _make_model(
            self.config,
            pretrained_path,
            train_dataset,
            val_dataset,
            task_loss_weights=task_loss_weights,
        )
        print("  Encoder state loading: passed (strict=True)")
        if self.config.model.transfer_checkpoint:
            print(
                "  Transfer encoder checkpoint: "
                f"{_resolved_path(self.config.model.transfer_checkpoint)}"
            )
        else:
            print("  Transfer encoder checkpoint: none")

        pretrained_audit = {
            "source": pretrained_source,
            "resolved_path": str(pretrained_path),
            "file_size_bytes": int(pretrained_path.stat().st_size),
            "source_url": (
                self.config.model.pretrained_url if not configured_path else None
            ),
            "expected_sha256": checksum,
            "sha256_verification": "passed" if checksum else "disabled",
            "encoder_state_loading": "passed_strict",
            "transfer_checkpoint": (
                str(_resolved_path(self.config.model.transfer_checkpoint))
                if self.config.model.transfer_checkpoint
                else None
            ),
            "transfer_loading": (
                "passed_strict_encoder_only"
                if self.config.model.transfer_checkpoint
                else "not_requested"
            ),
        }

        batch_size = int(self.config.training.batch_size)
        workers = int(self.config.training.num_workers)
        train_loader = data.build_dataloader(
            train_dataset,
            batch_size=batch_size,
            num_workers=workers,
            class_balance=(
                bool(self.config.training.class_balance)
                if self.config.base.task == "classification"
                else False
            ),
            seed=int(self.config.base.seed),
        )
        train_reference_loader = data.build_dataloader(
            train_dataset,
            batch_size=_complete_batch_size(len(train_dataset), batch_size),
            num_workers=workers,
            shuffle=False,
        )
        val_loader = data.build_dataloader(
            val_dataset,
            batch_size=_complete_batch_size(len(val_dataset), batch_size),
            num_workers=workers,
            shuffle=False,
        )
        test_loader = (
            data.build_dataloader(
                test_dataset,
                batch_size=_complete_batch_size(len(test_dataset), batch_size),
                num_workers=workers,
                shuffle=False,
            )
            if test_dataset is not None
            else None
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            # Full-state checkpoints preserve optimizer, scheduler, epoch,
            # callback, scaler, and loop state for exact Lightning resume.
            save_weights_only=False,
            auto_insert_metric_name=False,
        )
        callbacks: list[Callback] = [
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ]
        if bool(self.config.training.inspect_task_metrics):
            callbacks.append(
                TrainingTaskMetricsCallback(
                    train_loader=train_reference_loader,
                    val_loader=val_loader,
                    train_records=train_records,
                    val_records=val_records,
                    target_names=target_columns,
                    task=self.config.base.task,
                    task_types=task_types,
                    output_path=self.workdir / "training_task_metrics.csv",
                )
            )
        if int(self.config.model.freeze_epochs) > 0:
            callbacks.append(
                FreezeMessagePassingCallback(self.config.model.freeze_epochs)
            )
        if self.config.training.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=int(self.config.training.early_stopping_patience),
                )
            )

        loggers: list[Any] = [CSVLogger(save_dir=self.workdir, name="logs")]
        if bool(self.config.training.tensorboard):
            tensorboard_dir = self.workdir / self.config.training.tensorboard_dir
            loggers.append(
                TensorBoardLogger(
                    save_dir=tensorboard_dir.parent,
                    name=tensorboard_dir.name,
                    version="",
                )
            )
            print(f"TensorBoard logs: {tensorboard_dir}")
        trainer = pl.Trainer(
            default_root_dir=self.workdir,
            max_epochs=int(self.config.training.num_epochs),
            accelerator=self.config.training.accelerator,
            devices=self.config.training.devices,
            strategy=self.config.training.strategy,
            num_nodes=int(self.config.training.num_nodes),
            sync_batchnorm=bool(self.config.training.sync_batchnorm),
            precision=self.config.training.precision,
            deterministic=bool(self.config.training.deterministic),
            gradient_clip_val=float(self.config.training.gradient_clip_value),
            callbacks=callbacks,
            logger=loggers,
            enable_progress_bar=bool(self.config.training.verbose),
            enable_model_summary=bool(self.config.training.verbose),
            log_every_n_steps=1,
        )

        resolved = {
            "BaseConfig": asdict(self.config.base),
            "DatasetConfig": asdict(self.config.dataset),
            "CheMeleonTrainingConfig": asdict(self.config.training),
            "CheMeleonConfig": {
                **asdict(self.config.model),
                "resolved_pretrained_path": str(pretrained_path),
                "resolved_transfer_checkpoint": (
                    str(_resolved_path(self.config.model.transfer_checkpoint))
                    if self.config.model.transfer_checkpoint
                    else None
                ),
            },
            "pretrained_weight_audit": pretrained_audit,
            "dataset_summary": {
                "task_names": target_columns,
                "num_tasks": len(target_columns),
                "task_label_counts": {
                    name: int(count)
                    for name, count in zip(target_columns, task_label_counts)
                },
                "task_loss_weights": {
                    name: float(weight)
                    for name, weight in zip(target_columns, task_loss_weights)
                },
                "train_size": len(train_records),
                "validation_size": len(val_records),
                "test_size": len(test_records),
                "rejected_size": len(rejected),
            },
        }
        if trainer.is_global_zero:
            if rejected:
                pd.DataFrame(rejected).to_csv(
                    self.workdir / "rejected_rows.csv", index=False
                )
            _save_split_manifest(
                self.workdir,
                train_records,
                val_records,
                test_records,
                target_columns,
            )
            save_json(resolved, self.workdir / "config.json")

        resume_checkpoint = _resolve_resume_checkpoint(
            self.config.training,
            self.checkpoint_dir,
        )

        # Torch 2.12 deprecates direct construction of ``LeafSpec`` while
        # Lightning 2.6 still uses it internally. This narrow filter removes
        # only that upstream compatibility warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
                category=FutureWarning,
                module=r"lightning\.pytorch\.utilities\._pytree",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"The '.*_dataloader' does not have many workers.*",
                category=UserWarning,
                module=r"lightning\.pytorch\.trainer\.connectors\.data_connector",
            )
            trainer.fit(
                model,
                train_loader,
                val_loader,
                ckpt_path=(
                    str(resume_checkpoint)
                    if resume_checkpoint is not None
                    else None
                ),
            )

        # DDP workers must not race while copying logs, evaluating the full
        # hold-out set, or writing final artifacts. The global-zero worker
        # evaluates with the original sequential loader, preserving every row
        # and its order (distributed prediction samplers can pad or reorder).
        if not trainer.is_global_zero:
            return None

        csv_logger = loggers[0]
        metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
        if metrics_path.is_file():
            shutil.copy2(metrics_path, self.workdir / "training_history.csv")
            shutil.copy2(metrics_path, self.checkpoint_dir / "history.csv")
        if self.config.training.plot_training_history:
            _plot_history(metrics_path, self.workdir / "plots" / "training_history.png")

        best_path = checkpoint_callback.best_model_path
        if not best_path:
            raise RuntimeError("Training completed without producing a best checkpoint.")

        _embed_run_config(best_path, resolved)
        _embed_run_config(checkpoint_callback.last_model_path, resolved)

        best_checkpoint = torch.load(
            best_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(best_checkpoint["state_dict"], strict=True)
        applicability = _build_applicability_payload(
            model=model,
            train_loader=train_reference_loader,
            val_loader=val_loader,
            train_records=train_records,
            val_records=val_records,
            target_names=target_columns,
            task=self.config.base.task,
            task_types=task_types,
            seed=int(self.config.base.seed),
        )
        _embed_applicability_payload(best_path, applicability)

        run_summary = {
            "best_checkpoint": best_path,
            "best_val_loss": float(checkpoint_callback.best_model_score),
        }
        save_json(run_summary, self.workdir / "run_summary.json")

        if not self.config.training.evaluate_test or test_loader is None:
            return None

        # This checkpoint was produced by the current trusted training run and
        # contains ChemProp metric objects in addition to tensors.
        predictions = _predict_checkpoint(model, test_loader, best_path)
        targets = np.asarray(
            [item["targets"] for item in test_records],
            dtype=np.float64,
        )
        if predictions.shape[0] != targets.shape[0]:
            raise RuntimeError(
                "Prediction count does not match the hold-out test set: "
                f"{predictions.shape[0]} versus {targets.shape[0]}."
            )

        metrics = _evaluate(
            self.config.base.task,
            predictions,
            targets,
            target_columns,
            task_types,
        )
        save_json(
            {"checkpoint": best_path, **metrics},
            self.workdir / "test_metrics.json",
        )
        _save_task_metrics_csv(
            self.checkpoint_dir / "test_metrics.csv",
            self.config.base.task,
            metrics,
            target_columns,
            targets,
            task_types,
        )
        prediction_data: dict[str, Any] = {
            "original_index": [item["original_index"] for item in test_records],
            self.config.dataset.smiles_column: [item["smiles"] for item in test_records],
        }
        for task_index, target_name in enumerate(target_columns):
            prediction_data[f"true_{target_name}"] = targets[:, task_index]
            prediction_data[f"pred_{target_name}"] = predictions[:, task_index]
            if task_types[task_index] == "classification":
                prediction_data[f"class_{target_name}"] = (
                    predictions[:, task_index] >= 0.5
                ).astype(np.int64)
        prediction_frame = pd.DataFrame(prediction_data)
        prediction_frame.to_csv(self.workdir / "test_predictions.csv", index=False)

        metric_text = " ".join(
            f"{name}={value:.4f}"
            for name, value in metrics.items()
            if math.isfinite(value)
        )
        print(f"[CheMeleon hold-out test] {metric_text}")
        return metrics
