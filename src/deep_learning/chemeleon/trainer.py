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
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from rdkit import Chem

from src.deep_learning.chemeleon.pretrained import (
    CHEMELEON_WEIGHTS_SHA256,
    CHEMELEON_WEIGHTS_URL,
    ensure_pretrained_weights,
)
from src.deep_learning.graphormer.evaluation.classification import (
    ClassificationEvaluator,
)
from src.deep_learning.graphormer.evaluation.regression import RegressionEvaluator
from src.deep_learning.utils.train_utils import save_json, set_seed


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
    target_column: str = "target"
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
    verbose: bool = True


@dataclass
class ModelConfig:
    pretrained_path: str | None = None
    pretrained_url: str = CHEMELEON_WEIGHTS_URL
    pretrained_sha256: str | None = CHEMELEON_WEIGHTS_SHA256
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
    if base.task not in {"regression", "classification"}:
        raise ValueError("CheMeleon task must be 'regression' or 'classification'.")

    dataset.split_type = str(dataset.split_type).strip().lower()
    allowed_splits = {
        "random",
        "random_with_repeated_smiles",
        "scaffold_balanced",
        "kennard_stone",
        "kmeans",
    }
    if dataset.split_type not in allowed_splits:
        raise ValueError(
            f"Unsupported split_type {dataset.split_type!r}; "
            f"expected one of {sorted(allowed_splits)}."
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

    return CheMeleonRunConfig(base, dataset, training, model)


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _valid_records(frame: pd.DataFrame, config: DatasetConfig) -> tuple[list[dict], list[dict]]:
    required = [config.smiles_column, config.target_column]
    if config.split_column:
        required.append(config.split_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}")

    records: list[dict] = []
    rejected: list[dict] = []
    for row_index, row in frame.iterrows():
        smiles = str(row[config.smiles_column]).strip()
        target = pd.to_numeric(pd.Series([row[config.target_column]]), errors="coerce").iloc[0]
        reason = None
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            reason = "invalid_smiles"
        elif not np.isfinite(target):
            reason = "missing_or_non_numeric_target"

        if reason:
            rejected.append(
                {"original_index": row_index, "smiles": smiles, "reason": reason}
            )
            continue

        records.append(
            {
                "original_index": row_index,
                "smiles": smiles,
                "target": float(target),
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
    molecules = [item["mol"] for item in records]
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


def _datapoints(records: list[dict]) -> list[data.MoleculeDatapoint]:
    return [
        data.MoleculeDatapoint.from_smi(
            item["smiles"],
            np.asarray([item["target"]], dtype=np.float32),
        )
        for item in records
    ]


def _save_split_manifest(
    workdir: Path,
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> None:
    rows = []
    for split_name, records in (("train", train), ("validation", val), ("test", test)):
        rows.extend(
            {
                "original_index": item["original_index"],
                "smiles": item["smiles"],
                "target": item["target"],
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
) -> models.MPNN:
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    if not {"hyper_parameters", "state_dict"}.issubset(checkpoint):
        raise ValueError("Invalid CheMeleon checkpoint: expected hyper_parameters and state_dict.")

    parameters = dict(checkpoint["hyper_parameters"])
    parameters["dropout"] = float(config.model.dropout)
    message_passing = nn.BondMessagePassing(**parameters)
    message_passing.load_state_dict(checkpoint["state_dict"], strict=True)

    output_transform = None
    if config.base.task == "regression":
        scaler = train_dataset.normalize_targets()
        val_dataset.normalize_targets(scaler)
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        predictor = nn.RegressionFFN(
            input_dim=message_passing.output_dim,
            hidden_dim=int(config.model.ffn_hidden_dim),
            n_layers=int(config.model.ffn_num_layers),
            dropout=float(config.model.dropout),
            output_transform=output_transform,
        )
        metrics = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()]
    else:
        predictor = nn.BinaryClassificationFFN(
            input_dim=message_passing.output_dim,
            hidden_dim=int(config.model.ffn_hidden_dim),
            n_layers=int(config.model.ffn_num_layers),
            dropout=float(config.model.dropout),
        )
        metrics = [
            nn.metrics.BinaryAUROC(),
            nn.metrics.BinaryAUPRC(),
            nn.metrics.BinaryAccuracy(),
            nn.metrics.BinaryF1Score(),
        ]

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


def _evaluate(
    task: str,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    prediction_tensor = torch.as_tensor(predictions, dtype=torch.float32).reshape(-1, 1)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32).reshape(-1, 1)
    if task == "regression":
        loss = functional.mse_loss(prediction_tensor, target_tensor).item()
        return RegressionEvaluator().compute(
            prediction_tensor,
            target_tensor,
            loss=loss,
            prefix="test",
        )

    probabilities = prediction_tensor.clamp(1e-7, 1.0 - 1e-7)
    loss = functional.binary_cross_entropy(probabilities, target_tensor).item()
    logits = torch.logit(probabilities)
    return ClassificationEvaluator(loss_type="binary").compute(
        logits,
        target_tensor,
        loss=loss,
        prefix="test",
    )


def _predict_checkpoint(
    model: models.MPNN,
    dataloader,
    checkpoint_path: str | Path,
) -> np.ndarray:
    """Predict a complete hold-out set on one rank in dataloader order."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

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
            batches.append(prediction.detach().cpu().reshape(-1))

    if not batches:
        return np.asarray([], dtype=np.float32)
    return torch.cat(batches).numpy()


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

        if self.config.base.task == "classification":
            labels = {item["target"] for item in (*records, *test_records)}
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
        model = _make_model(
            self.config,
            pretrained_path,
            train_dataset,
            val_dataset,
        )

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
        val_loader = data.build_dataloader(
            val_dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
        )
        test_loader = (
            data.build_dataloader(
                test_dataset,
                batch_size=batch_size,
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
            auto_insert_metric_name=False,
        )
        callbacks: list[Callback] = [checkpoint_callback]
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

        logger = CSVLogger(save_dir=self.workdir, name="logs")
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
            logger=logger,
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
            },
            "dataset_summary": {
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
                self.workdir, train_records, val_records, test_records
            )
            save_json(resolved, self.workdir / "config.json")

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
            trainer.fit(model, train_loader, val_loader)

        # DDP workers must not race while copying logs, evaluating the full
        # hold-out set, or writing final artifacts. The global-zero worker
        # evaluates with the original sequential loader, preserving every row
        # and its order (distributed prediction samplers can pad or reorder).
        if not trainer.is_global_zero:
            return None

        metrics_path = Path(logger.log_dir) / "metrics.csv"
        if metrics_path.is_file():
            shutil.copy2(metrics_path, self.workdir / "training_history.csv")
        if self.config.training.plot_training_history:
            _plot_history(metrics_path, self.workdir / "plots" / "training_history.png")

        best_path = checkpoint_callback.best_model_path
        if not best_path:
            raise RuntimeError("Training completed without producing a best checkpoint.")

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
            [item["target"] for item in test_records],
            dtype=np.float64,
        )
        if predictions.shape[0] != targets.shape[0]:
            raise RuntimeError(
                "Prediction count does not match the hold-out test set: "
                f"{predictions.shape[0]} versus {targets.shape[0]}."
            )

        metrics = _evaluate(self.config.base.task, predictions, targets)
        save_json(
            {"checkpoint": best_path, **metrics},
            self.workdir / "test_metrics.json",
        )
        prediction_frame = pd.DataFrame(
            {
                "original_index": [item["original_index"] for item in test_records],
                self.config.dataset.smiles_column: [item["smiles"] for item in test_records],
                f"true_{self.config.dataset.target_column}": targets,
                f"pred_{self.config.dataset.target_column}": predictions,
            }
        )
        prediction_frame.to_csv(self.workdir / "test_predictions.csv", index=False)

        metric_text = " ".join(
            f"{name}={value:.4f}"
            for name, value in metrics.items()
            if math.isfinite(value)
        )
        print(f"[CheMeleon hold-out test] {metric_text}")
        return metrics
