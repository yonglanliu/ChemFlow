"""ChemFlow training with Hugging Face's Graphormer implementation."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from rdkit import Chem
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from chemflow.deep_learning.hf_graphormer.featurizer import (
    GraphormerFeaturizer,
)
from chemflow.deep_learning.task_weighting import (
    resolve_task_loss_weights,
    resolve_task_types,
)
from chemflow.deep_learning.utils import move_optimizer_state_to_device
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
    fit_embedding_projection,
    fit_multitask_validation_calibration,
    fit_ood_calibration,
    packed_morgan_fingerprints,
)


def _graphormer_imports():
    try:
        from transformers import (
            GraphormerConfig,
            GraphormerForGraphClassification,
        )
        try:
            from transformers.models.graphormer.collating_graphormer import (
                GraphormerDataCollator,
                preprocess_item,
            )
        except ModuleNotFoundError:
            from transformers.models.deprecated.graphormer.collating_graphormer import (
                GraphormerDataCollator,
                preprocess_item,
            )
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "Hugging Face Graphormer dependencies are unavailable. Install "
            "them with `pip install -e '.[hf-graphormer]'`."
        ) from error
    return GraphormerConfig, GraphormerForGraphClassification, GraphormerDataCollator, preprocess_item


@dataclass
class Record:
    original_index: int
    name: str
    smiles: str
    targets: np.ndarray
    split: str | None


class ReferenceDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return copy.deepcopy(self.items[index])


def _normalise_split(value: Any) -> str | None:
    aliases = {
        "train": "train", "training": "train",
        "val": "val", "valid": "val", "validation": "val",
        "test": "test", "testing": "test",
    }
    return aliases.get(str(value).strip().lower())


def _validate_split_config(config: dict[str, Any]) -> None:
    """Validate generated molecular splits or an existing split column."""
    split_type = str(config.get("split_type", "scaffold_balanced")).strip().lower()
    if config.get("split_column"):
        split_type = "predefined"
    allowed = {
        "random", "random_with_repeated_smiles", "scaffold_balanced",
        "kennard_stone", "kmeans", "predefined",
    }
    if split_type not in allowed:
        raise ValueError(
            f"Unsupported split_type {split_type!r}; expected one of {sorted(allowed)}."
        )
    if split_type == "predefined" and not config.get("split_column"):
        raise ValueError(
            "split_type='predefined' requires DatasetConfig.split_column."
        )
    if config.get("test_dataset_path") and float(
        config.get("test_fraction", 0.0)
    ) != 0.0:
        raise ValueError(
            "DatasetConfig.test_dataset_path requires test_fraction = 0.0."
        )
    config["split_type"] = split_type


def _select_device(requested: str) -> torch.device:
    name = requested.strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(name)


def _tensorboard_writer(
    workdir: Path,
    config: dict[str, Any],
    *,
    purge_step: int | None = None,
):
    """Create a TensorBoard writer while keeping dependency errors clear."""
    if not bool(config.get("tensorboard", True)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError(
            "TensorBoard logging is enabled. Install it with "
            "`python -m pip install tensorboard`, or set "
            "TrainingConfig.tensorboard = false."
        ) from error
    configured = Path(str(config.get("tensorboard_dir", "tensorboard"))).expanduser()
    log_dir = configured if configured.is_absolute() else workdir / configured
    print(f"TensorBoard logs: {log_dir}")
    kwargs = {"log_dir": str(log_dir)}
    if purge_step is not None:
        kwargs["purge_step"] = int(purge_step)
    return SummaryWriter(**kwargs)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {name: float("nan") for name in ("loss", "rmse", "mae", "r2", "pearson", "spearman")}
    result = {
        "loss": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }
    result["pearson"] = float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    result["spearman"] = float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    return result


def _classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate binary metrics from positive-class probabilities."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    metric_names = (
        "loss",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    )
    if y_true.size == 0:
        return {name: float("nan") for name in metric_names}

    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    predicted = (clipped >= float(threshold)).astype(np.int64)
    has_both_classes = np.unique(y_true).size > 1
    return {
        "loss": float(
            -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))
        ),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "roc_auc": (
            float(roc_auc_score(y_true, clipped))
            if has_both_classes
            else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(y_true, clipped))
            if has_both_classes
            else float("nan")
        ),
    }


class HuggingFaceGraphormerTrainer:
    """Masked single- or multitask regression/classification trainer."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        with self.config_path.open("rb") as stream:
            self.raw = tomllib.load(stream)

        self.base = self.raw.get("BaseConfig", {})
        self.data_cfg = self.raw.get("DatasetConfig", {})
        _validate_split_config(self.data_cfg)
        self.train_cfg = self.raw.get("TrainingConfig", {})
        self.model_cfg = self.raw.get("ModelConfig", {})
        self.task = str(self.base.get("task", "regression")).strip().lower()
        self.task_types = resolve_task_types(
            self.task, self._target_columns(), self.data_cfg.get("task_types")
        )
        self.threshold = float(self.train_cfg.get("classification_threshold", 0.5))
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "TrainingConfig.classification_threshold must be between 0 and 1."
            )
        self.workdir = Path(self.base.get("workdir", "./hf_graphormer_run")).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.device = _select_device(str(self.train_cfg.get("device", "auto")))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        strategy = str(self.train_cfg.get("strategy", "auto")).strip().lower()
        if strategy.startswith("ddp") and not self.distributed:
            raise RuntimeError(
                "DDP was requested but only one process is running. Launch with "
                "`torchrun --standalone --nproc_per_node=NUM_GPUS $(which chemflow) "
                "train hf-graphormer CONFIG`."
            )
        if self.distributed:
            if self.device.type != "cuda":
                raise RuntimeError("HF Graphormer DDP currently requires CUDA GPUs.")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
        self.is_main_process = self.rank == 0
        self.pretrained_weight_audit: dict[str, Any] = {}
        self._preserve_loaded_head = False
        self.seed = int(self.train_cfg.get("seed", 42))
        process_seed = self.seed + self.rank
        random.seed(process_seed)
        np.random.seed(process_seed)
        torch.manual_seed(process_seed)

        (
            self.GraphormerConfig,
            self.GraphormerForGraphClassification,
            self.GraphormerDataCollator,
            self.preprocess_item,
        ) = _graphormer_imports()

    def _target_columns(self) -> list[str]:
        configured = self.data_cfg["target_column"]
        columns = [configured] if isinstance(configured, str) else list(configured)
        columns = [str(column) for column in columns]
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("DatasetConfig.target_column must contain unique target names.")
        return columns

    def _load_records(self) -> list[Record]:
        path = Path(self.data_cfg["dataset_path"]).expanduser().resolve()
        frame = pd.read_csv(path)
        smiles_column = str(self.data_cfg.get("smiles_column", "SMILES"))
        target_columns = self._target_columns()
        split_column = self.data_cfg.get("split_column")
        name_column = self.data_cfg.get("name_column")
        required = {smiles_column, *target_columns}
        if split_column:
            required.add(str(split_column))
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Dataset is missing required columns: {missing}")

        records: list[Record] = []
        rejected: list[dict[str, Any]] = []
        for index, row in frame.iterrows():
            smiles = "" if pd.isna(row[smiles_column]) else str(row[smiles_column]).strip()
            targets = pd.to_numeric(row[target_columns], errors="coerce").to_numpy(dtype=float)
            split = _normalise_split(row[split_column]) if split_column else None
            if not smiles or Chem.MolFromSmiles(smiles) is None:
                rejected.append({
                    "dataset": "training",
                    "index": index,
                    "smiles": smiles,
                    "reason": "invalid_smiles",
                })
                continue
            if not np.isfinite(targets).any():
                rejected.append({
                    "dataset": "training",
                    "index": index,
                    "smiles": smiles,
                    "reason": "all_targets_missing",
                })
                continue
            if split_column and split is None:
                rejected.append({
                    "dataset": "training",
                    "index": index,
                    "smiles": smiles,
                    "reason": "invalid_split",
                })
                continue
            name = str(row[name_column]) if name_column and name_column in frame.columns else str(index)
            records.append(
                Record(
                    original_index=int(index),
                    name=name,
                    smiles=smiles,
                    targets=targets,
                    split=split,
                )
            )

        external_test_path = self.data_cfg.get("test_dataset_path")
        if external_test_path:
            if any(record.split == "test" for record in records):
                raise ValueError(
                    "The training dataset contains test rows while "
                    "test_dataset_path is also configured. Use only one test source."
                )
            resolved_test_path = Path(external_test_path).expanduser().resolve()
            if not resolved_test_path.is_file():
                raise FileNotFoundError(
                    f"Test dataset does not exist: {resolved_test_path}"
                )
            test_frame = pd.read_csv(resolved_test_path)
            missing_test = sorted(
                {smiles_column, *target_columns} - set(test_frame.columns)
            )
            if missing_test:
                raise KeyError(
                    "External test dataset is missing required columns: "
                    f"{missing_test}"
                )
            offset = len(frame)
            external_count = 0
            for position, row in test_frame.iterrows():
                smiles = (
                    ""
                    if pd.isna(row[smiles_column])
                    else str(row[smiles_column]).strip()
                )
                targets = pd.to_numeric(
                    row[target_columns], errors="coerce"
                ).to_numpy(dtype=float)
                if not smiles or Chem.MolFromSmiles(smiles) is None:
                    rejected.append({
                        "dataset": "external_test",
                        "index": position,
                        "smiles": smiles,
                        "reason": "invalid_smiles",
                    })
                    continue
                if not np.isfinite(targets).any():
                    rejected.append({
                        "dataset": "external_test",
                        "index": position,
                        "smiles": smiles,
                        "reason": "all_targets_missing",
                    })
                    continue
                name = (
                    str(row[name_column])
                    if name_column
                    and name_column in test_frame.columns
                    and pd.notna(row[name_column])
                    else f"test:{position}"
                )
                records.append(
                    Record(
                        original_index=offset + int(position),
                        name=name,
                        smiles=smiles,
                        targets=targets,
                        split="test",
                    )
                )
                external_count += 1
            if external_count == 0:
                raise ValueError(
                    "No valid labeled records remain in the external test dataset."
                )

        if self.is_main_process:
            pd.DataFrame(
                rejected,
                columns=["dataset", "index", "smiles", "reason"],
            ).to_csv(
                self.workdir / "rejected_rows.csv", index=False
            )
        if not records:
            raise ValueError("No valid labeled records remain.")
        if "classification" in self.task_types:
            indices = [i for i, value in enumerate(self.task_types) if value == "classification"]
            finite_labels = np.concatenate([
                record.targets[indices][np.isfinite(record.targets[indices])]
                for record in records
            ])
            invalid = sorted(set(finite_labels.tolist()) - {0.0, 1.0})
            if invalid:
                raise ValueError(
                    "Binary classification targets must contain only 0, 1, or "
                    f"missing values; found {invalid}."
                )
        return records

    def _split_records(self, records: list[Record]) -> dict[str, list[Record]]:
        if self.data_cfg.get("split_type") == "predefined":
            splits = {name: [r for r in records if r.split == name] for name in ("train", "val", "test")}
        else:
            # Use ChemFlow's established molecular split implementation so the
            # same seed and strategy can be reused by both Graphormer backends.
            from chemprop import data as chemprop_data

            external_test = [record for record in records if record.split == "test"]
            split_pool = [record for record in records if record.split != "test"]
            val_fraction = float(self.data_cfg.get("val_fraction", 0.1))
            test_fraction = (
                0.0
                if external_test
                else float(self.data_cfg.get("test_fraction", 0.1))
            )
            split_type = str(self.data_cfg.get("split_type", "scaffold_balanced"))
            indices = chemprop_data.make_split_indices(
                [Chem.MolFromSmiles(r.smiles) for r in split_pool],
                split=split_type,
                sizes=(1.0 - val_fraction - test_fraction, val_fraction, test_fraction),
                seed=self.seed,
            )
            splits = {
                name: [split_pool[int(i)] for i in values[0]]
                for name, values in zip(("train", "val", "test"), indices)
            }
            if external_test:
                splits["test"] = external_test
        if not splits["train"] or not splits["val"]:
            raise ValueError("Training and validation splits must be non-empty.")
        target_columns = self._target_columns()
        rows = []
        for split_name, values in splits.items():
            for record in values:
                row = {"Name": record.name, "SMILES": record.smiles, "split": split_name}
                row.update(dict(zip(target_columns, record.targets)))
                rows.append(row)
        if self.is_main_process:
            pd.DataFrame(rows).to_csv(self.workdir / "data_splits.csv", index=False)
        return splits

    def _featurize(
        self,
        records: list[Record],
        means: np.ndarray,
        stds: np.ndarray,
    ) -> ReferenceDataset:
        featurizer = GraphormerFeaturizer(
            remove_hs=bool(self.data_cfg.get("remove_hs", True)),
            reorder_atoms=bool(self.data_cfg.get("reorder_atoms", False)),
        )
        items: list[dict[str, Any]] = []
        max_nodes = int(self.model_cfg.get("max_nodes", 512))
        for record in records:
            raw = featurizer.smiles2graph(record.smiles)
            if int(raw["num_nodes"]) > max_nodes:
                continue
            labels = record.targets.copy()
            regression_mask = np.asarray(self.task_types) == "regression"
            labels[regression_mask] = (
                labels[regression_mask] - means[regression_mask]
            ) / stds[regression_mask]
            item = {
                "edge_index": np.asarray(raw["edge_index"], dtype=np.int64),
                "edge_attr": np.asarray(raw["edge_attr"], dtype=np.int64),
                "node_feat": np.asarray(raw["node_feat"], dtype=np.int64),
                "num_nodes": int(raw["num_nodes"]),
                # Labels are removed before calling the HF model so ChemFlow
                # can consistently mask missing single- or multitask labels.
                "y": labels.astype(np.float32),
                "labels": labels.astype(np.float32),
                "name": record.name,
                "smiles": record.smiles,
                "original_index": record.original_index,
                "raw_targets": record.targets.astype(float),
            }
            try:
                items.append(self.preprocess_item(item))
            except ImportError as error:
                raise ImportError(
                    "Hugging Face Graphormer preprocessing requires Cython. "
                    "Install the optional `hf-graphormer` dependencies."
                ) from error
        return ReferenceDataset(items)

    @staticmethod
    def _reset_head(model: torch.nn.Module) -> None:
        # The PCQM4M HOMO-LUMO output layer must not be reused for an ADMET target.
        for module in model.classifier.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        if hasattr(model.classifier, "lm_output_learned_bias"):
            torch.nn.init.zeros_(model.classifier.lm_output_learned_bias)

    @staticmethod
    def _parameter_count(module: torch.nn.Module) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in module.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        return total, trainable

    def _print_parameter_summary(self, model: torch.nn.Module) -> None:
        total, trainable = self._parameter_count(model)
        encoder_total, encoder_trainable = self._parameter_count(model.encoder)
        head_total, head_trainable = self._parameter_count(model.classifier)
        frozen = total - trainable
        ratio = 100.0 * trainable / total if total else 0.0

        print("Hugging Face Graphormer parameter summary:")
        print(f"  Total parameters:       {total:,}")
        print(f"  Trainable parameters:   {trainable:,} ({ratio:.2f}%)")
        print(f"  Frozen parameters:      {frozen:,}")
        print(
            "  Encoder:                "
            f"{encoder_trainable:,} / {encoder_total:,} trainable"
        )
        print(
            "  Prediction head:        "
            f"{head_trainable:,} / {head_total:,} trainable"
        )

    def _collator(self):
        base = self.GraphormerDataCollator(
            spatial_pos_max=int(self.model_cfg.get("spatial_pos_max", 1024)),
            on_the_fly_processing=False,
        )

        def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
            metadata = {
                "names": [item["name"] for item in items],
                "smiles": [item["smiles"] for item in items],
                "original_indices": [item["original_index"] for item in items],
                "raw_targets": [item["raw_targets"] for item in items],
            }
            model_items = [
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {"name", "smiles", "original_index", "raw_targets"}
                }
                for item in items
            ]
            batch = base(model_items)
            batch.update(metadata)
            return batch

        return collate

    def _load_model(self, num_tasks: int):
        """Load from a local HF-format checkpoint or from the Hub.

        The local path is useful on institutional networks whose TLS proxy is
        not trusted by Python Requests.  ChemFlow's cached checkpoint is the
        same state dictionary published in the Hugging Face repository.
        """
        transfer_value = self.model_cfg.get("transfer_checkpoint")
        if transfer_value:
            transfer_path = Path(str(transfer_value)).expanduser().resolve()
            if not transfer_path.is_dir():
                raise FileNotFoundError(
                    "Graphormer transfer_checkpoint must be a Hugging Face "
                    f"best_model directory: {transfer_path}"
                )

            encoder_only = bool(
                self.model_cfg.get("transfer_encoder_only", False)
            )
            model = self.GraphormerForGraphClassification.from_pretrained(
                transfer_path,
                num_classes=num_tasks,
                ignore_mismatched_sizes=encoder_only,
                local_files_only=True,
            )
            saved_targets = list(
                getattr(model.config, "chemflow_target_names", []) or []
            )
            requested_targets = self._target_columns()
            if not encoder_only:
                if int(model.config.num_classes) != num_tasks:
                    raise ValueError(
                        "Full Graphormer transfer requires the same number of "
                        "targets. Use transfer_encoder_only=true when changing "
                        "the prediction head."
                    )
                if saved_targets and saved_targets != requested_targets:
                    raise ValueError(
                        "Full Graphormer transfer requires identical targets in "
                        f"the same order. Saved: {saved_targets}; requested: "
                        f"{requested_targets}. Use transfer_encoder_only=true "
                        "when changing targets."
                    )
                self._preserve_loaded_head = True

            transfer_mode = "encoder_only" if encoder_only else "full_model"
            self.pretrained_weight_audit = {
                "source": "ChemFlow downstream Graphormer checkpoint",
                "resolved_path": str(transfer_path),
                "local_files_only": True,
                "checkpoint_compatibility": "passed_from_pretrained",
                "transfer_mode": transfer_mode,
                "saved_targets": saved_targets,
                "requested_targets": requested_targets,
                "prediction_head": (
                    "reset_for_new_tasks"
                    if encoder_only
                    else "restored_from_transfer_checkpoint"
                ),
            }
            if self.is_main_process:
                print("Hugging Face Graphormer transfer-weight audit:")
                print(f"  Source: {transfer_path}")
                print(f"  Transfer mode: {transfer_mode}")
                print(f"  Saved targets: {saved_targets or 'not recorded'}")
                print(f"  Requested targets: {requested_targets}")
                print(
                    "  Prediction head: "
                    f"{self.pretrained_weight_audit['prediction_head']}"
                )
            return model

        checkpoint_value = self.model_cfg.get("checkpoint_path")
        if not checkpoint_value:
            model_name = str(
                self.model_cfg.get(
                    "model_name",
                    "clefourrier/graphormer-base-pcqm4mv1",
                )
            )
            try:
                model = self.GraphormerForGraphClassification.from_pretrained(
                    model_name,
                    num_classes=num_tasks,
                    ignore_mismatched_sizes=num_tasks != 1,
                    local_files_only=bool(
                        self.model_cfg.get("local_files_only", False)
                    ),
                )
            except OSError as error:
                raise RuntimeError(
                    "Could not load Hugging Face Graphormer from the Hub. If "
                    "your network intercepts HTTPS, set ModelConfig.checkpoint_path "
                    "to the cached graphormer-base-pcqm4mv1.pt file."
                ) from error

            model_path = Path(model_name).expanduser()
            self.pretrained_weight_audit = {
                "source": "local Hugging Face directory" if model_path.exists() else "Hugging Face Hub/cache",
                "model_name": model_name,
                "resolved_path": str(model_path.resolve()) if model_path.exists() else None,
                "local_files_only": bool(self.model_cfg.get("local_files_only", False)),
                "checkpoint_compatibility": "passed_from_pretrained",
                "cryptographic_checksum": "not_configured",
                "prediction_head": "reset_for_downstream_tasks",
            }
            if self.is_main_process:
                print("Hugging Face Graphormer pretrained-weight audit:")
                print(f"  Source: {self.pretrained_weight_audit['source']}")
                print(f"  Model: {model_name}")
                if self.pretrained_weight_audit["resolved_path"]:
                    print(f"  Resolved path: {self.pretrained_weight_audit['resolved_path']}")
                print("  Model loading: passed (from_pretrained)")
                print("  Prediction head: reset for downstream tasks")
            return model

        checkpoint_path = Path(str(checkpoint_value)).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Hugging Face Graphormer checkpoint not found: {checkpoint_path}"
            )

        # Exact configuration stored by
        # clefourrier/graphormer-base-pcqm4mv1. Keeping it here makes the
        # reference run reproducible and avoids a config.json network request.
        config = self.GraphormerConfig(
            num_classes=num_tasks,
            num_atoms=4608,
            num_edges=1536,
            num_in_degree=512,
            num_out_degree=512,
            num_spatial=512,
            num_edge_dis=128,
            multi_hop_max_dist=int(
                self.model_cfg.get("multi_hop_max_dist", 5)
            ),
            spatial_pos_max=int(self.model_cfg.get("spatial_pos_max", 1024)),
            edge_type="multi_hop",
            max_nodes=int(self.model_cfg.get("max_nodes", 512)),
            num_hidden_layers=12,
            embedding_dim=768,
            ffn_embedding_dim=768,
            num_attention_heads=32,
            dropout=0.0,
            attention_dropout=0.1,
            activation_dropout=0.1,
            encoder_normalize_before=True,
            pre_layernorm=False,
            apply_graphormer_init=True,
            activation_fn="gelu",
        )
        model = self.GraphormerForGraphClassification(config)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Expected a checkpoint dictionary, got {type(checkpoint)}"
            )
        for key in ("model_state_dict", "model", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break
        expected_missing: set[str] = set()
        if num_tasks != 1:
            # PCQM4Mv1 has a one-output pretraining head. Reuse its encoder,
            # but initialize a correctly sized ADMET multitask head.
            for key in ("classifier.classifier.weight",):
                if key in checkpoint:
                    checkpoint.pop(key)
                    expected_missing.add(key)
        incompatible = model.load_state_dict(checkpoint, strict=False)
        actual_missing = set(incompatible.missing_keys)
        if actual_missing != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "The configured local checkpoint is not the expected Hugging "
                "Face PCQM4Mv1 Graphormer checkpoint. "
                f"Missing keys: {incompatible.missing_keys}; unexpected keys: "
                f"{incompatible.unexpected_keys}."
            )
        if self.is_main_process:
            print("Hugging Face Graphormer pretrained-weight audit:")
            print("  Source: configured local checkpoint")
            print(f"  Resolved path: {checkpoint_path}")
            print(f"  File size: {checkpoint_path.stat().st_size:,} bytes")
            print("  Checkpoint compatibility: passed")
            print("  Prediction head: reset for downstream tasks")
        self.pretrained_weight_audit = {
            "source": "configured local checkpoint",
            "model_name": self.model_cfg.get("model_name"),
            "resolved_path": str(checkpoint_path),
            "file_size_bytes": int(checkpoint_path.stat().st_size),
            "local_files_only": True,
            "checkpoint_compatibility": "passed_expected_keys",
            "cryptographic_checksum": "not_configured",
            "prediction_head": "reset_for_downstream_tasks",
        }
        return model

    def _loader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        distributed: bool = False,
    ) -> DataLoader:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                seed=self.seed,
                drop_last=False,
            )
            if distributed and self.distributed
            else None
        )
        return DataLoader(
            dataset,
            batch_size=int(self.train_cfg.get("batch_size", 16)),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=int(self.train_cfg.get("num_workers", 0)),
            collate_fn=self._collator(),
            pin_memory=self.device.type == "cuda",
        )

    @staticmethod
    def _masked_multitask_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        task_weights: torch.Tensor,
        task: str = "regression",
        task_types: list[str] | None = None,
    ) -> torch.Tensor:
        """Return a weighted mean over the finite labels in a batch."""
        mask = torch.isfinite(labels)
        if not mask.any():
            raise RuntimeError("A training batch contains no finite targets.")
        finite_labels = torch.nan_to_num(labels, nan=0.0)
        resolved_types = task_types or [task] * labels.shape[1]
        element_loss = torch.zeros_like(logits)
        for index, task_type in enumerate(resolved_types):
            if task_type == "classification":
                element_loss[:, index] = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[:, index], finite_labels[:, index], reduction="none"
                )
            else:
                element_loss[:, index] = (logits[:, index] - finite_labels[:, index]).square()
        weights = task_weights.reshape(1, -1).expand_as(labels)
        effective_weights = weights * mask
        return (element_loss * effective_weights).sum() / effective_weights.sum()

    def _evaluate(
        self,
        model,
        loader,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
        model.eval()
        rows = []
        target_columns = self._target_columns()
        with torch.no_grad():
            for batch in loader:
                names, smiles = batch.pop("names"), batch.pop("smiles")
                batch.pop("original_indices")
                raw_targets = np.asarray(batch.pop("raw_targets"), dtype=float)
                batch.pop("labels", None)
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                logits = model(**inputs).logits.detach().cpu().numpy()
                logits = logits.reshape(len(names), len(target_columns))
                predictions = logits * stds + means
                for index, task_type in enumerate(self.task_types):
                    if task_type == "classification":
                        predictions[:, index] = expit(logits[:, index])
                for name, smiles_value, truth, prediction in zip(
                    names, smiles, raw_targets, predictions
                ):
                    row: dict[str, Any] = {"Name": name, "SMILES": smiles_value}
                    for task_index, target in enumerate(target_columns):
                        row[f"{target}_true"] = truth[task_index]
                        row[f"{target}_prediction"] = prediction[task_index]
                        if self.task_types[task_index] == "classification":
                            row[f"{target}_probability"] = prediction[task_index]
                            row[f"{target}_class"] = int(
                                prediction[task_index] >= self.threshold
                            )
                    rows.append(row)
        frame = pd.DataFrame(rows)
        metrics: dict[str, dict[str, float]] = {}
        for task_index, target in enumerate(target_columns):
            true_values = frame[f"{target}_true"].to_numpy(dtype=float)
            predictions = frame[f"{target}_prediction"].to_numpy(dtype=float)
            mask = np.isfinite(true_values) & np.isfinite(predictions)
            if self.task_types[task_index] == "classification":
                metrics[target] = _classification_metrics(
                    true_values[mask],
                    predictions[mask],
                    threshold=self.threshold,
                )
            else:
                metrics[target] = _metrics(true_values[mask], predictions[mask])
                # Mixed-task checkpoint selection must not be dominated by the
                # physical units of one regression endpoint. Keep the reported
                # RMSE/MAE in original units, but define loss in the same
                # standardized target space used for optimization.
                scale = max(float(stds[task_index]), 1e-12)
                metrics[target]["loss"] = float(
                    np.mean(
                        (
                            (predictions[mask] - true_values[mask])
                            / scale
                        )
                        ** 2
                    )
                )
            metrics[target]["n"] = int(mask.sum())
        return metrics, frame

    def _predict_with_embeddings(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> dict[str, Any]:
        """Return ordered metadata, predictions, and fine-tuned graph tokens."""
        model.eval()
        names: list[str] = []
        smiles: list[str] = []
        original_indices: list[int] = []
        targets: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                names.extend(str(value) for value in batch.pop("names"))
                smiles.extend(str(value) for value in batch.pop("smiles"))
                original_indices.extend(int(value) for value in batch.pop("original_indices"))
                targets.extend(np.asarray(batch.pop("raw_targets"), dtype=np.float32))
                batch.pop("labels", None)
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                encoder_output = model.encoder(**inputs, return_dict=True)
                hidden = encoder_output["last_hidden_state"]
                logits = model.classifier(hidden)[:, 0, :]
                raw_prediction = logits.detach().float().cpu().numpy()
                prediction = raw_prediction * stds + means
                for index, task_type in enumerate(self.task_types):
                    if task_type == "classification":
                        prediction[:, index] = expit(raw_prediction[:, index])
                predictions.extend(prediction.astype(np.float32))
                embeddings.extend(hidden[:, 0, :].detach().float().cpu().numpy())
        return {
            "names": names,
            "smiles": smiles,
            "original_indices": original_indices,
            "targets": np.asarray(targets, dtype=np.float32),
            "predictions": np.asarray(predictions, dtype=np.float32),
            "embeddings": np.asarray(embeddings, dtype=np.float32),
        }

    @staticmethod
    def _reference_partition(
        reference: dict[str, Any],
        projected_embeddings: np.ndarray,
        target_names: list[str],
        selected: np.ndarray | None = None,
        *,
        predictions: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if selected is None:
            selected = np.arange(len(reference["smiles"]), dtype=np.int64)
        selected = np.asarray(selected, dtype=np.int64)
        partition: dict[str, Any] = {
            "original_index": [reference["original_indices"][int(i)] for i in selected],
            "name": [reference["names"][int(i)] for i in selected],
            "smiles": [reference["smiles"][int(i)] for i in selected],
            "target_names": list(target_names),
            "targets": torch.from_numpy(reference["targets"][selected].astype(np.float32)),
            "embeddings": torch.from_numpy(
                projected_embeddings[selected].astype(np.float16)
            ),
        }
        if predictions is not None:
            partition["predictions"] = torch.from_numpy(
                np.asarray(predictions, dtype=np.float32)
            )
        return partition

    def _build_applicability_bundle(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        means: np.ndarray,
        stds: np.ndarray,
        best_dir: Path,
    ) -> None:
        """Store training chemical space and validation calibration with a model."""
        target_names = self._target_columns()
        train_reference = self._predict_with_embeddings(
            model, self._loader(train_dataset, shuffle=False), means, stds
        )
        val_reference = self._predict_with_embeddings(
            model, self._loader(val_dataset, shuffle=False), means, stds
        )
        projection, train_projected, val_projected = fit_embedding_projection(
            train_reference["embeddings"],
            val_reference["embeddings"],
            dimensions=MAX_EMBEDDING_DIMENSIONS,
            seed=self.seed,
        )
        calibration = fit_multitask_validation_calibration(
            self.task_types,
            val_reference["targets"],
            val_reference["predictions"],
            target_names,
            confidence=DEFAULT_CALIBRATION_CONFIDENCE,
        )
        payload: dict[str, Any] = {
            "version": APPLICABILITY_VERSION,
            "model_family": "huggingface_graphormer",
            "embedding_space": "fine_tuned_graph_token_pca",
            "projection": projection,
            "similarity": {
                "method": "morgan_tanimoto",
                "radius": DEFAULT_SIMILARITY_RADIUS,
                "bits": DEFAULT_SIMILARITY_BITS,
            },
            "training": self._reference_partition(
                train_reference, train_projected, target_names
            ),
            "validation": self._reference_partition(
                val_reference,
                val_projected,
                target_names,
                predictions=val_reference["predictions"],
            ),
            "calibration": calibration,
            "training_by_task": {},
            "validation_by_task": {},
            "ood_calibration": {},
            "local_calibration": {
                "method": "fingerprint_embedding_nearest_validation_residuals",
                "min_samples": DEFAULT_LOCAL_MIN_SAMPLES,
                "max_samples": DEFAULT_LOCAL_MAX_SAMPLES,
                "min_fp_similarity": DEFAULT_LOCAL_MIN_FP_SIMILARITY,
                "min_embedding_similarity": (
                    DEFAULT_LOCAL_MIN_EMBEDDING_SIMILARITY
                ),
            },
        }
        payload["training"]["fingerprints"] = torch.from_numpy(
            packed_morgan_fingerprints(payload["training"]["smiles"])
        )

        for task_index, target_name in enumerate(target_names):
            train_selected = np.flatnonzero(
                np.isfinite(train_reference["targets"][:, task_index])
            )
            val_selected = np.flatnonzero(
                np.isfinite(val_reference["targets"][:, task_index])
            )
            if len(val_selected) == 0:
                raise ValueError(
                    f"Validation split has no labels for {target_name}; "
                    "calibration cannot be fitted."
                )
            train_partition = self._reference_partition(
                train_reference,
                train_projected,
                [target_name],
                train_selected,
            )
            train_partition["targets"] = train_partition["targets"][:, task_index : task_index + 1]
            train_partition["fingerprints"] = torch.from_numpy(
                packed_morgan_fingerprints(train_partition["smiles"])
            )
            payload["training_by_task"][target_name] = train_partition

            val_partition = self._reference_partition(
                val_reference,
                val_projected,
                [target_name],
                val_selected,
                predictions=val_reference["predictions"][
                    val_selected, task_index : task_index + 1
                ],
            )
            val_partition["targets"] = val_partition["targets"][:, task_index : task_index + 1]
            val_partition["fingerprints"] = torch.from_numpy(
                packed_morgan_fingerprints(val_partition["smiles"])
            )
            payload["validation_by_task"][target_name] = val_partition

            diagnostics = applicability_diagnostics(
                [val_reference["smiles"][int(i)] for i in val_selected],
                val_reference["embeddings"][val_selected],
                list(range(len(val_selected))),
                payload,
                reference_task=target_name,
            )
            payload["ood_calibration"][target_name] = fit_ood_calibration(
                diagnostics["train_max_tanimoto"],
                diagnostics["train_embedding_cosine_distance"],
                confidence=DEFAULT_OOD_CONFIDENCE,
            )

        torch.save(payload, best_dir / "applicability.pt")
        (best_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "validation_calibration": calibration,
                    "ood_calibration": payload["ood_calibration"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved applicability and calibration bundle to {best_dir}")

    @staticmethod
    def _aggregate_metric(
        metrics: dict[str, dict[str, float]], metric_name: str
    ) -> float:
        values = [
            task_metrics[metric_name]
            for task_metrics in metrics.values()
            if np.isfinite(task_metrics[metric_name])
        ]
        if not values:
            raise RuntimeError(
                f"No finite validation {metric_name} values are available for any task."
            )
        return float(np.mean(values))

    def _save_resume_checkpoint(
        self,
        path: Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_metric: float,
        bad_epochs: int,
        history: list[dict[str, Any]],
        target_means: np.ndarray,
        target_stds: np.ndarray,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "schema_version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "task": self.task,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
            "history": history,
            "target_means": target_means.tolist(),
            "target_stds": target_stds.tolist(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_random_state"] = torch.cuda.get_rng_state_all()

        # A process-specific temporary filename prevents concurrent or stale
        # writers from moving another process's temporary checkpoint. The
        # final os.replace remains atomic on the destination filesystem.
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _restore_resume_checkpoint(
        self,
        path: Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        expected_means: np.ndarray,
        expected_stds: np.ndarray,
    ) -> tuple[int, float, int, list[dict[str, Any]]]:
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            raise TypeError(f"Invalid resume checkpoint: {path}")
        if int(state.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported resume checkpoint schema: {path}")
        saved_task = str(state.get("task", "regression"))
        if saved_task != self.task:
            raise ValueError(
                f"Resume checkpoint task is {saved_task!r}, but the current "
                f"configuration requests {self.task!r}."
            )

        # Accept checkpoints produced by the original single-target trainer.
        saved_means = np.asarray(
            state.get("target_means", [state.get("target_mean")]), dtype=float
        )
        saved_stds = np.asarray(
            state.get("target_stds", [state.get("target_std")]), dtype=float
        )
        if not np.allclose(saved_means, expected_means) or not np.allclose(
            saved_stds, expected_stds
        ):
            raise ValueError(
                "Resume target scaling does not match the current training split. "
                "Use the same dataset and split assignments."
            )

        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, self.device)
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(
            state["torch_random_state"].detach().to(
                device="cpu", dtype=torch.uint8
            )
        )
        if torch.cuda.is_available() and "cuda_random_state" in state:
            cuda_states = state["cuda_random_state"]
            if len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(
                    [
                        value.detach().to(device="cpu", dtype=torch.uint8)
                        for value in cuda_states
                    ]
                )
            elif self.is_main_process:
                print(
                    "Skipping CUDA RNG restoration because the saved and "
                    "current GPU counts differ. Training state was restored.",
                    flush=True,
                )

        completed_epoch = int(state["epoch"])
        history = list(state.get("history", []))
        if self.is_main_process:
            print(f"Resuming from {path} after completed epoch {completed_epoch}.")
        return (
            completed_epoch + 1,
            float(
                state.get(
                    "best_metric",
                    state.get("best_rmse", float("inf")),
                )
            ),
            int(state.get("bad_epochs", 0)),
            history,
        )

    def _build_applicability_from_saved_best(self) -> None:
        """Backfill domain/calibration artifacts without retraining the model."""
        split_path = self.workdir / "data_splits.csv"
        best_dir = self.workdir / "best_model"
        scaling_path = best_dir / "target_scaling.json"
        for required in (split_path, best_dir / "config.json", scaling_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"Applicability-only mode requires an existing training artifact: {required}"
                )

        target_names = self._target_columns()
        frame = pd.read_csv(split_path)
        saved_target_columns = target_names
        if len(target_names) == 1 and target_names[0] not in frame.columns:
            # Compatibility with early reference runs that named the sole
            # response column simply `target` in data_splits.csv.
            if "target" not in frame.columns:
                raise KeyError(
                    f"Saved splits do not contain {target_names[0]!r} or legacy 'target'."
                )
            saved_target_columns = ["target"]
        required_columns = {"Name", "SMILES", "split", *saved_target_columns}
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise KeyError(f"Saved splits are missing required columns: {missing}")

        records: list[Record] = []
        for index, row in frame.iterrows():
            split = _normalise_split(row["split"])
            if split is None:
                continue
            targets = pd.to_numeric(
                row[saved_target_columns], errors="coerce"
            ).to_numpy(dtype=float)
            records.append(
                Record(
                    original_index=int(index),
                    name=str(row["Name"]),
                    smiles=str(row["SMILES"]),
                    targets=targets,
                    split=split,
                )
            )
        splits = {
            name: [record for record in records if record.split == name]
            for name in ("train", "val", "test")
        }
        if not splits["train"] or not splits["val"]:
            raise ValueError("Saved training and validation splits must be non-empty.")

        scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
        if "mean" in scaling and "std" in scaling:
            means = np.asarray([scaling["mean"]], dtype=float)
            stds = np.asarray([scaling["std"]], dtype=float)
        else:
            means = np.asarray([scaling[name]["mean"] for name in target_names])
            stds = np.asarray([scaling[name]["std"] for name in target_names])
        datasets = {
            name: self._featurize(values, means, stds)
            for name, values in splits.items()
        }
        model = self.GraphormerForGraphClassification.from_pretrained(best_dir).to(
            self.device
        )
        self._build_applicability_bundle(
            model, datasets["train"], datasets["val"], means, stds, best_dir
        )
        print("Applicability-only mode completed; model weights were not changed.")

    def train(self) -> None:
        if bool(self.train_cfg.get("applicability_only", False)):
            if self.is_main_process:
                self._build_applicability_from_saved_best()
            if self.distributed:
                dist.barrier()
                dist.destroy_process_group()
            return
        records = self._load_records()
        splits = self._split_records(records)
        target_columns = self._target_columns()
        train_targets = np.stack([r.targets for r in splits["train"]])
        val_targets = np.stack([r.targets for r in splits["val"]])
        missing_training_tasks = [
            target_columns[index]
            for index in range(len(target_columns))
            if not np.isfinite(train_targets[:, index]).any()
        ]
        missing_validation_tasks = [
            target_columns[index]
            for index in range(len(target_columns))
            if not np.isfinite(val_targets[:, index]).any()
        ]
        if missing_training_tasks or missing_validation_tasks:
            raise ValueError(
                "Every task needs at least one label in both training and "
                "validation splits. Missing training tasks: "
                f"{missing_training_tasks}; missing validation tasks: "
                f"{missing_validation_tasks}. Use a task-aware or predefined split."
            )
        if "classification" in self.task_types:
            single_class_tasks = []
            for task_index, target_name in enumerate(target_columns):
                if self.task_types[task_index] != "classification":
                    continue
                values = train_targets[:, task_index]
                classes = np.unique(values[np.isfinite(values)])
                if classes.size < 2:
                    single_class_tasks.append(target_name)
            if single_class_tasks:
                raise ValueError(
                    "Classification training requires both classes for every "
                    f"task; single-class training tasks: {single_class_tasks}."
                )
        task_label_counts, task_loss_weights = resolve_task_loss_weights(
            train_targets,
            target_columns,
            strategy=str(self.train_cfg.get("task_loss_weighting", "uniform")),
            configured=self.train_cfg.get("task_loss_weights"),
        )
        if self.is_main_process:
            print("Hugging Face Graphormer task loss weighting:")
            for target_name, count, weight in zip(
                target_columns, task_label_counts, task_loss_weights
            ):
                print(f"  {target_name}: labels={int(count)}, weight={float(weight):.6f}")
        target_means = np.nanmean(train_targets, axis=0)
        target_stds = np.nanstd(train_targets, axis=0)
        for index, task_type in enumerate(self.task_types):
            if task_type == "classification":
                target_means[index], target_stds[index] = 0.0, 1.0
        target_stds[~np.isfinite(target_stds) | (target_stds <= 0)] = 1.0

        datasets = {
            name: self._featurize(values, target_means, target_stds)
            for name, values in splits.items()
        }
        loaders = {
            name: self._loader(
                ds,
                shuffle=name == "train",
                distributed=name == "train",
            )
            for name, ds in datasets.items()
            if len(ds)
        }
        # The optimization loader may be shuffled or distributed. Keep a
        # deterministic full-training loader for comparable epoch metrics.
        train_metrics_loader = self._loader(datasets["train"], shuffle=False)

        model = self._load_model(len(target_columns))
        if not self._preserve_loaded_head:
            self._reset_head(model)
        model.config.problem_type = (
            "multi_label_classification"
            if "classification" in self.task_types
            else "regression"
        )
        model.config.chemflow_task = self.task
        model.config.chemflow_target_names = target_columns
        model.config.chemflow_task_types = self.task_types
        model.config.chemflow_classification_threshold = self.threshold
        model.to(self.device)
        loss_weights_tensor = torch.as_tensor(
            task_loss_weights, dtype=torch.float32, device=self.device
        )

        if bool(self.model_cfg.get("freeze_encoder", False)):
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
        self.pretrained_weight_audit["encoder_frozen"] = bool(
            self.model_cfg.get("freeze_encoder", False)
        )
        if self.is_main_process:
            print(
                "  Encoder frozen: "
                f"{self.pretrained_weight_audit['encoder_frozen']}"
            )

        if self.is_main_process:
            self._print_parameter_summary(model)

        encoder_lr = float(self.train_cfg.get("encoder_learning_rate", 1e-5))
        head_lr = float(self.train_cfg.get("head_learning_rate", 1e-4))
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": encoder_lr},
                {"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": head_lr},
            ],
            weight_decay=float(self.train_cfg.get("weight_decay", 0.01)),
        )

        epochs = int(self.train_cfg.get("num_epochs", 30))
        patience = int(self.train_cfg.get("early_stopping_patience", 8))
        checkpoint_dir = self.workdir / "checkpoints"
        last_checkpoint = checkpoint_dir / "last.pt"
        configured_resume_path = self.train_cfg.get("resume_checkpoint")
        if configured_resume_path:
            last_checkpoint = Path(str(configured_resume_path)).expanduser().resolve()
        selection_metric_name = "loss" if self.task != "regression" else "rmse"
        best_metric = float("inf")
        bad_epochs = 0
        history: list[dict[str, Any]] = []
        task_history_path = self.workdir / "training_task_metrics.csv"
        task_history: list[dict[str, Any]] = []
        if bool(self.train_cfg.get("resume", False)) and task_history_path.is_file():
            task_history = pd.read_csv(task_history_path).to_dict(orient="records")
        start_epoch = 1
        best_dir = self.workdir / "best_model"

        if bool(self.train_cfg.get("resume", False)):
            start_epoch, best_metric, bad_epochs, history = (
                self._restore_resume_checkpoint(
                    last_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    expected_means=target_means,
                    expected_stds=target_stds,
                )
            )
            if self.distributed:
                # The rank-0 checkpoint contains one RNG snapshot. Give each
                # resumed worker a deterministic but distinct dropout stream.
                resumed_seed = self.seed + self.rank + start_epoch * self.world_size
                random.seed(resumed_seed)
                np.random.seed(resumed_seed)
                torch.manual_seed(resumed_seed)
                torch.cuda.manual_seed(resumed_seed)

        tensorboard_writer = (
            _tensorboard_writer(
                self.workdir,
                self.train_cfg,
                purge_step=start_epoch if start_epoch > 1 else None,
            )
            if self.is_main_process
            else None
        )

        training_model: torch.nn.Module = model
        if self.distributed:
            training_model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        if start_epoch > epochs and self.is_main_process:
            print(
                f"Checkpoint already completed epoch {start_epoch - 1}, which "
                f"meets num_epochs={epochs}; skipping training."
            )

        if self.is_main_process:
            print(
                f"Hugging Face Graphormer device: {self.device}; "
                f"DDP world size: {self.world_size}"
            )
            print("Split sizes:", {name: len(ds) for name, ds in datasets.items()})
        for epoch in range(start_epoch, epochs + 1):
            training_model.train()
            train_sampler = loaders["train"].sampler
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
            total_loss = 0.0
            batches = 0
            progress = tqdm(
                loaders["train"],
                desc=f"Epoch {epoch}/{epochs}",
                unit="batch",
                dynamic_ncols=True,
                mininterval=0.5,
                file=sys.stdout,
                disable=(
                    not self.is_main_process
                    or not bool(self.train_cfg.get("progress_bar", True))
                ),
            )
            for batch in progress:
                batch.pop("names")
                batch.pop("smiles")
                batch.pop("original_indices")
                batch.pop("raw_targets")
                labels = batch.pop("labels").to(self.device).float()
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                logits = training_model(**inputs).logits
                labels = labels.reshape_as(logits)
                loss = self._masked_multitask_loss(
                    logits,
                    labels,
                    loss_weights_tensor,
                    task=self.task,
                    task_types=self.task_types,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.train_cfg.get("gradient_clip", 1.0)))
                optimizer.step()
                batch_loss = float(loss.detach())
                total_loss += batch_loss
                batches += 1
                if self.is_main_process:
                    progress.set_postfix(
                        loss=f"{batch_loss:.4f}",
                        mean=f"{total_loss / batches:.4f}",
                        encoder_lr=f"{encoder_lr:.1e}",
                        head_lr=f"{head_lr:.1e}",
                        refresh=False,
                    )

            loss_totals = torch.tensor(
                [total_loss, float(batches)],
                # MPS does not implement float64 tensors. Float32 is also
                # sufficient for cross-rank epoch-loss aggregation.
                dtype=torch.float32,
                device=self.device,
            )
            if self.distributed:
                dist.all_reduce(loss_totals, op=dist.ReduceOp.SUM)
                dist.barrier()
            mean_train_loss = float(
                loss_totals[0].item() / max(loss_totals[1].item(), 1.0)
            )

            should_stop = False
            if self.is_main_process:
                train_metrics, _ = self._evaluate(
                    model,
                    train_metrics_loader,
                    target_means,
                    target_stds,
                )
                val_metrics, _ = self._evaluate(
                    model, loaders["val"], target_means, target_stds
                )
                val_selection_metric = self._aggregate_metric(
                    val_metrics, selection_metric_name
                )
                row: dict[str, Any] = {
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    f"val_{selection_metric_name}": val_selection_metric,
                    "encoder_lr": encoder_lr,
                    "head_lr": head_lr,
                }
                task_history = [
                    previous
                    for previous in task_history
                    if int(previous["epoch"]) != epoch
                ]
                for split_name, split_metrics in (
                    ("train", train_metrics),
                    ("validation", val_metrics),
                ):
                    prefix = "train" if split_name == "train" else "val"
                    for target, task_metrics in split_metrics.items():
                        row.update(
                            {
                                f"{prefix}_{target}_{metric}": value
                                for metric, value in task_metrics.items()
                            }
                        )
                        task_history.append(
                            {
                                "epoch": epoch,
                                "split": split_name,
                                "task": target,
                                "num_labeled": int(task_metrics["n"]),
                                **{
                                    metric: value
                                    for metric, value in task_metrics.items()
                                    if metric != "n"
                                },
                            }
                        )
                    macro_row: dict[str, Any] = {
                        "epoch": epoch,
                        "split": split_name,
                        "task": "overall_macro",
                        "num_labeled": int(
                            sum(values["n"] for values in split_metrics.values())
                        ),
                    }
                    metric_names = tuple(sorted({
                        metric
                        for values in split_metrics.values()
                        for metric in values
                        if metric != "n"
                    }))
                    for metric_name in metric_names:
                        finite_values = [
                            values[metric_name]
                            for values in split_metrics.values()
                            if metric_name in values
                            and np.isfinite(values[metric_name])
                        ]
                        macro_row[metric_name] = (
                            float(np.mean(finite_values))
                            if finite_values
                            else float("nan")
                        )
                    task_history.append(macro_row)
                history.append(row)
                print(
                    f"Epoch {epoch}: loss={mean_train_loss:.6f}, "
                    f"mean_val_{selection_metric_name}={val_selection_metric:.6f}"
                )
                for split_name, split_metrics in (
                    ("Train", train_metrics),
                    ("Validation", val_metrics),
                ):
                    print(f"{split_name} metrics by task — epoch {epoch}:")
                    for target, values in split_metrics.items():
                        task_index = target_columns.index(target)
                        if self.task_types[task_index] == "classification":
                            print(
                                f"  {target} (n={int(values['n'])}): "
                                f"loss={values['loss']:.4f}, accuracy={values['accuracy']:.4f}, "
                                f"F1={values['f1']:.4f}, MCC={values['mcc']:.4f}, "
                                f"ROC-AUC={values['roc_auc']:.4f}, PR-AUC={values['pr_auc']:.4f}"
                            )
                        else:
                            print(
                                f"  {target} (n={int(values['n'])}): RMSE={values['rmse']:.4f}, "
                                f"MAE={values['mae']:.4f}, R2={values['r2']:.4f}, "
                                f"Pearson={values['pearson']:.4f}, Spearman={values['spearman']:.4f}"
                            )
                pd.DataFrame(history).to_csv(self.workdir / "training_history.csv", index=False)
                pd.DataFrame(task_history).sort_values(
                    ["epoch", "split", "task"]
                ).to_csv(task_history_path, index=False)
                if tensorboard_writer is not None:
                    for metric_name, value in row.items():
                        if metric_name == "epoch" or not isinstance(
                            value, (int, float, np.integer, np.floating)
                        ):
                            continue
                        if np.isfinite(float(value)):
                            tensorboard_writer.add_scalar(
                                metric_name, float(value), epoch
                            )
                    tensorboard_writer.flush()

                if val_selection_metric < best_metric:
                    best_metric = val_selection_metric
                    bad_epochs = 0
                    model.save_pretrained(best_dir)
                    (best_dir / "target_scaling.json").write_text(
                        json.dumps(
                            {
                                target: {"mean": float(mean), "std": float(std)}
                                for target, mean, std in zip(target_columns, target_means, target_stds)
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                else:
                    bad_epochs += 1
                self._save_resume_checkpoint(
                    last_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_metric,
                    bad_epochs=bad_epochs,
                    history=history,
                    target_means=target_means,
                    target_stds=target_stds,
                )
                print(f"Saved resume checkpoint: {last_checkpoint}")
                should_stop = bad_epochs >= patience

            control = torch.tensor(
                [best_metric, float(bad_epochs), float(should_stop)],
                # Keep synchronization compatible with CUDA, MPS, and CPU.
                dtype=torch.float32,
                device=self.device,
            )
            if self.distributed:
                dist.broadcast(control, src=0)
            best_metric = float(control[0].item())
            bad_epochs = int(control[1].item())
            should_stop = bool(control[2].item())
            if should_stop:
                if self.is_main_process:
                    print(f"Early stopping after {epoch} epochs.")
                break

        if self.distributed:
            dist.barrier()
        if not self.is_main_process:
            dist.barrier()
            dist.destroy_process_group()
            return

        best_model = self.GraphormerForGraphClassification.from_pretrained(best_dir).to(self.device)
        self._build_applicability_bundle(
            best_model,
            datasets["train"],
            datasets["val"],
            target_means,
            target_stds,
            best_dir,
        )
        scaling = {
            target: {"mean": float(mean), "std": float(std)}
            for target, mean, std in zip(target_columns, target_means, target_stds)
        }
        summary: dict[str, Any] = {
            "task": self.task,
            "task_types": dict(zip(target_columns, self.task_types)),
            f"best_mean_val_{selection_metric_name}": best_metric,
            "ddp_world_size": self.world_size,
            "target_scaling": scaling,
            "classification_threshold": (
                self.threshold if "classification" in self.task_types else None
            ),
            "targets": target_columns,
            "task_label_counts": {
                name: int(count)
                for name, count in zip(target_columns, task_label_counts)
            },
            "task_loss_weights": {
                name: float(weight)
                for name, weight in zip(target_columns, task_loss_weights)
            },
            "applicability_bundle": str(best_dir / "applicability.pt"),
            "calibration_summary": str(best_dir / "calibration.json"),
            "pretrained_weight_audit": self.pretrained_weight_audit,
        }
        if "test" in loaders:
            test_metrics, predictions = self._evaluate(
                best_model, loaders["test"], target_means, target_stds
            )
            predictions.to_csv(self.workdir / "test_predictions.csv", index=False)
            summary["test_metrics"] = test_metrics
        (self.workdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        resolved_config = copy.deepcopy(self.raw)
        resolved_config["pretrained_weight_audit"] = self.pretrained_weight_audit
        (self.workdir / "config.json").write_text(
            json.dumps(resolved_config, indent=2), encoding="utf-8"
        )
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        print(f"Saved Hugging Face Graphormer results to {self.workdir}")
        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()
