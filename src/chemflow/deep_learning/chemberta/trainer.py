"""Fine-tune a Hugging Face ChemBERTa encoder for molecular properties."""

from __future__ import annotations

import json
import math
import os
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from chemprop import data as chemprop_data
from rdkit import Chem
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score,
)
from scipy.stats import spearmanr
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from chemflow.deep_learning.task_weighting import (
    resolve_task_loss_weights,
    resolve_task_types,
)
from chemflow.deep_learning.chemeleon.applicability import (
    APPLICABILITY_VERSION, DEFAULT_CALIBRATION_CONFIDENCE,
    DEFAULT_LOCAL_MAX_SAMPLES, DEFAULT_LOCAL_MIN_EMBEDDING_SIMILARITY,
    DEFAULT_LOCAL_MIN_FP_SIMILARITY, DEFAULT_LOCAL_MIN_SAMPLES,
    DEFAULT_OOD_CONFIDENCE, DEFAULT_SIMILARITY_BITS, DEFAULT_SIMILARITY_RADIUS,
    MAX_EMBEDDING_DIMENSIONS, applicability_diagnostics,
    fit_embedding_projection, fit_multitask_validation_calibration,
    fit_ood_calibration, packed_morgan_fingerprints,
)


def _transformer_imports():
    try:
        from transformers import AutoModel, AutoTokenizer
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "ChemBERTa requires Transformers. Install it with "
            "`pip install -e '.[chemberta]'`."
        ) from error
    return AutoModel, AutoTokenizer


def _exception_chain_text(error: BaseException) -> str:
    """Return nested exception text, including Hugging Face transport causes."""
    messages: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return "\n".join(messages)


def _load_pretrained_components(
    model_name: str,
    *,
    local_files_only: bool,
    architecture: str,
    auto_model,
    auto_tokenizer,
):
    """Load a ChemBERTa checkpoint with an explicit legacy-RoBERTa path."""
    architecture = str(architecture).strip().lower()
    try:
        if architecture == "auto":
            tokenizer = auto_tokenizer.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            encoder = auto_model.from_pretrained(
                model_name, local_files_only=local_files_only
            )
        elif architecture == "roberta":
            # Older ChemBERTa repositories were published as RoBERTa models.
            # Concrete classes also avoid an opaque AutoConfig error when a
            # stale/incomplete Hub cache lacks its `model_type` field.
            from transformers import RobertaModel, RobertaTokenizerFast

            tokenizer = RobertaTokenizerFast.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            encoder = RobertaModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                add_pooling_layer=False,
            )
        else:
            raise ValueError(
                "ModelConfig.architecture must be 'roberta' or 'auto', "
                f"not {architecture!r}."
            )
    except (OSError, ValueError) as error:
        details = _exception_chain_text(error).lower()
        if "certificate_verify_failed" in details or "ssl" in details:
            reason = (
                "Python could not verify the Hugging Face TLS certificate. "
                "Use your organization CA bundle through REQUESTS_CA_BUNDLE, "
                "or install pip-system-certs and restart Python."
            )
        elif local_files_only:
            reason = "The local model directory/cache is incomplete."
        else:
            reason = "The model repository could not be downloaded or recognized."
        raise RuntimeError(
            f"Unable to load ChemBERTa model {model_name!r}. {reason} "
            "For offline use, download the complete repository on a machine "
            "with Hub access, set ModelConfig.model_name to that directory, "
            "and set local_files_only = true."
        ) from error
    return tokenizer, encoder


def _load_saved_encoder(model_directory: str | Path, architecture: str, auto_model):
    """Reload an encoder without constructing an unused RoBERTa pooler."""
    if str(architecture).strip().lower() == "roberta":
        from transformers import RobertaModel

        return RobertaModel.from_pretrained(
            model_directory,
            local_files_only=True,
            add_pooling_layer=False,
        )
    return auto_model.from_pretrained(model_directory, local_files_only=True)


def _select_device(requested: str) -> torch.device:
    name = str(requested).strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(name)


@dataclass
class Record:
    index: int
    name: str
    smiles: str
    targets: np.ndarray
    split: str | None


class SmilesDataset(Dataset):
    def __init__(self, records: list[Record], labels: np.ndarray) -> None:
        self.records = records
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "smiles": record.smiles,
            "name": record.name,
            "index": record.index,
            "raw_targets": record.targets,
            "labels": self.labels[index],
        }


class ChemBERTaPropertyModel(nn.Module):
    def __init__(self, encoder: nn.Module, n_tasks: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(hidden_size, int(n_tasks))
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, **tokens) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(**tokens).last_hidden_state
        # RoBERTa/ChemBERTa place the sequence representation at the first
        # (<s>) token, matching the standard sequence-classification head.
        pooled = encoded[:, 0, :]
        return self.head(self.dropout(pooled)), pooled


class ChemBERTaTrainer:
    """Single-task, multitask, classification, or mixed ChemBERTa trainer."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        with self.config_path.open("rb") as stream:
            self.raw = tomllib.load(stream)
        self.base = self.raw.get("BaseConfig", {})
        self.data_cfg = self.raw.get("DatasetConfig", {})
        self.model_cfg = self.raw.get("ModelConfig", {})
        self.train_cfg = self.raw.get("TrainingConfig", {})
        self.workdir = Path(self.base.get("workdir", "chemberta_run")).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.task = str(self.base.get("task", "regression")).lower()
        self.targets = self._target_columns()
        self.task_types = resolve_task_types(
            self.task, self.targets, self.data_cfg.get("task_types")
        )
        self.seed = int(self.train_cfg.get("seed", 42))
        self.device = _select_device(self.train_cfg.get("device", "auto"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        strategy = str(self.train_cfg.get("strategy", "auto")).lower()
        if strategy.startswith("ddp") and not self.distributed:
            raise RuntimeError("DDP requires launch through torchrun.")
        if self.distributed:
            if self.device.type != "cuda":
                raise RuntimeError("ChemBERTa DDP requires CUDA.")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
        process_seed = self.seed + self.rank
        random.seed(process_seed)
        np.random.seed(process_seed)
        torch.manual_seed(process_seed)
        self.AutoModel, self.AutoTokenizer = _transformer_imports()

    def _target_columns(self) -> list[str]:
        value = self.data_cfg.get("target_column")
        names = [value] if isinstance(value, str) else list(value or [])
        if not names or len(set(names)) != len(names):
            raise ValueError("DatasetConfig.target_column must contain unique targets.")
        return [str(name) for name in names]

    def _load_records(self) -> list[Record]:
        path = Path(self.data_cfg["dataset_path"]).expanduser().resolve()
        frame = pd.read_csv(path)
        smiles_column = str(self.data_cfg.get("smiles_column", "SMILES"))
        name_column = self.data_cfg.get("name_column")
        split_column = self.data_cfg.get("split_column")
        required = {smiles_column, *self.targets}
        if split_column:
            required.add(str(split_column))
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Dataset is missing required columns: {missing}")
        aliases = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
        records: list[Record] = []
        rejected: list[dict[str, Any]] = []
        for index, row in frame.iterrows():
            smiles = "" if pd.isna(row[smiles_column]) else str(row[smiles_column]).strip()
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            values = pd.to_numeric(row[self.targets], errors="coerce").to_numpy(float)
            if molecule is None or not np.isfinite(values).any():
                rejected.append({"index": index, "smiles": smiles})
                continue
            canonical = Chem.MolToSmiles(molecule, canonical=True)
            split = aliases.get(str(row[split_column]).strip().lower()) if split_column else None
            if split_column and split is None:
                rejected.append({"index": index, "smiles": smiles})
                continue
            name = str(row[name_column]) if name_column and pd.notna(row[name_column]) else str(index)
            records.append(Record(int(index), name, canonical, values, split))
        external_test = self.data_cfg.get("test_dataset_path")
        if external_test:
            if float(self.data_cfg.get("test_fraction", 0.0)) != 0.0:
                raise ValueError("test_dataset_path requires test_fraction = 0.0.")
            test_frame = pd.read_csv(Path(external_test).expanduser().resolve())
            missing_test = sorted({smiles_column, *self.targets} - set(test_frame.columns))
            if missing_test:
                raise KeyError(f"External test dataset is missing columns: {missing_test}")
            offset = len(frame)
            for position, row in test_frame.iterrows():
                smiles = "" if pd.isna(row[smiles_column]) else str(row[smiles_column]).strip()
                molecule = Chem.MolFromSmiles(smiles) if smiles else None
                values = pd.to_numeric(row[self.targets], errors="coerce").to_numpy(float)
                if molecule is None or not np.isfinite(values).any():
                    rejected.append({"index": f"test:{position}", "smiles": smiles})
                    continue
                name = str(row[name_column]) if name_column and name_column in test_frame and pd.notna(row[name_column]) else f"test:{position}"
                records.append(Record(
                    offset + int(position), name,
                    Chem.MolToSmiles(molecule, canonical=True), values, "test",
                ))
        if self.is_main_process:
            pd.DataFrame(rejected).to_csv(self.workdir / "rejected_rows.csv", index=False)
        if not records:
            raise ValueError("No valid labeled molecules remain.")
        return records

    def _split(self, records: list[Record]) -> dict[str, list[Record]]:
        if self.data_cfg.get("split_column"):
            splits = {name: [r for r in records if r.split == name] for name in ("train", "val", "test")}
        else:
            external_test = [record for record in records if record.split == "test"]
            split_pool = [record for record in records if record.split != "test"]
            val_fraction = float(self.data_cfg.get("val_fraction", 0.1))
            test_fraction = 0.0 if external_test else float(self.data_cfg.get("test_fraction", 0.1))
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
        rows = []
        for split, values in splits.items():
            for record in values:
                rows.append({"Name": record.name, "SMILES": record.smiles, **dict(zip(self.targets, record.targets)), "split": split})
        if self.is_main_process:
            pd.DataFrame(rows).to_csv(self.workdir / "data_splits.csv", index=False)
        return splits

    @staticmethod
    def _loss(logits, labels, weights, task_types):
        total = logits.new_tensor(0.0)
        denominator = logits.new_tensor(0.0)
        for index, task_type in enumerate(task_types):
            valid = torch.isfinite(labels[:, index])
            if not valid.any():
                continue
            if task_type == "classification":
                values = nn.functional.binary_cross_entropy_with_logits(
                    logits[valid, index], labels[valid, index], reduction="sum"
                )
            else:
                values = nn.functional.mse_loss(
                    logits[valid, index], labels[valid, index], reduction="sum"
                )
            total = total + values * weights[index]
            denominator = denominator + valid.sum() * weights[index]
        return total / denominator.clamp_min(1.0)

    def _collate(self, tokenizer):
        max_length = int(self.model_cfg.get("max_length", 512))
        def collate(items):
            tokens = tokenizer(
                [item["smiles"] for item in items], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            tokens["labels"] = torch.as_tensor(
                np.stack([item["labels"] for item in items]), dtype=torch.float32
            )
            tokens["raw_targets"] = np.stack([item["raw_targets"] for item in items])
            tokens["names"] = [item["name"] for item in items]
            tokens["smiles"] = [item["smiles"] for item in items]
            tokens["indices"] = [item["index"] for item in items]
            return tokens
        return collate

    def _evaluate(self, model, loader, means, stds) -> tuple[float, pd.DataFrame]:
        model.eval()
        rows = []
        losses = []
        with torch.inference_mode():
            for batch in loader:
                names, smiles = batch.pop("names"), batch.pop("smiles")
                batch.pop("indices", None)
                raw_targets = np.asarray(batch.pop("raw_targets"), dtype=float)
                labels = batch.pop("labels").to(self.device)
                tokens = {key: value.to(self.device) for key, value in batch.items()}
                logits, _ = model(**tokens)
                losses.append(float(self._loss(logits, labels, self.task_weights, self.task_types)))
                values = logits.float().cpu().numpy()
                for index, task_type in enumerate(self.task_types):
                    values[:, index] = (
                        1.0 / (1.0 + np.exp(-values[:, index]))
                        if task_type == "classification"
                        else values[:, index] * stds[index] + means[index]
                    )
                for row_index, (name, smi) in enumerate(zip(names, smiles)):
                    row = {"Name": name, "SMILES": smi}
                    for task_index, target in enumerate(self.targets):
                        row[f"{target}_true"] = raw_targets[row_index, task_index]
                        row[f"{target}_prediction"] = values[row_index, task_index]
                    rows.append(row)
        return float(np.mean(losses)), pd.DataFrame(rows)

    def _task_metrics(self, frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for target, task_type in zip(self.targets, self.task_types):
            observed = frame[f"{target}_true"].to_numpy(float)
            predicted = frame[f"{target}_prediction"].to_numpy(float)
            valid = np.isfinite(observed) & np.isfinite(predicted)
            y, p = observed[valid], predicted[valid]
            values: dict[str, float] = {"n": int(valid.sum())}
            if not len(y):
                result[target] = values
                continue
            if task_type == "regression":
                values.update({
                    "rmse": float(mean_squared_error(y, p) ** 0.5),
                    "mae": float(mean_absolute_error(y, p)),
                    "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
                    "pearson": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else float("nan"),
                    "spearman": float(spearmanr(y, p).statistic) if len(y) > 1 else float("nan"),
                })
            else:
                clipped = np.clip(p, 1e-7, 1 - 1e-7)
                classes = (clipped >= float(self.train_cfg.get("classification_threshold", 0.5))).astype(int)
                truth = y.astype(int)
                values.update({
                    "log_loss": float(-np.mean(truth * np.log(clipped) + (1-truth) * np.log(1-clipped))),
                    "accuracy": float(accuracy_score(truth, classes)),
                    "balanced_accuracy": float(balanced_accuracy_score(truth, classes)),
                    "precision": float(precision_score(truth, classes, zero_division=0)),
                    "recall": float(recall_score(truth, classes, zero_division=0)),
                    "f1": float(f1_score(truth, classes, zero_division=0)),
                    "mcc": float(matthews_corrcoef(truth, classes)),
                    "roc_auc": float(roc_auc_score(truth, clipped)) if np.unique(truth).size > 1 else float("nan"),
                    "pr_auc": float(average_precision_score(truth, clipped)) if np.unique(truth).size > 1 else float("nan"),
                })
            result[target] = values
        return result

    def _predict_reference(self, model, loader, means, stds) -> dict[str, Any]:
        model.eval()
        names, smiles, original_indices = [], [], []
        targets, predictions, embeddings = [], [], []
        with torch.inference_mode():
            for batch in loader:
                names.extend(batch.pop("names")); smiles.extend(batch.pop("smiles"))
                original_indices.extend(int(value) for value in batch.pop("indices"))
                targets.extend(np.asarray(batch.pop("raw_targets"), dtype=np.float32))
                batch.pop("labels", None)
                tokens = {key: value.to(self.device) for key, value in batch.items()}
                logits, pooled = model(**tokens)
                values = logits.float().cpu().numpy()
                for index, task_type in enumerate(self.task_types):
                    values[:, index] = (
                        1.0 / (1.0 + np.exp(-values[:, index]))
                        if task_type == "classification"
                        else values[:, index] * stds[index] + means[index]
                    )
                predictions.extend(values); embeddings.extend(pooled.float().cpu().numpy())
        return {
            "names": names, "smiles": smiles, "original_indices": original_indices,
            "targets": np.asarray(targets, dtype=np.float32),
            "predictions": np.asarray(predictions, dtype=np.float32),
            "embeddings": np.asarray(embeddings, dtype=np.float32),
        }

    @staticmethod
    def _partition(reference, projected, target_names, selected=None, predictions=None):
        if selected is None:
            selected = np.arange(len(reference["smiles"]))
        selected = np.asarray(selected, dtype=int)
        payload = {
            "original_index": [reference["original_indices"][i] for i in selected],
            "name": [reference["names"][i] for i in selected],
            "smiles": [reference["smiles"][i] for i in selected],
            "target_names": list(target_names),
            "targets": torch.from_numpy(reference["targets"][selected].astype(np.float32)),
            "embeddings": torch.from_numpy(projected[selected].astype(np.float16)),
        }
        if predictions is not None:
            payload["predictions"] = torch.from_numpy(np.asarray(predictions, dtype=np.float32))
        return payload

    def _build_applicability(self, model, train_loader, val_loader, means, stds, best_dir):
        train = self._predict_reference(model, train_loader, means, stds)
        validation = self._predict_reference(model, val_loader, means, stds)
        projection, train_projected, val_projected = fit_embedding_projection(
            train["embeddings"], validation["embeddings"], MAX_EMBEDDING_DIMENSIONS, self.seed
        )
        calibration = fit_multitask_validation_calibration(
            self.task_types, validation["targets"], validation["predictions"],
            self.targets, DEFAULT_CALIBRATION_CONFIDENCE,
        )
        payload = {
            "version": APPLICABILITY_VERSION, "model_family": "chemberta",
            "embedding_space": "fine_tuned_cls_token_pca", "projection": projection,
            "similarity": {"method": "morgan_tanimoto", "radius": DEFAULT_SIMILARITY_RADIUS, "bits": DEFAULT_SIMILARITY_BITS},
            "training": self._partition(train, train_projected, self.targets),
            "validation": self._partition(validation, val_projected, self.targets, predictions=validation["predictions"]),
            "calibration": calibration, "training_by_task": {}, "validation_by_task": {}, "ood_calibration": {},
            "local_calibration": {"method": "fingerprint_embedding_nearest_validation_residuals", "min_samples": DEFAULT_LOCAL_MIN_SAMPLES, "max_samples": DEFAULT_LOCAL_MAX_SAMPLES, "min_fp_similarity": DEFAULT_LOCAL_MIN_FP_SIMILARITY, "min_embedding_similarity": DEFAULT_LOCAL_MIN_EMBEDDING_SIMILARITY},
        }
        payload["training"]["fingerprints"] = torch.from_numpy(packed_morgan_fingerprints(payload["training"]["smiles"]))
        for task_index, target in enumerate(self.targets):
            train_selected = np.flatnonzero(np.isfinite(train["targets"][:, task_index]))
            val_selected = np.flatnonzero(np.isfinite(validation["targets"][:, task_index]))
            train_part = self._partition(train, train_projected, [target], train_selected)
            train_part["targets"] = train_part["targets"][:, task_index:task_index+1]
            train_part["fingerprints"] = torch.from_numpy(packed_morgan_fingerprints(train_part["smiles"]))
            payload["training_by_task"][target] = train_part
            val_part = self._partition(validation, val_projected, [target], val_selected, validation["predictions"][val_selected, task_index:task_index+1])
            val_part["targets"] = val_part["targets"][:, task_index:task_index+1]
            val_part["fingerprints"] = torch.from_numpy(packed_morgan_fingerprints(val_part["smiles"]))
            payload["validation_by_task"][target] = val_part
            diagnostics = applicability_diagnostics(
                val_part["smiles"], validation["embeddings"][val_selected], list(range(len(val_selected))), payload, reference_task=target,
            )
            payload["ood_calibration"][target] = fit_ood_calibration(
                diagnostics["train_max_tanimoto"], diagnostics["train_embedding_cosine_distance"], confidence=DEFAULT_OOD_CONFIDENCE,
            )
        torch.save(payload, best_dir / "applicability.pt")
        (best_dir / "calibration.json").write_text(json.dumps({"validation_calibration": calibration, "ood_calibration": payload["ood_calibration"]}, indent=2), encoding="utf-8")

    def train(self) -> None:
        records = self._load_records()
        splits = self._split(records)
        train_targets = np.asarray([r.targets for r in splits["train"]], dtype=float)
        for index, (target, task_type) in enumerate(zip(self.targets, self.task_types)):
            if task_type != "classification":
                continue
            labels = train_targets[:, index]
            labels = labels[np.isfinite(labels)]
            if not np.isin(labels, [0.0, 1.0]).all():
                raise ValueError(f"Classification target {target} must contain only 0/1 labels.")
            if np.unique(labels).size < 2:
                raise ValueError(f"Classification target {target} needs both classes in training.")
        means = np.zeros(len(self.targets), dtype=np.float32)
        stds = np.ones(len(self.targets), dtype=np.float32)
        for index, task_type in enumerate(self.task_types):
            if task_type == "regression":
                values = train_targets[:, index]
                means[index] = np.nanmean(values)
                stds[index] = max(float(np.nanstd(values)), 1e-8)
        labels_by_split = {}
        for split, values in splits.items():
            labels = np.asarray([r.targets for r in values], dtype=np.float32)
            for index, task_type in enumerate(self.task_types):
                if task_type == "regression":
                    labels[:, index] = (labels[:, index] - means[index]) / stds[index]
            labels_by_split[split] = labels
        counts, weights = resolve_task_loss_weights(
            train_targets, self.targets,
            strategy=self.train_cfg.get("task_loss_weighting", "uniform"),
            configured=self.train_cfg.get("task_loss_weights"),
        )
        self.task_weights = torch.as_tensor(weights, device=self.device)
        model_name = str(self.model_cfg.get("model_name", "DeepChem/ChemBERTa-77M-MLM"))
        local = bool(self.model_cfg.get("local_files_only", False))
        tokenizer, encoder = _load_pretrained_components(
            model_name,
            local_files_only=local,
            architecture=self.model_cfg.get("architecture", "roberta"),
            auto_model=self.AutoModel,
            auto_tokenizer=self.AutoTokenizer,
        )
        model = ChemBERTaPropertyModel(
            encoder, len(self.targets), float(self.model_cfg.get("dropout", 0.1))
        ).to(self.device)
        if bool(self.model_cfg.get("freeze_encoder", False)):
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
        if self.is_main_process:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            encoder_total = sum(p.numel() for p in model.encoder.parameters())
            encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
            head_total = sum(p.numel() for p in model.head.parameters())
            head_trainable = sum(
                p.numel() for p in model.head.parameters() if p.requires_grad
            )
            trainable_percent = 100.0 * trainable / max(total, 1)
            print(
                "ChemBERTa model summary:\n"
                f"  Total parameters:     {total:,}\n"
                f"  Trainable parameters: {trainable:,} ({trainable_percent:.2f}%)\n"
                f"  Frozen parameters:    {total - trainable:,}\n"
                f"  Encoder trainable:    {encoder_trainable:,}/{encoder_total:,}\n"
                f"  Head trainable:       {head_trainable:,}/{head_total:,}",
                flush=True,
            )
        collate = self._collate(tokenizer)
        samplers = {
            name: DistributedSampler(values, num_replicas=self.world_size, rank=self.rank, shuffle=name == "train", seed=self.seed)
            if self.distributed and name == "train" else None
            for name, values in splits.items() if values
        }
        loaders = {
            name: DataLoader(
                SmilesDataset(values, labels_by_split[name]),
                batch_size=int(self.train_cfg.get("batch_size", 32)), sampler=samplers[name],
                shuffle=name == "train" and samplers[name] is None,
                num_workers=int(self.train_cfg.get("num_workers", 0)),
                collate_fn=collate,
            )
            for name, values in splits.items() if values
        }
        reference_loaders = {
            name: DataLoader(
                SmilesDataset(values, labels_by_split[name]),
                batch_size=int(self.train_cfg.get("batch_size", 32)), shuffle=False,
                num_workers=int(self.train_cfg.get("num_workers", 0)), collate_fn=collate,
            )
            for name, values in splits.items() if values
        }
        if bool(self.train_cfg.get("applicability_only", False)):
            if self.distributed and not self.is_main_process:
                dist.barrier(); dist.destroy_process_group(); return
            best_dir = self.workdir / "best_model"
            if not (best_dir / "property_head.pt").is_file():
                raise FileNotFoundError(f"No saved ChemBERTa best model in {best_dir}")
            saved_encoder = _load_saved_encoder(
                best_dir,
                self.model_cfg.get("architecture", "roberta"),
                self.AutoModel,
            ).to(self.device)
            saved_model = ChemBERTaPropertyModel(
                saved_encoder, len(self.targets), float(self.model_cfg.get("dropout", 0.1))
            ).to(self.device)
            saved_model.head.load_state_dict(torch.load(
                best_dir / "property_head.pt", map_location=self.device, weights_only=True
            ))
            self._build_applicability(
                saved_model, reference_loaders["train"], reference_loaders["val"],
                means, stds, best_dir,
            )
            print(f"Rebuilt ChemBERTa applicability artifacts in {best_dir}")
            if self.distributed:
                dist.barrier(); dist.destroy_process_group()
            return
        encoder_lr = float(self.train_cfg.get(
            "encoder_learning_rate", self.train_cfg.get("learning_rate", 2e-5)
        ))
        head_lr = float(self.train_cfg.get("head_learning_rate", 1e-4))
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": encoder_lr},
                {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": head_lr},
            ],
            weight_decay=float(self.train_cfg.get("weight_decay", 0.01)),
        )
        epochs = int(self.train_cfg.get("num_epochs", 20))
        patience = int(self.train_cfg.get("early_stopping_patience", 5))
        best_loss, bad_epochs = math.inf, 0
        history = []
        best_dir = self.workdir / "best_model"
        checkpoint_dir = self.workdir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        last_path = Path(self.train_cfg.get("resume_checkpoint", checkpoint_dir / "last.pt")).expanduser().resolve()
        start_epoch = 1
        if bool(self.train_cfg.get("resume", False)):
            state = torch.load(last_path, map_location=self.device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"]) + 1
            best_loss = float(state["best_loss"]); bad_epochs = int(state["bad_epochs"])
            history = list(state.get("history", []))
            if not self.distributed:
                torch.set_rng_state(state.get("torch_rng", torch.get_rng_state()))
                np.random.set_state(state.get("numpy_rng", np.random.get_state()))
                random.setstate(state.get("python_rng", random.getstate()))
        if self.distributed:
            model = DistributedDataParallel(model, device_ids=[self.local_rank])
        for epoch in range(start_epoch, epochs + 1):
            if samplers.get("train") is not None:
                samplers["train"].set_epoch(epoch)
            model.train()
            train_losses = []
            progress = tqdm(
                loaders["train"], desc=f"Epoch {epoch}/{epochs}", unit="batch",
                disable=not self.is_main_process or not bool(self.train_cfg.get("progress_bar", True)),
            )
            for batch in progress:
                batch.pop("names"); batch.pop("smiles"); batch.pop("raw_targets"); batch.pop("indices")
                labels = batch.pop("labels").to(self.device)
                tokens = {key: value.to(self.device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(**tokens)
                loss = self._loss(logits, labels, self.task_weights, self.task_types)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.train_cfg.get("gradient_clip", 1.0)))
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
                progress.set_postfix(
                    loss=f"{train_losses[-1]:.4f}",
                    encoder_lr=f"{encoder_lr:.1e}", head_lr=f"{head_lr:.1e}",
                )
            loss_totals = torch.tensor(
                [float(np.sum(train_losses)), float(len(train_losses))],
                dtype=torch.float32, device=self.device,
            )
            if self.distributed:
                dist.all_reduce(loss_totals, op=dist.ReduceOp.SUM)
            global_train_loss = float(
                loss_totals[0].item() / max(loss_totals[1].item(), 1.0)
            )
            should_stop = False
            if self.is_main_process:
                plain_model = model.module if isinstance(model, DistributedDataParallel) else model
                val_loss, val_frame = self._evaluate(plain_model, reference_loaders["val"], means, stds)
                _, train_frame = self._evaluate(plain_model, reference_loaders["train"], means, stds)
                train_loss = global_train_loss
                train_metrics, val_metrics = self._task_metrics(train_frame), self._task_metrics(val_frame)
                row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                task_rows = []
                for split_name, metrics in (("train", train_metrics), ("validation", val_metrics)):
                    for target, values in metrics.items():
                        for metric, value in values.items():
                            if metric != "n": row[f"{split_name}_{target}_{metric}"] = value
                        task_rows.append({"epoch": epoch, "split": split_name, "task": target, **values})
                    metric_names = sorted({
                        key for values in metrics.values() for key in values if key != "n"
                    })
                    macro = {
                        metric: float(np.nanmean([
                            values.get(metric, np.nan) for values in metrics.values()
                        ]))
                        for metric in metric_names
                    }
                    task_rows.append({
                        "epoch": epoch, "split": split_name,
                        "task": "overall_macro",
                        "n": int(sum(values.get("n", 0) for values in metrics.values())),
                        **macro,
                    })
                history.append(row)
                pd.DataFrame(history).to_csv(self.workdir / "training_history.csv", index=False)
                previous = []
                task_path = self.workdir / "training_task_metrics.csv"
                if task_path.is_file():
                    previous = pd.read_csv(task_path).query("epoch != @epoch").to_dict("records")
                pd.DataFrame(previous + task_rows).to_csv(task_path, index=False)
                print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                for split_name, metrics in (("Train", train_metrics), ("Validation", val_metrics)):
                    print(f"{split_name} metrics: {metrics}")
                if val_loss < best_loss:
                    best_loss, bad_epochs = val_loss, 0
                    best_dir.mkdir(parents=True, exist_ok=True)
                    plain_model.encoder.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    torch.save(plain_model.head.state_dict(), best_dir / "property_head.pt")
                    (best_dir / "chemflow_config.json").write_text(json.dumps({
                        "targets": self.targets, "task": self.task, "task_types": self.task_types,
                        "means": means.tolist(), "stds": stds.tolist(),
                        "dropout": float(self.model_cfg.get("dropout", 0.1)),
                        "architecture": self.model_cfg.get("architecture", "roberta"),
                        "label_counts": dict(zip(self.targets, counts.tolist())),
                    }, indent=2), encoding="utf-8")
                    (best_dir / "target_scaling.json").write_text(json.dumps({
                        target: {"mean": float(mean), "std": float(std)}
                        for target, mean, std in zip(self.targets, means, stds)
                    }, indent=2), encoding="utf-8")
                else:
                    bad_epochs += 1
                torch.save({
                    "schema_version": 1, "epoch": epoch,
                    "model": plain_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss, "bad_epochs": bad_epochs, "history": history,
                    "means": means, "stds": stds, "task": self.task,
                    "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                }, last_path)
                should_stop = bad_epochs >= patience
            if self.distributed:
                control = torch.tensor([float(should_stop)], device=self.device)
                dist.broadcast(control, src=0)
                should_stop = bool(control.item())
            if should_stop:
                break
        if self.distributed:
            dist.barrier()
        if not self.is_main_process:
            dist.destroy_process_group()
            return
        encoder = _load_saved_encoder(
            best_dir,
            self.model_cfg.get("architecture", "roberta"),
            self.AutoModel,
        ).to(self.device)
        best_model = ChemBERTaPropertyModel(encoder, len(self.targets), float(self.model_cfg.get("dropout", 0.1))).to(self.device)
        best_model.head.load_state_dict(torch.load(best_dir / "property_head.pt", map_location=self.device, weights_only=True))
        self._build_applicability(best_model, reference_loaders["train"], reference_loaders["val"], means, stds, best_dir)
        summary = {"task": self.task, "task_types": dict(zip(self.targets, self.task_types)), "best_val_loss": best_loss, "ddp_world_size": self.world_size, "target_scaling": {target: {"mean": float(mean), "std": float(std)} for target, mean, std in zip(self.targets, means, stds)}, "applicability_bundle": str(best_dir / "applicability.pt")}
        if "test" in reference_loaders:
            _, predictions = self._evaluate(best_model, reference_loaders["test"], means, stds)
            predictions.to_csv(self.workdir / "test_predictions.csv", index=False)
            summary["test_metrics"] = self._task_metrics(predictions)
        (self.workdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.workdir / "config.json").write_text(json.dumps(self.raw, indent=2), encoding="utf-8")
        print(f"Saved ChemBERTa results to {self.workdir}")
        if self.distributed:
            dist.destroy_process_group()
