"""Inference, calibration, and uncertainty for ChemFlow Graphormer models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem
from scipy.special import expit

from chemflow.deep_learning.chemeleon.applicability import (
    DEFAULT_CALIBRATION_CONFIDENCE,
    DEFAULT_SIMILARITY_BITS,
    DEFAULT_SIMILARITY_RADIUS,
    applicability_diagnostics,
    apply_local_calibration,
    apply_mc_dropout_intervals,
)
from chemflow.deep_learning.hf_graphormer.featurizer import GraphormerFeaturizer
from chemflow.deep_learning.hf_graphormer.trainer import _graphormer_imports


def _select_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _enable_stochastic_dropout(model: torch.nn.Module) -> int:
    """Enable inference-time stochastic layers without updating parameters."""
    model.train()
    for module in model.modules():
        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm,
        ):
            module.eval()
    return sum(
        1
        for module in model.modules()
        if isinstance(module, torch.nn.Dropout)
        and float(module.p) > 0.0
    )


class HuggingFaceGraphormerPredictor:
    """Load a saved Graphormer ``best_model`` directory and predict SMILES."""

    def __init__(
        self,
        model_directory: str | Path,
        *,
        device: str | None = None,
        threshold: float | None = None,
        applicability_domain: bool = True,
        embedding_dimensions: int | None = None,
        calibration_confidence: float = DEFAULT_CALIBRATION_CONFIDENCE,
        similarity_radius: int = DEFAULT_SIMILARITY_RADIUS,
        similarity_bits: int = DEFAULT_SIMILARITY_BITS,
        mc_dropout_samples: int = 0,
    ) -> None:
        self.model_directory = Path(model_directory).expanduser().resolve()
        if not self.model_directory.is_dir():
            raise FileNotFoundError(
                f"Graphormer model directory does not exist: {self.model_directory}"
            )
        if int(mc_dropout_samples) == 1 or int(mc_dropout_samples) < 0:
            raise ValueError("mc_dropout_samples must be 0 or at least 2.")
        if not 0.0 < float(calibration_confidence) < 1.0:
            raise ValueError("calibration_confidence must be between 0 and 1.")
        if embedding_dimensions is not None and int(embedding_dimensions) < 1:
            raise ValueError("embedding_dimensions must be at least 1.")
        if int(similarity_radius) < 1:
            raise ValueError("similarity_radius must be at least 1.")
        if int(similarity_bits) < 64:
            raise ValueError("similarity_bits must be at least 64.")

        (
            _,
            graphormer_model,
            graphormer_collator,
            self.preprocess_item,
        ) = _graphormer_imports()
        self.device = _select_device(device)
        self.model = graphormer_model.from_pretrained(
            self.model_directory,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.collator = graphormer_collator(
            spatial_pos_max=int(self.model.config.spatial_pos_max),
            on_the_fly_processing=False,
        )
        run_config: dict[str, Any] = {}
        run_config_path = self.model_directory.parent / "config.json"
        if run_config_path.is_file():
            loaded_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            if isinstance(loaded_config, dict):
                run_config = loaded_config
        dataset_config = run_config.get("DatasetConfig", {})
        self.featurizer = GraphormerFeaturizer(
            remove_hs=bool(dataset_config.get("remove_hs", True)),
            reorder_atoms=bool(dataset_config.get("reorder_atoms", False)),
        )
        self.max_nodes = int(getattr(self.model.config, "max_nodes", 512))
        self.target_names = list(
            getattr(self.model.config, "chemflow_target_names", []) or []
        )
        if not self.target_names:
            self.target_names = [
                f"target_{index + 1}"
                for index in range(int(self.model.config.num_classes))
            ]
        if len(self.target_names) != int(self.model.config.num_classes):
            raise ValueError(
                "Graphormer checkpoint target metadata does not match num_classes."
            )
        self.task_types = list(
            getattr(self.model.config, "chemflow_task_types", []) or []
        )
        if not self.task_types:
            task = str(getattr(self.model.config, "chemflow_task", "regression"))
            self.task_types = [task] * len(self.target_names)
        if len(self.task_types) != len(self.target_names):
            raise ValueError(
                "Graphormer checkpoint task-type metadata does not match its targets."
            )
        configured_threshold = float(
            getattr(self.model.config, "chemflow_classification_threshold", 0.5)
        )
        self.threshold = configured_threshold if threshold is None else float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        scaling_path = self.model_directory / "target_scaling.json"
        if not scaling_path.is_file():
            raise FileNotFoundError(
                f"Graphormer target scaling file is missing: {scaling_path}"
            )
        scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
        self.means = np.asarray(
            [float(scaling[name]["mean"]) for name in self.target_names],
            dtype=np.float32,
        )
        self.stds = np.asarray(
            [float(scaling[name]["std"]) for name in self.target_names],
            dtype=np.float32,
        )

        applicability_path = self.model_directory / "applicability.pt"
        self.applicability: dict[str, Any] | None = None
        if applicability_domain:
            if not applicability_path.is_file():
                raise FileNotFoundError(
                    "Graphormer applicability/calibration bundle is missing: "
                    f"{applicability_path}. Rebuild it with applicability_only=true."
                )
            self.applicability = torch.load(
                applicability_path,
                map_location="cpu",
                weights_only=False,
            )
        self.embedding_dimensions = embedding_dimensions
        self.calibration_confidence = float(calibration_confidence)
        self.similarity_radius = int(similarity_radius)
        self.similarity_bits = int(similarity_bits)
        self.mc_dropout_samples = int(mc_dropout_samples)
        self.invalid_indices: list[int] = []

    def _prepare_item(self, smiles: str, index: int) -> dict[str, Any] | None:
        if not smiles or Chem.MolFromSmiles(smiles) is None:
            return None
        raw = self.featurizer.smiles2graph(smiles)
        if int(raw["num_nodes"]) > self.max_nodes:
            return None
        item = {
            "edge_index": np.asarray(raw["edge_index"], dtype=np.int64),
            "edge_attr": np.asarray(raw["edge_attr"], dtype=np.int64),
            "node_feat": np.asarray(raw["node_feat"], dtype=np.int64),
            "num_nodes": int(raw["num_nodes"]),
            "y": np.zeros(len(self.target_names), dtype=np.float32),
            "idx": int(index),
        }
        return self.preprocess_item(item)

    def _transform(self, logits: torch.Tensor) -> np.ndarray:
        raw = logits.detach().float().cpu().numpy().reshape(-1, len(self.target_names))
        values = raw * self.stds + self.means
        for index, task_type in enumerate(self.task_types):
            if task_type == "classification":
                values[:, index] = expit(raw[:, index])
        return values.astype(np.float32)

    def predict_smiles(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int = 16,
        task_names: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        smiles_list = [str(value).strip() for value in smiles]
        names = list(task_names) if task_names is not None else list(self.target_names)
        if len(names) != len(self.target_names):
            raise ValueError(
                f"Expected {len(self.target_names)} task names, received {len(names)}."
            )
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1.")

        valid_indices: list[int] = []
        items: list[dict[str, Any]] = []
        self.invalid_indices = []
        for index, value in enumerate(smiles_list):
            item = self._prepare_item(value, index)
            if item is None:
                self.invalid_indices.append(index)
            else:
                valid_indices.append(index)
                items.append(item)

        size = len(smiles_list)
        predictions = np.full(
            (size, len(names)), np.nan, dtype=np.float32
        )
        direct_predictions = np.full_like(predictions, np.nan)
        mc_standard_deviations = np.full_like(predictions, np.nan)
        valid_embeddings: list[np.ndarray] = []
        for start in range(0, len(items), int(batch_size)):
            batch_items = items[start : start + int(batch_size)]
            batch_indices = valid_indices[start : start + int(batch_size)]
            batch = self.collator(batch_items)
            batch.pop("labels", None)
            inputs = {key: value.to(self.device) for key, value in batch.items()}

            self.model.eval()
            with torch.no_grad():
                encoder_output = self.model.encoder(**inputs, return_dict=True)
                hidden = encoder_output["last_hidden_state"]
                deterministic_logits = self.model.classifier(hidden)[:, 0, :]
                valid_embeddings.extend(
                    hidden[:, 0, :].detach().float().cpu().numpy()
                )
                direct_batch = self._transform(deterministic_logits)
                direct_predictions[np.asarray(batch_indices)] = direct_batch

            if self.mc_dropout_samples:
                dropout_count = _enable_stochastic_dropout(self.model)
                if dropout_count == 0:
                    self.model.eval()
                    raise RuntimeError(
                        "MC dropout was requested, but the Graphormer checkpoint "
                        "contains no active torch.nn.Dropout modules."
                    )
                draws = []
                with torch.no_grad():
                    for _ in range(self.mc_dropout_samples):
                        draws.append(self._transform(self.model(**inputs).logits))
                self.model.eval()
                stacked = np.stack(draws, axis=0)
                batch_predictions = stacked.mean(axis=0)
                batch_standard_deviations = stacked.std(axis=0, ddof=1)
                mc_standard_deviations[np.asarray(batch_indices)] = (
                    batch_standard_deviations
                )
            else:
                batch_predictions = self._transform(deterministic_logits)
            predictions[np.asarray(batch_indices)] = batch_predictions

        output: dict[str, np.ndarray] = {}
        for task_index, name in enumerate(names):
            values = predictions[:, task_index]
            output[f"direct_{name}"] = direct_predictions[:, task_index]
            if self.mc_dropout_samples:
                output[f"mc_mean_{name}"] = values.copy()
                output[f"mc_std_{name}"] = mc_standard_deviations[:, task_index]
            if self.task_types[task_index] == "classification":
                output[name] = values
                output[f"prob_{name}"] = values
                classes = np.full(size, np.nan, dtype=np.float32)
                finite = np.isfinite(values)
                classes[finite] = (values[finite] >= self.threshold).astype(np.float32)
                output[f"class_{name}"] = classes
            else:
                output[name] = values

        if self.applicability and valid_indices:
            embeddings = np.asarray(valid_embeddings, dtype=np.float32)
            task_partitions = self.applicability.get("training_by_task", {})
            if task_partitions:
                for target_name in self.target_names:
                    diagnostics = applicability_diagnostics(
                        smiles_list,
                        embeddings,
                        valid_indices,
                        self.applicability,
                        embedding_dimensions=self.embedding_dimensions,
                        similarity_radius=self.similarity_radius,
                        similarity_bits=self.similarity_bits,
                        reference_task=target_name,
                    )
                    output.update(
                        {
                            f"{target_name}_{key}": value
                            for key, value in diagnostics.items()
                        }
                    )
            else:
                output.update(
                    applicability_diagnostics(
                        smiles_list,
                        embeddings,
                        valid_indices,
                        self.applicability,
                        embedding_dimensions=self.embedding_dimensions,
                        similarity_radius=self.similarity_radius,
                        similarity_bits=self.similarity_bits,
                    )
                )

        if names:
            bounded_names = [
                names[index]
                for index, task_type in enumerate(self.task_types)
                if task_type == "classification"
            ]
            apply_local_calibration(
                output,
                self.target_names,
                names,
                self.calibration_confidence,
                bounded_names,
            )
            apply_mc_dropout_intervals(
                output,
                names,
                self.calibration_confidence,
                bounded_names,
            )
        return output
