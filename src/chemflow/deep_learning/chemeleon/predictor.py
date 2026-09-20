"""Inference for single-task and multitask CheMeleon checkpoints."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from chemprop import data, featurizers, models
from rdkit import Chem

from chemflow.deep_learning.chemeleon.applicability import (
    DEFAULT_CALIBRATION_CONFIDENCE,
    DEFAULT_SIMILARITY_BITS,
    DEFAULT_SIMILARITY_RADIUS,
    applicability_diagnostics,
    apply_calibration,
    apply_local_calibration,
    apply_mc_dropout_intervals,
)
from chemflow.deep_learning.task_weighting import resolve_task_types


def _select_device(requested: str | None = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _complete_batch_size(dataset_size: int, requested_batch_size: int) -> int:
    """Avoid ChemProp's automatic drop-last behavior during prediction."""
    batch_size = min(int(requested_batch_size), int(dataset_size))
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    while batch_size > 1 and dataset_size % batch_size == 1:
        batch_size -= 1
    return batch_size


def _checkpoint_config(checkpoint_path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("chemflow_config")
    if isinstance(config, dict):
        return config

    config_path = checkpoint_path.parent.parent / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        if isinstance(config, dict):
            return config

    raise KeyError(
        "CheMeleon checkpoint has no ChemFlow configuration. Expected "
        "'chemflow_config' in the checkpoint or config.json two directories above it."
    )


class CheMeleonPredictor:
    """Load a ChemFlow CheMeleon checkpoint and predict SMILES."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        threshold: float = 0.5,
        applicability_domain: bool = True,
        embedding_dimensions: int | None = None,
        calibration_confidence: float = DEFAULT_CALIBRATION_CONFIDENCE,
        similarity_radius: int = DEFAULT_SIMILARITY_RADIUS,
        similarity_bits: int = DEFAULT_SIMILARITY_BITS,
        mc_dropout_samples: int = 0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"CheMeleon checkpoint does not exist: {self.checkpoint_path}"
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if embedding_dimensions is not None and int(embedding_dimensions) < 1:
            raise ValueError("embedding_dimensions must be at least 1.")
        if not 0.0 < float(calibration_confidence) < 1.0:
            raise ValueError("calibration_confidence must be between 0 and 1.")
        if int(similarity_radius) < 1:
            raise ValueError("similarity_radius must be at least 1.")
        if int(similarity_bits) < 64:
            raise ValueError("similarity_bits must be at least 64.")
        if int(mc_dropout_samples) not in (0, 1) and int(mc_dropout_samples) < 2:
            raise ValueError("mc_dropout_samples must be 0 or at least 2.")
        if int(mc_dropout_samples) == 1:
            raise ValueError("mc_dropout_samples=1 cannot estimate uncertainty.")

        self.device = _select_device(device)
        self.threshold = float(threshold)
        self.embedding_dimensions = embedding_dimensions
        self.calibration_confidence = float(calibration_confidence)
        self.similarity_radius = int(similarity_radius)
        self.similarity_bits = int(similarity_bits)
        self.mc_dropout_samples = int(mc_dropout_samples)
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Expected the CheMeleon checkpoint to contain a dictionary.")
        self.config = _checkpoint_config(self.checkpoint_path, checkpoint)
        self.applicability = (
            checkpoint.get("chemflow_applicability")
            if applicability_domain
            else None
        )

        base_config = self.config.get("BaseConfig", {})
        dataset_config = self.config.get("DatasetConfig", {})
        self.task = str(base_config.get("task", "regression")).strip().lower()
        raw_target_names = dataset_config.get("target_column", "prediction")
        self.target_names = (
            [raw_target_names]
            if isinstance(raw_target_names, str)
            else [str(name) for name in raw_target_names]
        )
        if not self.target_names:
            raise ValueError("The checkpoint configuration contains no target names.")
        self.task_types = resolve_task_types(
            self.task, self.target_names, dataset_config.get("task_types")
        )

        # ChemProp's direct loader reconstructs all submodules without going
        # through Lightning's PyTorch 2.6+ weights-only checkpoint default.
        self.model = models.MPNN.load_from_file(
            self.checkpoint_path,
            map_location="cpu",
        )
        self.model.to(self.device)
        self.model.eval()
        self.dropout_modules = [
            module
            for module in self.model.modules()
            if isinstance(
                module,
                (
                    torch.nn.Dropout,
                    torch.nn.Dropout1d,
                    torch.nn.Dropout2d,
                    torch.nn.Dropout3d,
                    torch.nn.AlphaDropout,
                ),
            )
        ]
        if self.mc_dropout_samples and not any(
            float(getattr(module, "p", 0.0)) > 0.0
            for module in self.dropout_modules
        ):
            warnings.warn(
                "MC dropout was requested, but this checkpoint contains no "
                "dropout layer with p > 0. MC dropout has been disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.mc_dropout_samples = 0
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.invalid_indices: list[int] = []

    def predict_smiles(
        self,
        smiles_list: list[str],
        batch_size: int = 64,
        num_workers: int = 0,
        task_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        names = self.target_names if task_names is None else list(task_names)
        if len(names) != len(self.target_names):
            raise ValueError(
                f"Expected {len(self.target_names)} task names, got {len(names)}."
            )
        if any(not str(name).strip() for name in names):
            raise ValueError("Task names cannot be empty.")
        if len(set(names)) != len(names):
            raise ValueError(f"Task names must be unique: {names}")

        valid_indices: list[int] = []
        datapoints: list[data.MoleculeDatapoint] = []
        self.invalid_indices = []
        for index, value in enumerate(smiles_list):
            smiles = str(value).strip()
            if not smiles or Chem.MolFromSmiles(smiles) is None:
                self.invalid_indices.append(index)
                continue
            valid_indices.append(index)
            datapoints.append(data.MoleculeDatapoint.from_smi(smiles))

        predictions = np.full(
            (len(smiles_list), len(names)), np.nan, dtype=np.float32
        )
        valid_embeddings = np.empty((0, 0), dtype=np.float32)
        valid_mc_mean: np.ndarray | None = None
        valid_mc_std: np.ndarray | None = None
        if datapoints:
            dataset = data.MoleculeDataset(datapoints, self.featurizer)
            loader = data.build_dataloader(
                dataset,
                batch_size=_complete_batch_size(len(dataset), batch_size),
                num_workers=int(num_workers),
                shuffle=False,
            )
            batches: list[torch.Tensor] = []
            embedding_batches: list[torch.Tensor] = []
            mc_batches: list[torch.Tensor] = []
            mc_std_batches: list[torch.Tensor] = []
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    batch = self.model.transfer_batch_to_device(
                        batch, self.device, 0
                    )
                    values = self.model.predict_step(batch, batch_index).detach().cpu()
                    if values.ndim == 1:
                        values = values.reshape(-1, 1)
                    batches.append(values)
                    bmg, atom_descriptors, extra_descriptors = batch[:3]
                    embedding_batches.append(
                        self.model.fingerprint(
                            bmg, atom_descriptors, extra_descriptors
                        ).detach().float().cpu()
                    )
                    if self.mc_dropout_samples:
                        for module in self.dropout_modules:
                            module.train()
                        stochastic = []
                        for _ in range(self.mc_dropout_samples):
                            stochastic_value = self.model.predict_step(
                                batch, batch_index
                            ).detach().float().cpu()
                            if stochastic_value.ndim == 1:
                                stochastic_value = stochastic_value.reshape(-1, 1)
                            stochastic.append(stochastic_value)
                        self.model.eval()
                        stacked = torch.stack(stochastic, dim=0)
                        mc_batches.append(stacked.mean(dim=0))
                        mc_std_batches.append(stacked.std(dim=0, unbiased=True))
            valid_predictions = torch.cat(batches, dim=0).numpy()
            valid_embeddings = torch.cat(embedding_batches, dim=0).numpy()
            if mc_batches:
                valid_mc_mean = torch.cat(mc_batches, dim=0).numpy()
                valid_mc_std = torch.cat(mc_std_batches, dim=0).numpy()
                valid_predictions = valid_mc_mean
            if valid_predictions.shape != (len(valid_indices), len(names)):
                raise RuntimeError(
                    "Unexpected CheMeleon prediction shape: "
                    f"{valid_predictions.shape}; expected {(len(valid_indices), len(names))}."
                )
            predictions[np.asarray(valid_indices)] = valid_predictions

        output: dict[str, np.ndarray] = {}
        for task_index, name in enumerate(names):
            values = predictions[:, task_index]
            if valid_mc_mean is not None and valid_mc_std is not None:
                mc_mean = np.full(len(smiles_list), np.nan, dtype=np.float32)
                mc_std = np.full(len(smiles_list), np.nan, dtype=np.float32)
                mc_mean[np.asarray(valid_indices)] = valid_mc_mean[:, task_index]
                mc_std[np.asarray(valid_indices)] = valid_mc_std[:, task_index]
                output[f"mc_mean_{name}"] = mc_mean
                output[f"mc_std_{name}"] = mc_std
            if self.task_types[task_index] == "regression":
                output[name] = values
            else:
                output[f"prob_{name}"] = values
                classes = np.full(len(values), np.nan, dtype=np.float32)
                finite = np.isfinite(values)
                classes[finite] = (values[finite] >= self.threshold).astype(np.float32)
                output[f"class_{name}"] = classes
        calibration = (
            self.applicability.get("calibration")
            if isinstance(self.applicability, dict)
            else None
        )
        for original_name, output_name, task_type in zip(
            self.target_names, names, self.task_types
        ):
            apply_calibration(
                output,
                calibration,
                [original_name],
                [output_name],
                task_type,
                self.threshold,
                self.calibration_confidence,
            )
        task_partitions = (
            self.applicability.get("training_by_task", {})
            if isinstance(self.applicability, dict)
            else {}
        )
        if task_partitions:
            for task_name in self.target_names:
                task_diagnostics = applicability_diagnostics(
                    smiles_list,
                    valid_embeddings,
                    valid_indices,
                    self.applicability,
                    embedding_dimensions=self.embedding_dimensions,
                    similarity_radius=self.similarity_radius,
                    similarity_bits=self.similarity_bits,
                    reference_task=task_name,
                )
                output.update(
                    {
                        f"{task_name}_{name}": values
                        for name, values in task_diagnostics.items()
                    }
                )
        else:
            output.update(applicability_diagnostics(
                smiles_list,
                valid_embeddings,
                valid_indices,
                self.applicability,
                embedding_dimensions=self.embedding_dimensions,
                similarity_radius=self.similarity_radius,
                similarity_bits=self.similarity_bits,
            ))
        regression_indices = [
            index for index, value in enumerate(self.task_types)
            if value == "regression"
        ]
        if regression_indices:
            apply_local_calibration(
                output,
                [self.target_names[index] for index in regression_indices],
                [names[index] for index in regression_indices],
                self.calibration_confidence,
            )
            apply_mc_dropout_intervals(
                output,
                [names[index] for index in regression_indices],
                self.calibration_confidence,
            )
        return output
