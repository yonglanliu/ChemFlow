"""Inference for single-task and multitask CheMeleon checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from chemprop import data, featurizers, models
from rdkit import Chem


def _select_device(requested: str | None = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"CheMeleon checkpoint does not exist: {self.checkpoint_path}"
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        self.device = _select_device(device)
        self.threshold = float(threshold)
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Expected the CheMeleon checkpoint to contain a dictionary.")
        self.config = _checkpoint_config(self.checkpoint_path, checkpoint)

        base_config = self.config.get("BaseConfig", {})
        dataset_config = self.config.get("DatasetConfig", {})
        self.task = str(base_config.get("task", "regression")).strip().lower()
        if self.task not in {"regression", "classification"}:
            raise ValueError(f"Unsupported CheMeleon task in checkpoint: {self.task!r}")

        raw_target_names = dataset_config.get("target_column", "prediction")
        self.target_names = (
            [raw_target_names]
            if isinstance(raw_target_names, str)
            else [str(name) for name in raw_target_names]
        )
        if not self.target_names:
            raise ValueError("The checkpoint configuration contains no target names.")

        # ChemProp's direct loader reconstructs all submodules without going
        # through Lightning's PyTorch 2.6+ weights-only checkpoint default.
        self.model = models.MPNN.load_from_file(
            self.checkpoint_path,
            map_location="cpu",
        )
        self.model.to(self.device)
        self.model.eval()
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
        if datapoints:
            dataset = data.MoleculeDataset(datapoints, self.featurizer)
            loader = data.build_dataloader(
                dataset,
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                shuffle=False,
            )
            batches: list[torch.Tensor] = []
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    batch = self.model.transfer_batch_to_device(
                        batch, self.device, 0
                    )
                    values = self.model.predict_step(batch, batch_index).detach().cpu()
                    if values.ndim == 1:
                        values = values.reshape(-1, 1)
                    batches.append(values)
            valid_predictions = torch.cat(batches, dim=0).numpy()
            if valid_predictions.shape != (len(valid_indices), len(names)):
                raise RuntimeError(
                    "Unexpected CheMeleon prediction shape: "
                    f"{valid_predictions.shape}; expected {(len(valid_indices), len(names))}."
                )
            predictions[np.asarray(valid_indices)] = valid_predictions

        output: dict[str, np.ndarray] = {}
        for task_index, name in enumerate(names):
            values = predictions[:, task_index]
            if self.task == "regression":
                output[name] = values
            else:
                output[f"prob_{name}"] = values
                classes = np.full(len(values), np.nan, dtype=np.float32)
                finite = np.isfinite(values)
                classes[finite] = (values[finite] >= self.threshold).astype(np.float32)
                output[f"class_{name}"] = classes
        return output
