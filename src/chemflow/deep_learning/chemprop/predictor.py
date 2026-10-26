"""Lightweight inference for standard Chemprop v2 checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models
from rdkit import Chem


def _select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prediction_batch_size(dataset_size: int, requested: int) -> int:
    """Avoid Chemprop's drop-last edge case for a final one-row batch."""
    size = min(dataset_size, int(requested))
    if size < 1:
        raise ValueError("batch_size must be at least 1.")
    while size > 1 and dataset_size % size == 1:
        size -= 1
    return size


class ChempropCheckpointPredictor:
    """Load a Chemprop v2 checkpoint and retain invalid rows as NaN."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Chemprop checkpoint does not exist: {self.checkpoint_path}"
            )
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Expected the Chemprop checkpoint to contain a dictionary.")
        columns = checkpoint.get("output_columns")
        if not isinstance(columns, (list, tuple)) or not columns:
            raise KeyError("Chemprop checkpoint does not define output_columns.")
        self.output_columns = tuple(str(column) for column in columns)
        self.device = _select_device(device)
        self.model = models.MPNN.load_from_file(
            self.checkpoint_path, map_location="cpu"
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
            and float(module.p) > 0.0
        ]
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def predict_smiles(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        mc_dropout_samples: int = 0,
    ) -> pd.DataFrame:
        if mc_dropout_samples == 1 or mc_dropout_samples < 0:
            raise ValueError("mc_dropout_samples must be 0 or at least 2.")
        if mc_dropout_samples and not self.dropout_modules:
            raise ValueError(
                "MC dropout was requested, but the checkpoint has no active "
                "dropout layers."
            )
        values = [str(value).strip() for value in smiles]
        predictions = np.full(
            (len(values), len(self.output_columns)), np.nan, dtype=np.float32
        )
        mc_mean = np.full_like(predictions, np.nan)
        mc_std = np.full_like(predictions, np.nan)
        valid_indices: list[int] = []
        datapoints: list[data.MoleculeDatapoint] = []
        for index, value in enumerate(values):
            if value and Chem.MolFromSmiles(value) is not None:
                valid_indices.append(index)
                datapoints.append(data.MoleculeDatapoint.from_smi(value))

        if datapoints:
            dataset = data.MoleculeDataset(datapoints, self.featurizer)
            loader = data.build_dataloader(
                dataset,
                batch_size=_prediction_batch_size(len(dataset), batch_size),
                num_workers=int(num_workers),
                shuffle=False,
            )
            batches: list[torch.Tensor] = []
            mc_mean_batches: list[torch.Tensor] = []
            mc_std_batches: list[torch.Tensor] = []
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    batch = self.model.transfer_batch_to_device(
                        batch, self.device, 0
                    )
                    batch_values = self.model.predict_step(batch, batch_index)
                    if batch_values.ndim == 1:
                        batch_values = batch_values.reshape(-1, 1)
                    batches.append(batch_values.detach().float().cpu())
                    if mc_dropout_samples:
                        for module in self.dropout_modules:
                            module.train()
                        stochastic: list[torch.Tensor] = []
                        for _ in range(mc_dropout_samples):
                            sample = self.model.predict_step(batch, batch_index)
                            if sample.ndim == 1:
                                sample = sample.reshape(-1, 1)
                            stochastic.append(sample.detach().float().cpu())
                        self.model.eval()
                        stacked = torch.stack(stochastic)
                        mc_mean_batches.append(stacked.mean(dim=0))
                        mc_std_batches.append(stacked.std(dim=0, unbiased=True))
            valid_predictions = torch.cat(batches).numpy()
            expected = (len(valid_indices), len(self.output_columns))
            if valid_predictions.shape != expected:
                raise RuntimeError(
                    f"Unexpected Chemprop prediction shape {valid_predictions.shape}; "
                    f"expected {expected}."
                )
            predictions[np.asarray(valid_indices)] = valid_predictions
            if mc_mean_batches:
                mc_mean[np.asarray(valid_indices)] = torch.cat(mc_mean_batches).numpy()
                mc_std[np.asarray(valid_indices)] = torch.cat(mc_std_batches).numpy()

        result = pd.DataFrame(predictions, columns=self.output_columns)
        if mc_dropout_samples:
            for index, column in enumerate(self.output_columns):
                result[f"{column}__mc_mean"] = mc_mean[:, index]
                result[f"{column}__mc_std"] = mc_std[:, index]
                result[f"{column}__mc_lower_95"] = (
                    mc_mean[:, index] - 1.96 * mc_std[:, index]
                )
                result[f"{column}__mc_upper_95"] = (
                    mc_mean[:, index] + 1.96 * mc_std[:, index]
                )
        return result
