"""Deployment wrapper for the CheMeleon clearance model ensemble."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from chemflow.deep_learning.chemeleon.predictor import CheMeleonPredictor


LOG_OUTPUT_COLUMNS = {
    "HLM": "Log_HLM_CLint_prediction",
    "RLM": "Log_RLM_CLint_prediction",
    "MLM": "Log_MLM_CLint_prediction",
}
RAW_OUTPUT_COLUMNS = {
    "HLM": "HLM_CLint_prediction (mL/min/kg)",
    "RLM": "RLM_CLint_prediction (mL/min/kg)",
    "MLM": "MLM_CLint_prediction (mL/min/kg)",
}


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _resolve_task_name(target_names: Sequence[str], species: str) -> str:
    marker = species.lower()
    matches = [name for name in target_names if marker in _normalized_name(name)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {species} target in checkpoint tasks "
            f"{list(target_names)}, found {matches}."
        )
    return str(matches[0])


def _inverse_log10(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    with np.errstate(over="ignore", invalid="ignore"):
        result[finite] = np.power(10.0, values[finite])
    return result


class ADMEModel(nn.Module):
    """Combine multitask HLM/RLM and single-task MLM CheMeleon models.

    This is an inference-only ensemble. ``forward`` accepts SMILES strings
    rather than a tensor because each CheMeleon model performs its own molecular
    graph featurization before executing its registered PyTorch module.
    """

    def __init__(
        self,
        multitask_checkpoint: str | Path | None,
        mlm_single_checkpoint: str | Path | None,
        *,
        device: str | None = None,
        applicability_domain: bool = True,
        embedding_dimensions: int | None = None,
        calibration_confidence: float = 0.90,
        similarity_radius: int = 2,
        similarity_bits: int = 2048,
        mc_dropout_samples: int = 0,
    ) -> None:
        super().__init__()
        if multitask_checkpoint is None and mlm_single_checkpoint is None:
            raise ValueError("At least one ADMET checkpoint is required.")

        self.multitask_predictor = (
            CheMeleonPredictor(
                checkpoint_path=multitask_checkpoint,
                device=device,
                applicability_domain=applicability_domain,
                embedding_dimensions=embedding_dimensions,
                calibration_confidence=calibration_confidence,
                similarity_radius=similarity_radius,
                similarity_bits=similarity_bits,
                mc_dropout_samples=mc_dropout_samples,
            )
            if multitask_checkpoint is not None
            else None
        )
        self.mlm_single_predictor = (
            CheMeleonPredictor(
                checkpoint_path=mlm_single_checkpoint,
                device=device,
                applicability_domain=applicability_domain,
                embedding_dimensions=embedding_dimensions,
                calibration_confidence=calibration_confidence,
                similarity_radius=similarity_radius,
                similarity_bits=similarity_bits,
                mc_dropout_samples=mc_dropout_samples,
            )
            if mlm_single_checkpoint is not None
            else None
        )

        if self.multitask_predictor and self.multitask_predictor.task != "regression":
            raise ValueError("The HLM/RLM checkpoint must be a regression model.")
        if self.mlm_single_predictor and self.mlm_single_predictor.task != "regression":
            raise ValueError("The MLM checkpoint must be a regression model.")

        self._hlm_task = (
            _resolve_task_name(self.multitask_predictor.target_names, "HLM")
            if self.multitask_predictor
            else None
        )
        self._rlm_task = (
            _resolve_task_name(self.multitask_predictor.target_names, "RLM")
            if self.multitask_predictor
            else None
        )
        self._mlm_task = (
            _resolve_task_name(self.mlm_single_predictor.target_names, "MLM")
            if self.mlm_single_predictor
            else None
        )
        self.available_properties = {
            species
            for species, available in {
                "HLM": self.multitask_predictor is not None,
                "RLM": self.multitask_predictor is not None,
                "MLM": self.mlm_single_predictor is not None,
            }.items()
            if available
        }

        # Register the underlying torch modules on this ensemble. Prediction is
        # delegated to the predictors so ChemProp graph batching remains intact.
        self.multitask = (
            self.multitask_predictor.model if self.multitask_predictor else None
        )
        self.mlm_single = (
            self.mlm_single_predictor.model if self.mlm_single_predictor else None
        )
        self.eval()

    @torch.inference_mode()
    def forward(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        properties: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        smiles_list = [str(value).strip() for value in smiles]
        requested = (
            ["HLM", "RLM", "MLM"]
            if properties is None
            else [str(value).strip().upper() for value in properties]
        )
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Select one or more unique ADMET properties.")
        unavailable = sorted(set(requested).difference(self.available_properties))
        if unavailable:
            raise ValueError(
                f"No loaded checkpoint provides these properties: {unavailable}."
            )

        log_predictions: dict[str, np.ndarray] = {}
        calibrated_predictions: dict[str, np.ndarray] = {}
        calibrated_intervals: dict[str, dict[str, np.ndarray]] = {}
        diagnostics: dict[str, np.ndarray] = {}
        multitask_values: dict[str, np.ndarray] = {}
        mlm_values: dict[str, np.ndarray] = {}

        def collect_diagnostics(
            species: str, values: dict[str, np.ndarray], task_name: str
        ) -> None:
            task_prefix = f"{task_name}_"
            task_specific = {
                f"{species}_{name[len(task_prefix):]}": np.asarray(data)
                for name, data in values.items()
                if name.startswith(task_prefix)
                and name[len(task_prefix):].startswith(
                    ("train_", "ood_", "local_")
                )
            }
            if task_specific:
                diagnostics.update(task_specific)
                return
            diagnostics.update(
                {
                    f"{species}_{name}": np.asarray(data)
                    for name, data in values.items()
                    if name.startswith("train_")
                }
            )

        if {"HLM", "RLM"}.intersection(requested):
            assert self.multitask_predictor is not None
            multitask_values = self.multitask_predictor.predict_smiles(
                smiles_list,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            if "HLM" in requested:
                assert self._hlm_task is not None
                log_predictions["HLM"] = np.asarray(
                    multitask_values[self._hlm_task], dtype=np.float64
                )
                calibrated_key = f"calibrated_{self._hlm_task}"
                if calibrated_key in multitask_values:
                    calibrated_predictions["HLM"] = np.asarray(
                        multitask_values[calibrated_key], dtype=np.float64
                    )
                collect_diagnostics("HLM", multitask_values, self._hlm_task)
            if "RLM" in requested:
                assert self._rlm_task is not None
                log_predictions["RLM"] = np.asarray(
                    multitask_values[self._rlm_task], dtype=np.float64
                )
                calibrated_key = f"calibrated_{self._rlm_task}"
                if calibrated_key in multitask_values:
                    calibrated_predictions["RLM"] = np.asarray(
                        multitask_values[calibrated_key], dtype=np.float64
                    )
                collect_diagnostics("RLM", multitask_values, self._rlm_task)
        if "MLM" in requested:
            assert self.mlm_single_predictor is not None
            assert self._mlm_task is not None
            mlm_values = self.mlm_single_predictor.predict_smiles(
                smiles_list,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            log_predictions["MLM"] = np.asarray(
                mlm_values[self._mlm_task], dtype=np.float64
            )
            calibrated_key = f"calibrated_{self._mlm_task}"
            if calibrated_key in mlm_values:
                calibrated_predictions["MLM"] = np.asarray(
                    mlm_values[calibrated_key], dtype=np.float64
                )
            collect_diagnostics("MLM", mlm_values, self._mlm_task)

        predictor_values = {
            "HLM": multitask_values,
            "RLM": multitask_values,
            "MLM": mlm_values,
        }
        task_names = {
            "HLM": self._hlm_task,
            "RLM": self._rlm_task,
            "MLM": self._mlm_task,
        }
        for species in requested:
            task_name = task_names[species]
            values = predictor_values[species]
            if task_name is None:
                continue
            lower = next(
                (
                    name
                    for name in values
                    if name.startswith(f"{task_name}_lower_")
                ),
                None,
            )
            upper = next(
                (
                    name
                    for name in values
                    if name.startswith(f"{task_name}_upper_")
                ),
                None,
            )
            if lower and upper:
                coverage = lower.rsplit("_", 1)[-1]
                calibrated_intervals[species] = {
                    f"lower_{coverage}": np.asarray(values[lower], dtype=np.float64),
                    f"upper_{coverage}": np.asarray(values[upper], dtype=np.float64),
                }

        output: dict[str, np.ndarray] = {}
        for species in requested:
            output[LOG_OUTPUT_COLUMNS[species]] = log_predictions[species]
            output[RAW_OUTPUT_COLUMNS[species]] = _inverse_log10(
                log_predictions[species]
            )
            task_name = task_names[species]
            values = predictor_values[species]
            if task_name is not None and f"mc_std_{task_name}" in values:
                output[f"Log_{species}_CLint_mc_std"] = np.asarray(
                    values[f"mc_std_{task_name}"], dtype=np.float64
                )
            if species in calibrated_predictions:
                calibrated = calibrated_predictions[species]
                output[f"Log_{species}_CLint_calibrated"] = calibrated
                output[f"{species}_CLint_calibrated (mL/min/kg)"] = _inverse_log10(
                    calibrated
                )
            for bound_name, bound_values in calibrated_intervals.get(
                species, {}
            ).items():
                output[f"Log_{species}_CLint_{bound_name}"] = bound_values
                output[f"{species}_CLint_{bound_name} (mL/min/kg)"] = _inverse_log10(
                    bound_values
                )
        output.update(diagnostics)
        return output

    def predict_frame(
        self,
        smiles: Sequence[str],
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        properties: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        smiles_list = [str(value).strip() for value in smiles]
        predictions = self.forward(
            smiles_list,
            batch_size=batch_size,
            num_workers=num_workers,
            properties=properties,
        )
        return pd.DataFrame({"SMILES": smiles_list, **predictions})
