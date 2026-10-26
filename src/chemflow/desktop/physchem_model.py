"""Deployment wrapper for the bundled LogD and solubility models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from chemflow.deep_learning.chemprop.predictor import ChempropCheckpointPredictor
from chemflow.desktop.physchem_domain import PhysChemDomainAnalyzer


PROPERTY_TARGETS = {
    "LOGD": "LogD_pooled",
    "SOLUBILITY": "LogS_K_pooled",
    "THERMODYNAMIC_SOLUBILITY": "LogS_T_ChEMBL",
}
PROPERTY_PREFIXES = {
    "LOGD": "LogD",
    "SOLUBILITY": "Kinetic_LogS",
    "THERMODYNAMIC_SOLUBILITY": "Thermodynamic_LogS",
}


def bundled_physchem_checkpoints() -> dict[str, Path]:
    model_directory = Path(__file__).with_name("models")
    return {
        "LOGD": model_directory / "logd_best.pt",
        "SOLUBILITY": model_directory / "solubility_best.pt",
        "THERMODYNAMIC_SOLUBILITY": model_directory / "solubility_best.pt",
    }


class PhysChemModel:
    """Run the deployed physicochemical models through one prediction API."""

    def __init__(
        self,
        checkpoint_paths: dict[str, str | Path],
        *,
        device: str | None = None,
        applicability_domain: bool = True,
        ood_confidence: float = 0.95,
    ) -> None:
        if not checkpoint_paths:
            raise ValueError("At least one physicochemical checkpoint is required.")
        self.predictors: dict[str, ChempropCheckpointPredictor] = {}
        predictors_by_path: dict[Path, ChempropCheckpointPredictor] = {}
        for property_name, path in checkpoint_paths.items():
            normalized = str(property_name).upper()
            if normalized not in PROPERTY_TARGETS:
                raise ValueError(f"Unsupported physicochemical property: {property_name}")
            resolved_path = Path(path).expanduser().resolve()
            predictor = predictors_by_path.get(resolved_path)
            if predictor is None:
                predictor = ChempropCheckpointPredictor(resolved_path, device=device)
                predictors_by_path[resolved_path] = predictor
            target = PROPERTY_TARGETS[normalized]
            if target not in predictor.output_columns:
                raise ValueError(
                    f"{Path(path).name} does not contain required output {target!r}; "
                    f"available outputs: {list(predictor.output_columns)}"
                )
            self.predictors[normalized] = predictor
        self.domain_analyzer = (
            PhysChemDomainAnalyzer(
                self.predictors, confidence=ood_confidence
            )
            if applicability_domain
            else None
        )

    def predict_frame(
        self,
        smiles: Sequence[str],
        *,
        properties: Sequence[str] | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        mc_dropout_samples: int = 0,
    ) -> pd.DataFrame:
        requested = (
            list(self.predictors)
            if properties is None
            else [str(value).upper() for value in properties]
        )
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("Select one or more unique physicochemical properties.")
        unavailable = sorted(set(requested).difference(self.predictors))
        if unavailable:
            raise ValueError(f"No loaded checkpoint provides: {unavailable}")

        frame = pd.DataFrame({"SMILES": [str(value) for value in smiles]})
        prediction_cache: dict[int, pd.DataFrame] = {}
        for property_name in requested:
            predictor = self.predictors[property_name]
            cache_key = id(predictor)
            values = prediction_cache.get(cache_key)
            if values is None:
                values = predictor.predict_smiles(
                    smiles,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    mc_dropout_samples=mc_dropout_samples,
                )
                prediction_cache[cache_key] = values
            target = PROPERTY_TARGETS[property_name]
            if property_name == "LOGD":
                frame["LogD_prediction"] = values[target]
                self._add_mc_columns(
                    frame, values, target, "LogD", convert_to_molar=False
                )
            elif property_name == "SOLUBILITY":
                log_s = pd.to_numeric(values[target], errors="coerce").to_numpy()
                frame["Kinetic_LogS_prediction (log10 mol/L)"] = log_s
                molar = np.full(log_s.shape, np.nan, dtype=float)
                finite = np.isfinite(log_s)
                molar[finite] = np.power(10.0, log_s[finite])
                frame["Kinetic_solubility_prediction (µM)"] = molar * 1e6
                self._add_mc_columns(
                    frame, values, target, "Kinetic_LogS", convert_to_molar=True
                )
            else:
                log_s = pd.to_numeric(values[target], errors="coerce").to_numpy()
                frame["Thermodynamic_LogS_prediction (log10 mol/L)"] = log_s
                molar = np.full(log_s.shape, np.nan, dtype=float)
                finite = np.isfinite(log_s)
                molar[finite] = np.power(10.0, log_s[finite])
                frame["Thermodynamic_solubility_prediction (µM)"] = molar * 1e6
                self._add_mc_columns(
                    frame,
                    values,
                    target,
                    "Thermodynamic_LogS",
                    convert_to_molar=True,
                )
            if self.domain_analyzer is not None:
                diagnostics = self.domain_analyzer.analyze(smiles, property_name)
                prefix = PROPERTY_PREFIXES[property_name]
                for column in diagnostics:
                    frame[f"{prefix}_{column}"] = diagnostics[column].to_numpy()
        ood_columns = [column for column in frame if column.endswith("_ood_flag")]
        domain_columns = [column for column in frame if column.endswith("_domain")]
        if ood_columns:
            frame["OOD_flag"] = frame[ood_columns].astype(bool).any(axis=1)
            assessed = frame[domain_columns].ne("NOT ASSESSED").all(axis=1)
            frame["Domain"] = np.where(
                ~assessed,
                "NOT ASSESSED",
                np.where(frame["OOD_flag"], "OOD", "IN DOMAIN"),
            )
        return frame

    @staticmethod
    def _add_mc_columns(
        frame: pd.DataFrame,
        values: pd.DataFrame,
        target: str,
        prefix: str,
        *,
        convert_to_molar: bool,
    ) -> None:
        mean_column = f"{target}__mc_mean"
        if mean_column not in values:
            return
        mean = pd.to_numeric(values[mean_column], errors="coerce").to_numpy()
        standard_deviation = pd.to_numeric(
            values[f"{target}__mc_std"], errors="coerce"
        ).to_numpy()
        lower = pd.to_numeric(
            values[f"{target}__mc_lower_95"], errors="coerce"
        ).to_numpy()
        upper = pd.to_numeric(
            values[f"{target}__mc_upper_95"], errors="coerce"
        ).to_numpy()
        frame[f"{prefix}_MC_prediction"] = mean
        frame[f"{prefix}_MC_uncertainty_std"] = standard_deviation
        frame[f"{prefix}_MC_lower_95"] = lower
        frame[f"{prefix}_MC_upper_95"] = upper
        if convert_to_molar:
            frame[f"{prefix}_MC_prediction (µM)"] = np.power(10.0, mean) * 1e6
            frame[f"{prefix}_MC_lower_95 (µM)"] = np.power(10.0, lower) * 1e6
            frame[f"{prefix}_MC_upper_95 (µM)"] = np.power(10.0, upper) * 1e6
