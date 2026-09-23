"""Inference for ChemFlow ChemBERTa property models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from .trainer import (
    ChemBERTaPropertyModel,
    _load_saved_encoder,
    _select_device,
    _transformer_imports,
)
from chemflow.deep_learning.chemeleon.applicability import (
    applicability_diagnostics, apply_local_calibration,
    apply_mc_dropout_intervals,
)


class ChemBERTaPredictor:
    def __init__(
        self,
        model_directory: str | Path,
        *,
        device: str = "auto",
        threshold: float = 0.5,
        mc_dropout_samples: int = 0,
    ) -> None:
        self.directory = Path(model_directory).expanduser().resolve()
        metadata_path = self.directory / "chemflow_config.json"
        head_path = self.directory / "property_head.pt"
        if not metadata_path.is_file() or not head_path.is_file():
            raise FileNotFoundError(
                "ChemBERTa model directory must contain chemflow_config.json "
                "and property_head.pt."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.targets = [str(value) for value in self.metadata["targets"]]
        self.task_types = [str(value) for value in self.metadata["task_types"]]
        self.means = np.asarray(self.metadata["means"], dtype=np.float32)
        self.stds = np.asarray(self.metadata["stds"], dtype=np.float32)
        self.threshold = float(threshold)
        self.mc_dropout_samples = int(mc_dropout_samples)
        if self.mc_dropout_samples == 1 or self.mc_dropout_samples < 0:
            raise ValueError("mc_dropout_samples must be 0 or at least 2.")
        self.device = _select_device(device)
        AutoModel, AutoTokenizer = _transformer_imports()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.directory, local_files_only=True
        )
        encoder = _load_saved_encoder(
            self.directory,
            self.metadata.get("architecture", "roberta"),
            AutoModel,
        )
        self.model = ChemBERTaPropertyModel(
            encoder,
            len(self.targets),
            float(self.metadata.get("dropout", 0.1)),
        ).to(self.device)
        self.model.head.load_state_dict(
            torch.load(head_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        applicability_path = self.directory / "applicability.pt"
        self.applicability = (
            torch.load(applicability_path, map_location="cpu", weights_only=False)
            if applicability_path.is_file() else None
        )
        self.invalid_indices: list[int] = []

    def predict_smiles(
        self, smiles: list[str], *, batch_size: int = 32, max_length: int = 512
    ) -> dict[str, np.ndarray]:
        valid_indices, canonical = [], []
        self.invalid_indices = []
        for index, value in enumerate(smiles):
            molecule = Chem.MolFromSmiles(str(value).strip())
            if molecule is None:
                self.invalid_indices.append(index)
            else:
                valid_indices.append(index)
                canonical.append(Chem.MolToSmiles(molecule, canonical=True))
        means = np.full((len(smiles), len(self.targets)), np.nan, dtype=np.float32)
        direct_means = np.full_like(means, np.nan)
        standard_deviations = np.full_like(means, np.nan)
        valid_embedding_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(canonical), int(batch_size)):
                batch_smiles = canonical[start : start + int(batch_size)]
                tokens = self.tokenizer(
                    batch_smiles, padding=True, truncation=True,
                    max_length=int(max_length), return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                if self.mc_dropout_samples:
                    self.model.eval()
                    direct_logits, pooled = self.model(**tokens)
                    for module in self.model.modules():
                        if isinstance(module, torch.nn.Dropout):
                            module.train()
                    stochastic = [self.model(**tokens) for _ in range(self.mc_dropout_samples)]
                    passes = torch.stack([value[0] for value in stochastic])
                    logits = passes.mean(0)
                    logit_std = passes.std(0, unbiased=True)
                    self.model.eval()
                else:
                    logits, pooled = self.model(**tokens)
                    direct_logits = logits
                    logit_std = torch.zeros_like(logits)
                values = logits.float().cpu().numpy()
                direct_values = direct_logits.float().cpu().numpy()
                spreads = logit_std.float().cpu().numpy()
                for task_index, task_type in enumerate(self.task_types):
                    if task_type == "classification":
                        probabilities = 1.0 / (1.0 + np.exp(-values[:, task_index]))
                        # Delta-method conversion from logit spread.
                        spreads[:, task_index] *= probabilities * (1.0 - probabilities)
                        values[:, task_index] = probabilities
                        direct_values[:, task_index] = 1.0 / (
                            1.0 + np.exp(-direct_values[:, task_index])
                        )
                    else:
                        values[:, task_index] = (
                            values[:, task_index] * self.stds[task_index]
                            + self.means[task_index]
                        )
                        spreads[:, task_index] *= self.stds[task_index]
                        direct_values[:, task_index] = (
                            direct_values[:, task_index] * self.stds[task_index]
                            + self.means[task_index]
                        )
                selected = np.asarray(valid_indices[start : start + len(batch_smiles)])
                means[selected] = values
                direct_means[selected] = direct_values
                standard_deviations[selected] = spreads
                valid_embedding_batches.append(pooled.float().cpu().numpy())
        output: dict[str, np.ndarray] = {}
        for index, (name, task_type) in enumerate(zip(self.targets, self.task_types)):
            output[f"direct_{name}_prediction"] = direct_means[:, index]
            output[f"{name}_prediction"] = means[:, index]
            if self.mc_dropout_samples:
                output[f"mc_mean_{name}_prediction"] = means[:, index]
            output[f"{name}_mc_std"] = standard_deviations[:, index]
            output[f"mc_std_{name}_prediction"] = standard_deviations[:, index]
            if task_type == "classification":
                output[f"prob_{name}"] = means[:, index]
                classes = np.full(len(smiles), np.nan, dtype=np.float32)
                finite = np.isfinite(means[:, index])
                classes[finite] = (means[finite, index] >= self.threshold).astype(float)
                output[f"{name}_class"] = classes
        if self.applicability and valid_embedding_batches:
            embeddings = np.concatenate(valid_embedding_batches, axis=0)
            for name in self.targets:
                diagnostics = applicability_diagnostics(
                    smiles, embeddings, valid_indices, self.applicability,
                    reference_task=name,
                )
                output.update({f"{name}_{key}": value for key, value in diagnostics.items()})
        output_names = [f"{name}_prediction" for name in self.targets]
        bounded_names = [
            f"{name}_prediction"
            for name, task_type in zip(self.targets, self.task_types)
            if task_type == "classification"
        ]
        confidence = float(
            self.applicability.get("calibration", {}).get("confidence", 0.90)
            if self.applicability else 0.90
        )
        apply_local_calibration(
            output, self.targets, output_names, confidence, bounded_names
        )
        apply_mc_dropout_intervals(
            output, output_names, confidence, bounded_names
        )
        return output
