"""Prepare ChemFlow datasets and launch the vendored KERMT backend."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .pretrained import (
    KERMT_CHECKPOINT,
    KERMT_REPO_ID,
    KERMT_REVISION,
    ensure_pretrained_artifacts,
)


_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def _targets(value: Any) -> list[str]:
    if isinstance(value, str):
        result = [value]
    elif isinstance(value, (list, tuple)):
        result = [str(item) for item in value]
    else:
        raise TypeError("DatasetConfig.target_column must be a string or list.")
    result = [item.strip() for item in result if item.strip()]
    if not result:
        raise ValueError("At least one target column is required.")
    return result


def _path(value: str, base: Path) -> Path:
    candidate = Path(os.path.expandvars(value)).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _split_counts(size: int, val_fraction: float, test_fraction: float) -> tuple[int, int]:
    if size < 2:
        raise ValueError("KERMT training requires at least two molecules.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("DatasetConfig.val_fraction must be between 0 and 1.")
    if not 0.0 <= test_fraction < 1.0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")
    val_count = max(1, int(round(size * val_fraction)))
    test_count = max(1, int(round(size * test_fraction))) if test_fraction else 0
    if val_count + test_count >= size:
        raise ValueError("Split fractions leave no training molecules.")
    return val_count, test_count


def _random_indices(
    frame: pd.DataFrame,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    val_count, test_count = _split_counts(len(frame), val_fraction, test_fraction)
    indices = np.random.default_rng(seed).permutation(len(frame)).tolist()
    test = indices[:test_count]
    val = indices[test_count : test_count + val_count]
    train = indices[test_count + val_count :]
    return {"train": train, "val": val, "test": test}


def _grouped_indices(
    frame: pd.DataFrame,
    group_values: list[str],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    """Assign intact molecule/scaffold groups near requested split sizes."""
    val_count, test_count = _split_counts(len(frame), val_fraction, test_fraction)
    targets = {
        "train": len(frame) - val_count - test_count,
        "val": val_count,
        "test": test_count,
    }
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(group_values):
        grouped[value].append(index)
    rng = np.random.default_rng(seed)
    groups = list(grouped.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    assigned = {name: [] for name in targets}
    active = [name for name, target in targets.items() if target > 0]
    for group in groups:
        destination = max(
            active,
            key=lambda name: (targets[name] - len(assigned[name])) / targets[name],
        )
        assigned[destination].extend(group)
    if any(not assigned[name] for name in active):
        raise ValueError(
            "Grouped split produced an empty partition; use more molecules, "
            "a smaller fraction, or a predefined split."
        )
    return assigned


def _canonical_smiles(value: str) -> str:
    molecule = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else str(value)


def _scaffold(value: str, index: int) -> str:
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return f"invalid-{index}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
    return scaffold or f"acyclic-{index}"


def _cuik_descriptor_finite_mask(
    smiles: list[str],
    *,
    normalization_type: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Preflight the same normalized descriptors generated by MolCollator."""
    from descriptastorus.descriptors import rdDescriptors

    from .vendor.kermt.data.molgraph import MoleculeFeaturizer

    descriptor_names = [column[0] for column in rdDescriptors.RDKit2D().columns]
    featurizer = MoleculeFeaturizer(
        molecular_descriptor_type="rdkit2D",
        rdkit2D_descriptor_list=descriptor_names,
        rdkit2D_normalization_type=normalization_type,
    )
    finite = np.ones(len(smiles), dtype=bool)

    def batch_finite(values: list[str]) -> np.ndarray:
        features = featurizer.featurize(values)
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        array = np.asarray(features, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] != len(values):
            raise ValueError(
                "KERMT descriptor preflight returned an unexpected row count: "
                f"{array.shape[0]} for {len(values)} molecules."
            )
        return np.isfinite(array).all(axis=1)

    for start in range(0, len(smiles), batch_size):
        stop = min(start + batch_size, len(smiles))
        values = smiles[start:stop]
        try:
            finite[start:stop] = batch_finite(values)
        except Exception:
            # Some descriptor implementations fail the entire batch when one
            # structure is unsupported. Retry individually so only the actual
            # offending structures are rejected and reported.
            for offset, value in enumerate(values, start=start):
                try:
                    finite[offset] = bool(batch_finite([value])[0])
                except Exception:
                    finite[offset] = False
    return finite


class KERMTTrainer:
    """Prepare data and launch vendored KERMT fine-tuning."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with self.config_path.open("rb") as stream:
            raw = tomllib.load(stream)
        self.base_cfg = dict(raw.get("BaseConfig", {}))
        self.data_cfg = dict(raw.get("DatasetConfig", {}))
        self.model_cfg = dict(raw.get("KERMTConfig", {}))
        self.training_cfg = dict(raw.get("TrainingConfig", {}))
        self.config_dir = self.config_path.parent
        self._validate()

    def _validate(self) -> None:
        task = str(self.base_cfg.get("task", "regression")).lower()
        if task != "regression":
            raise ValueError("The current KERMT adapter supports regression only.")
        self.targets = _targets(self.data_cfg.get("target_column"))
        self.smiles_column = str(self.data_cfg.get("smiles_column", "SMILES"))
        self.split_type = str(
            self.data_cfg.get("split_type", "scaffold_balanced")
        ).strip().lower()
        allowed = {
            "random",
            "random_with_repeated_smiles",
            "scaffold_balanced",
            "kennard_stone",
            "kmeans",
            "predefined",
        }
        if self.data_cfg.get("split_column"):
            self.split_type = "predefined"
        if self.split_type not in allowed:
            raise ValueError(f"Unsupported KERMT split_type: {self.split_type!r}.")
        if self.split_type == "predefined" and not self.data_cfg.get("split_column"):
            raise ValueError("split_type='predefined' requires DatasetConfig.split_column.")

        base = self.config_dir
        self.dataset_path = _path(str(self.data_cfg["dataset_path"]), base)
        test_value = str(self.data_cfg.get("test_dataset_path", "")).strip()
        self.test_dataset_path = _path(test_value, base) if test_value else None
        checkpoint_value = str(self.model_cfg.get("checkpoint_path", "")).strip()
        self.checkpoint_path = _path(checkpoint_value, base) if checkpoint_value else None
        if self.checkpoint_path is not None and not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"KERMT pretrained checkpoint not found: {self.checkpoint_path}"
            )
        cache_value = str(self.model_cfg.get("cache_dir", "")).strip()
        self.cache_dir = _path(cache_value, base) if cache_value else None
        self.model_repo_id = str(
            self.model_cfg.get("model_repo_id", KERMT_REPO_ID)
        ).strip()
        # Keep the official artifact revision internal so ordinary training
        # configurations cannot accidentally mix checkpoint/vocabulary versions.
        self.model_revision = KERMT_REVISION
        self.local_files_only = bool(
            self.model_cfg.get("local_files_only", False)
        )
        self.resume = bool(self.training_cfg.get("resume", False))
        resume_value = str(self.training_cfg.get("resume_checkpoint", "")).strip()
        self.resume_checkpoint = _path(resume_value, base) if resume_value else None
        if self.test_dataset_path is not None and float(
            self.data_cfg.get("test_fraction", 0.0)
        ) != 0.0:
            raise ValueError("An external test dataset requires test_fraction = 0.0.")
        uses_cuik = bool(
            self.model_cfg.get("use_cuikmolmaker_featurization", False)
        )
        feature_generator = str(
            self.model_cfg.get("features_generator", "rdkit_2d_normalized")
        ).strip()
        if not uses_cuik and feature_generator.endswith("_cuik_molmaker"):
            raise ValueError(
                "A cuik_molmaker feature generator requires "
                "KERMTConfig.use_cuikmolmaker_featurization = true. For the "
                "portable path, use features_generator = "
                "'rdkit_2d_normalized'."
            )
        descriptor_policy = str(
            self.model_cfg.get("nonfinite_descriptor_policy", "drop")
        ).strip().lower()
        if descriptor_policy not in {"drop", "error"}:
            raise ValueError(
                "KERMTConfig.nonfinite_descriptor_policy must be 'drop' or 'error'."
            )
        self.nonfinite_descriptor_policy = descriptor_policy
        workdir_value = str(self.base_cfg.get("workdir", "./kermt_run"))
        self.workdir = _path(workdir_value, base)

    def _resolve_pretrained_checkpoint(self) -> None:
        """Use an explicit checkpoint or download the pinned official release."""
        if self.checkpoint_path is not None:
            print("KERMT pretrained source: explicit local checkpoint", flush=True)
            print(f"KERMT checkpoint: {self.checkpoint_path}", flush=True)
            print(f"KERMT checkpoint SHA256: {_sha256(self.checkpoint_path)}", flush=True)
            self.pretrained_audit = {
                "source": "explicit local checkpoint",
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_sha256": _sha256(self.checkpoint_path),
                "artifacts": [str(self.checkpoint_path)],
            }
            return

        print(
            "No KERMT checkpoint_path was configured; resolving the official "
            "pretrained release.",
            flush=True,
        )
        artifacts = ensure_pretrained_artifacts(
            cache_dir=self.cache_dir,
            repo_id=self.model_repo_id,
            revision=self.model_revision,
            local_files_only=self.local_files_only,
            log=lambda message: print(message, flush=True),
        )
        self.checkpoint_path = artifacts[KERMT_CHECKPOINT]
        self.pretrained_audit = {
            "source": "Hugging Face Hub/cache",
            "repo_id": self.model_repo_id,
            "revision": self.model_revision,
            "cache_dir": str(self.checkpoint_path.parent),
            "local_files_only": self.local_files_only,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": _sha256(self.checkpoint_path),
            "artifacts": [str(path) for path in artifacts.values()],
        }

    def _select(self, frame: pd.DataFrame, source: Path) -> pd.DataFrame:
        required = [self.smiles_column, *self.targets]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{source} is missing required columns: {missing}")
        valid_smiles = (
            frame[self.smiles_column].notna()
            & frame[self.smiles_column].astype(str).str.strip().ne("")
        )
        has_selected_label = frame[self.targets].notna().any(axis=1)
        selected = frame.loc[valid_smiles & has_selected_label, required].copy()
        missing_all_targets = int((valid_smiles & ~has_selected_label).sum())
        if missing_all_targets:
            print(
                f"Excluded {missing_all_targets:,} rows from {source} because all "
                f"selected targets are missing: {', '.join(self.targets)}",
                flush=True,
            )
        selected[self.smiles_column] = selected[self.smiles_column].astype(str).str.strip()
        selected = selected.reset_index(drop=True)
        selected = selected.rename(columns={self.smiles_column: "smiles"})
        return selected

    def prepare_data(self) -> dict[str, Path]:
        frame = _read_table(self.dataset_path)
        selected = self._select(frame, self.dataset_path)
        external_test = None
        if self.test_dataset_path is not None:
            external_test = self._select(
                _read_table(self.test_dataset_path), self.test_dataset_path
            )

        if self.split_type == "predefined":
            column = str(self.data_cfg["split_column"])
            if column not in frame.columns:
                raise ValueError(f"Split column {column!r} is missing from {self.dataset_path}.")
            usable = frame.loc[
                frame[self.smiles_column].notna()
                & frame[self.smiles_column].astype(str).str.strip().ne("")
                & frame[self.targets].notna().any(axis=1)
            ].copy()
            labels = usable[column].map(
                lambda value: _SPLIT_ALIASES.get(str(value).strip().lower())
            ).reset_index(drop=True)
            if labels.isna().any():
                raise ValueError("Predefined split values must be train, val/valid, or test.")
            selected = self._select(usable, self.dataset_path)
            split_frames = {
                name: selected.loc[labels == name].reset_index(drop=True)
                for name in ("train", "val", "test")
            }
            if external_test is not None:
                if not split_frames["test"].empty:
                    raise ValueError("Do not combine predefined test rows with test_dataset_path.")
                split_frames["test"] = external_test
        else:
            val_fraction = float(self.data_cfg.get("val_fraction", 0.1))
            test_fraction = 0.0 if external_test is not None else float(
                self.data_cfg.get("test_fraction", 0.1)
            )
            seed = int(self.training_cfg.get("seed", 42))
            if self.split_type == "random":
                indices = _random_indices(
                    selected,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                    seed=seed,
                )
            elif self.split_type in {"kennard_stone", "kmeans"}:
                from chemprop import data as chemprop_data

                molecules = [Chem.MolFromSmiles(value) for value in selected["smiles"]]
                invalid = [index for index, molecule in enumerate(molecules) if molecule is None]
                if invalid:
                    raise ValueError(
                        f"{self.split_type} splitting requires valid SMILES; "
                        f"invalid prepared-row indices: {invalid[:10]}"
                    )
                split_indices = chemprop_data.make_split_indices(
                    molecules,
                    split=self.split_type,
                    sizes=(
                        1.0 - val_fraction - test_fraction,
                        val_fraction,
                        test_fraction,
                    ),
                    seed=seed,
                )
                indices = {
                    name: [int(index) for index in values[0]]
                    for name, values in zip(
                        ("train", "val", "test"), split_indices
                    )
                }
            else:
                smiles = selected["smiles"].tolist()
                groups = (
                    [_canonical_smiles(value) for value in smiles]
                    if self.split_type == "random_with_repeated_smiles"
                    else [_scaffold(value, index) for index, value in enumerate(smiles)]
                )
                indices = _grouped_indices(
                    selected,
                    groups,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                    seed=seed,
                )
            split_frames = {
                name: selected.iloc[values].reset_index(drop=True)
                for name, values in indices.items()
            }
            if external_test is not None:
                split_frames["test"] = external_test

        feature_generator = str(
            self.model_cfg.get("features_generator", "rdkit_2d_normalized")
        ).strip()
        if "rdkit_2d_normalized_cuik_molmaker" in feature_generator:
            normalization_type = str(
                self.model_cfg.get("rdkit2d_normalization_type", "fast")
            ).strip()
            rejected_frames: list[pd.DataFrame] = []
            for split_name, split_frame in split_frames.items():
                finite = _cuik_descriptor_finite_mask(
                    split_frame["smiles"].astype(str).tolist(),
                    normalization_type=normalization_type,
                )
                if finite.all():
                    continue
                rejected = split_frame.loc[~finite].copy()
                rejected.insert(0, "split", split_name)
                rejected["reason"] = "nonfinite_rdkit2d_normalized_descriptor"
                rejected_frames.append(rejected)
                failing_smiles = rejected["smiles"].astype(str).tolist()
                if self.nonfinite_descriptor_policy == "error":
                    raise ValueError(
                        "KERMT descriptor preflight rejected "
                        f"{len(failing_smiles)} molecule(s) from {split_name}: "
                        f"{failing_smiles}"
                    )
                split_frames[split_name] = split_frame.loc[finite].reset_index(
                    drop=True
                )
                print(
                    "Excluded "
                    f"{len(failing_smiles):,} molecule(s) from the {split_name} "
                    "split because KERMT generated non-finite normalized "
                    "descriptors. See kermt_rejected_rows.csv.",
                    flush=True,
                )
            if rejected_frames:
                self.workdir.mkdir(parents=True, exist_ok=True)
                pd.concat(rejected_frames, ignore_index=True).to_csv(
                    self.workdir / "kermt_rejected_rows.csv", index=False
                )

        if split_frames["train"].empty or split_frames["val"].empty:
            raise ValueError("KERMT requires non-empty training and validation sets.")
        if split_frames["test"].empty:
            raise ValueError("KERMT requires an internal or independent test set.")
        for split_name in ("train", "val"):
            missing_targets = [
                target
                for target in self.targets
                if not split_frames[split_name][target].notna().any()
            ]
            if missing_targets:
                raise ValueError(
                    f"KERMT {split_name} split has no labels after descriptor "
                    f"screening for targets: {missing_targets}"
                )

        data_dir = self.workdir / "prepared_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        manifest_frames = []
        for name, split_frame in split_frames.items():
            destination = data_dir / f"{name}.csv"
            split_frame.to_csv(destination, index=False)
            paths[name] = destination
            manifest_frames.append(split_frame.assign(split=name))
        pd.concat(manifest_frames, ignore_index=True).rename(
            columns={"smiles": "SMILES"}
        ).to_csv(self.workdir / "data_splits.csv", index=False)
        print(
            "Prepared KERMT splits: "
            + ", ".join(
                f"{name}={len(split_frames[name]):,}"
                for name in ("train", "val", "test")
            ),
            flush=True,
        )
        return paths

    def build_command(self, paths: dict[str, Path]) -> list[str]:
        if self.checkpoint_path is None:
            raise RuntimeError("KERMT checkpoint has not been resolved.")
        python_value = str(self.model_cfg.get("python_executable", "")).strip()
        python_executable = python_value or sys.executable
        output_dir = self.workdir / "kermt_output"
        freeze_encoder = bool(self.model_cfg.get("freeze_encoder", False))
        # KERMT's upstream CLI expresses encoder freezing as an encoder learning-
        # rate coefficient of zero. Keep that implementation detail behind the
        # clearer ChemFlow freeze_encoder option. The old fine_tune_coff key is
        # accepted as a compatibility fallback for existing configurations.
        encoder_lr_multiplier = float(
            self.training_cfg.get(
                "encoder_lr_multiplier",
                self.training_cfg.get("fine_tune_coff", 1.0),
            )
        )
        if encoder_lr_multiplier < 0.0:
            raise ValueError("TrainingConfig.encoder_lr_multiplier cannot be negative.")
        early_stopping_patience = int(
            self.training_cfg.get("early_stopping_patience", 0)
        )
        early_stopping_min_delta = float(
            self.training_cfg.get("early_stopping_min_delta", 0.0)
        )
        early_stopping_monitor = str(
            self.training_cfg.get("early_stopping_monitor", "metric")
        ).strip().lower()
        if early_stopping_patience < 0:
            raise ValueError(
                "TrainingConfig.early_stopping_patience cannot be negative."
            )
        if early_stopping_min_delta < 0.0:
            raise ValueError(
                "TrainingConfig.early_stopping_min_delta cannot be negative."
            )
        if early_stopping_monitor not in {"metric", "loss"}:
            raise ValueError(
                "TrainingConfig.early_stopping_monitor must be 'metric' or 'loss'."
            )
        checkpoint_interval = int(
            self.training_cfg.get("checkpoint_every_n_epochs", 0)
        )
        if checkpoint_interval < 0:
            raise ValueError(
                "TrainingConfig.checkpoint_every_n_epochs cannot be negative."
            )
        regression_loss = str(
            self.training_cfg.get("regression_loss", "l2")
        ).strip().lower()
        allowed_regression_losses = {
            "l2", "mse", "mae", "huber", "nll", "gaussian_nll"
        }
        if regression_loss not in allowed_regression_losses:
            raise ValueError(
                "TrainingConfig.regression_loss must be one of "
                f"{sorted(allowed_regression_losses)}."
            )
        huber_delta = float(self.training_cfg.get("huber_delta", 1.0))
        nll_scale = float(self.training_cfg.get("nll_scale", 1.0))
        gaussian_nll_variance = float(
            self.training_cfg.get("gaussian_nll_variance", 1.0)
        )
        if huber_delta <= 0.0:
            raise ValueError("TrainingConfig.huber_delta must be positive.")
        if nll_scale <= 0.0:
            raise ValueError("TrainingConfig.nll_scale must be positive.")
        if gaussian_nll_variance <= 0.0:
            raise ValueError(
                "TrainingConfig.gaussian_nll_variance must be positive."
            )
        fine_tune_coefficient = 0.0 if freeze_encoder else encoder_lr_multiplier
        command = [
            python_executable,
            "-m",
            "chemflow.deep_learning.kermt.vendor.main",
            "finetune",
            "--data_path", str(paths["train"]),
            "--separate_val_path", str(paths["val"]),
            "--separate_test_path", str(paths["test"]),
            "--save_dir", str(output_dir),
            "--checkpoint_path", str(self.checkpoint_path),
            "--dataset_type", "regression",
            "--split_type", "random",
            "--ensemble_size", str(int(self.training_cfg.get("ensemble_size", 1))),
            "--num_folds", str(int(self.training_cfg.get("num_folds", 1))),
            "--ffn_hidden_size", str(int(self.model_cfg.get("ffn_hidden_size", 700))),
            "--ffn_num_layers", str(int(self.model_cfg.get("ffn_num_layers", 3))),
            "--bond_drop_rate", str(float(self.model_cfg.get("bond_drop_rate", 0.1))),
            "--epochs", str(int(self.training_cfg.get("num_epochs", 100))),
            "--checkpoint_every_n_epochs", str(checkpoint_interval),
            "--metric", str(self.training_cfg.get("metric", "mae")),
            "--regression_loss", regression_loss,
            "--huber_delta", str(huber_delta),
            "--nll_scale", str(nll_scale),
            "--gaussian_nll_variance", str(gaussian_nll_variance),
            "--dist_coff", str(float(self.model_cfg.get("dist_coff", 0.15))),
            "--init_lr", str(float(self.training_cfg.get("init_lr", 1e-5))),
            "--max_lr", str(float(self.training_cfg.get("max_lr", 1e-4))),
            "--final_lr", str(float(self.training_cfg.get("final_lr", 2e-5))),
            "--warmup_epochs", str(int(self.training_cfg.get("warmup_epochs", 2))),
            "--weight_decay", str(float(self.training_cfg.get("weight_decay", 0.0))),
            "--early_stopping_patience", str(early_stopping_patience),
            "--early_stopping_min_delta", str(early_stopping_min_delta),
            "--early_stopping_monitor", early_stopping_monitor,
            "--fine_tune_coff", str(fine_tune_coefficient),
            "--dropout", str(float(self.model_cfg.get("dropout", 0.0))),
            "--batch_size", str(int(self.training_cfg.get("batch_size", 32))),
            "--seed", str(int(self.training_cfg.get("seed", 42))),
            "--test_mc_dropout_samples", str(
                int(self.training_cfg.get("test_mc_dropout_samples", 0))
            ),
            "--test_calibration_confidence", str(
                float(self.training_cfg.get("test_calibration_confidence", 0.90))
            ),
        ]
        if (
            int(self.training_cfg.get("test_mc_dropout_samples", 0)) >= 2
            and float(self.model_cfg.get("dropout", 0.0)) <= 0.0
        ):
            raise ValueError(
                "KERMT MC-dropout test evaluation requires KERMTConfig.dropout > 0."
            )
        if self.resume:
            if self.resume_checkpoint is not None:
                if not self.resume_checkpoint.exists():
                    raise FileNotFoundError(
                        f"KERMT resume checkpoint does not exist: {self.resume_checkpoint}"
                    )
                command.extend(["--resume", "--resume_checkpoint", str(self.resume_checkpoint)])
            else:
                ensemble_size = int(self.training_cfg.get("ensemble_size", 1))
                num_folds = int(self.training_cfg.get("num_folds", 1))
                missing = [
                    output_dir / f"fold_{fold}" / f"model_{model}" / "last_checkpoint.pt"
                    for fold in range(num_folds)
                    for model in range(ensemble_size)
                    if not (
                        output_dir
                        / f"fold_{fold}"
                        / f"model_{model}"
                        / "last_checkpoint.pt"
                    ).is_file()
                ]
                if missing:
                    raise FileNotFoundError(
                        "KERMT resume was requested, but checkpoint(s) are missing: "
                        + ", ".join(str(path) for path in missing)
                    )
                command.append("--resume")
        flag_options = (
            ("no_features_scaling", "--no_features_scaling", True),
            ("self_attention", "--self_attention", True),
            ("use_cuikmolmaker_featurization", "--use_cuikmolmaker_featurization", False),
        )
        for key, flag, default in flag_options:
            if bool(self.model_cfg.get(key, default)):
                command.append(flag)
        features = str(
            self.model_cfg.get("features_generator", "rdkit_2d_normalized")
        ).strip()
        if features:
            command.extend(["--features_generator", features])
        normalization = str(
            self.model_cfg.get("rdkit2d_normalization_type", "")
        ).strip()
        if normalization:
            command.extend(["--rdkit2D_normalization_type", normalization])
        extra_args = self.training_cfg.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(
            isinstance(value, str) for value in extra_args
        ):
            raise TypeError("TrainingConfig.extra_args must be a list of strings.")
        command.extend(extra_args)
        return command

    def train(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._resolve_pretrained_checkpoint()
        paths = self.prepare_data()
        command = self.build_command(paths)
        audit = {
            "backend": "vendored NVIDIA-BioNeMo/KERMT",
            "upstream_commit": "8828743036675d1b6d5f4586ef7bcfea70233d59",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": _sha256(self.checkpoint_path),
            "pretrained_weight_audit": self.pretrained_audit,
            "targets": self.targets,
            "command": command,
        }
        (self.workdir / "kermt_run_manifest.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        print("KERMT backend: vendored NVIDIA-BioNeMo/KERMT", flush=True)
        print(f"Pretrained checkpoint: {self.checkpoint_path}", flush=True)
        print(f"Checkpoint SHA256: {audit['checkpoint_sha256']}", flush=True)
        print(f"Targets: {', '.join(self.targets)}", flush=True)
        print(f"Resume training: {'yes' if self.resume else 'no'}", flush=True)
        if self.resume:
            print(
                "Resume source: "
                + (
                    str(self.resume_checkpoint)
                    if self.resume_checkpoint is not None
                    else str(self.workdir / "kermt_output")
                ),
                flush=True,
            )
        print(f"Command: {shlex.join(command)}", flush=True)
        if bool(self.training_cfg.get("dry_run", False)):
            print("Dry run requested; KERMT was not launched.", flush=True)
            return
        subprocess.run(command, cwd=self.workdir, check=True)
        history_paths = sorted(
            (self.workdir / "kermt_output").glob(
                "fold_*/model_*/training_history.csv"
            )
        )
        if history_paths:
            histories = []
            for history_path in history_paths:
                frame = pd.read_csv(history_path)
                frame.insert(0, "model", history_path.parent.name)
                frame.insert(0, "fold", history_path.parent.parent.name)
                histories.append(frame)
            combined_history = self.workdir / "training_history.csv"
            pd.concat(histories, ignore_index=True).to_csv(
                combined_history,
                index=False,
            )
            from chemflow.deep_learning.plotter.training_plotter import (
                plot_training_history,
            )

            plot_training_history(
                combined_history,
                self.workdir / "plots" / "training_history.png",
                title="KERMT fine-tuning",
            )
            print(f"KERMT epoch history: {combined_history}", flush=True)
