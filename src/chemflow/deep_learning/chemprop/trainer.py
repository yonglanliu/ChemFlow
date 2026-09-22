"""Validate ChemFlow configuration and launch the official Chemprop v2 CLI."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
_GENERATED_SPLITS = {
    "random",
    "random_with_repeated_smiles",
    "scaffold_balanced",
    "kennard_stone",
    "kmeans",
}
_REGRESSION_METRICS = {"mse", "mae", "rmse", "r2"}


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Calculate independent-test metrics on finite, paired observations."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[finite]
    prediction = prediction[finite]
    if truth.size < 2:
        return {
            "n": int(truth.size),
            "rmse": None,
            "mae": None,
            "r2": None,
            "pearson": None,
            "spearman": None,
            "kendall": None,
        }
    return {
        "n": int(truth.size),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)),
        "pearson": float(pearsonr(truth, prediction).statistic),
        "spearman": float(spearmanr(truth, prediction).statistic),
        "kendall": float(kendalltau(truth, prediction).statistic),
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
    if len(result) != len(set(result)):
        raise ValueError("DatasetConfig.target_column contains duplicate names.")
    return result


def _items(value: Any, *, name: str) -> list[str]:
    if isinstance(value, str):
        result = [value]
    elif isinstance(value, (list, tuple)):
        result = [str(item) for item in value]
    else:
        raise TypeError(f"{name} must be a string or list.")
    return [item.strip() for item in result if item.strip()]


def _path(value: str, base: Path) -> Path:
    candidate = Path(os.path.expandvars(value)).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("The official Chemprop CLI requires CSV input files.")
    return pd.read_csv(path)


def _version_tuple(value: str) -> tuple[int, int]:
    try:
        parts = value.split(".")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return (0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChempropTrainer:
    """Configuration adapter for Chemprop v2 single- and multitask training."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with self.config_path.open("rb") as stream:
            raw = tomllib.load(stream)
        self.base_cfg = dict(raw.get("BaseConfig", {}))
        self.data_cfg = dict(raw.get("DatasetConfig", {}))
        self.model_cfg = dict(raw.get("ChempropConfig", {}))
        self.training_cfg = dict(raw.get("TrainingConfig", {}))
        self.config_dir = self.config_path.parent
        self._validate_config()

    def _validate_config(self) -> None:
        self.task = str(self.base_cfg.get("task", "regression")).strip().lower()
        if self.task != "regression":
            raise ValueError("The current Chemprop adapter supports regression only.")

        self.workdir = _path(
            str(self.base_cfg.get("workdir", "./chemprop_run")), self.config_dir
        )
        self.dataset_path = _path(str(self.data_cfg["dataset_path"]), self.config_dir)
        test_value = str(self.data_cfg.get("test_dataset_path", "")).strip()
        self.test_dataset_path = (
            _path(test_value, self.config_dir) if test_value else None
        )
        self.smiles_column = str(
            self.data_cfg.get("smiles_column", "SMILES")
        ).strip()
        self.targets = _targets(self.data_cfg.get("target_column"))
        self.split_type = str(
            self.data_cfg.get("split_type", "scaffold_balanced")
        ).strip().lower()
        self.split_column = str(self.data_cfg.get("split_column", "")).strip()
        if self.split_column:
            self.split_type = "predefined"
        if self.split_type not in {*_GENERATED_SPLITS, "predefined"}:
            raise ValueError(f"Unsupported Chemprop split_type: {self.split_type!r}.")
        if self.split_type == "predefined" and not self.split_column:
            raise ValueError(
                "split_type='predefined' requires DatasetConfig.split_column."
            )

        self.val_fraction = float(self.data_cfg.get("val_fraction", 0.1))
        self.test_fraction = float(self.data_cfg.get("test_fraction", 0.1))
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("DatasetConfig.val_fraction must be between 0 and 1.")
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("DatasetConfig.test_fraction must be in [0, 1).")
        if self.test_dataset_path is not None and self.test_fraction != 0.0:
            raise ValueError("An external test dataset requires test_fraction = 0.0.")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be less than 1.")

        self.metrics = _items(
            self.training_cfg.get("metrics", ["rmse", "mae", "r2"]),
            name="TrainingConfig.metrics",
        )
        unsupported = sorted(set(self.metrics) - _REGRESSION_METRICS)
        if unsupported:
            raise ValueError(f"Unsupported Chemprop regression metrics: {unsupported}")
        if not self.metrics:
            raise ValueError("TrainingConfig.metrics must contain at least one metric.")
        extra_args = self.training_cfg.get("extra_args", [])
        if not isinstance(extra_args, (list, tuple)):
            raise TypeError("TrainingConfig.extra_args must be a list.")
        self.extra_args = [str(item) for item in extra_args]
        self.dry_run = bool(self.training_cfg.get("dry_run", False))
        self.resume = bool(self.training_cfg.get("resume", False))
        resume_value = self.training_cfg.get("resume_checkpoint", "")
        if isinstance(resume_value, str):
            resume_values = [resume_value] if resume_value.strip() else []
        elif isinstance(resume_value, (list, tuple)):
            resume_values = [str(item) for item in resume_value if str(item).strip()]
        else:
            raise TypeError(
                "TrainingConfig.resume_checkpoint must be a path or list of paths."
            )
        if resume_values and not self.resume:
            raise ValueError(
                "Set TrainingConfig.resume = true when resume_checkpoint is supplied."
            )
        self.resume_checkpoints = [
            _path(value, self.config_dir) for value in resume_values
        ]
        epochs = int(self.training_cfg.get("num_epochs", 100))
        warmup_epochs = int(self.training_cfg.get("warmup_epochs", 2))
        if epochs <= warmup_epochs:
            raise ValueError(
                "TrainingConfig.num_epochs must be greater than warmup_epochs."
            )
        self.early_stopping_patience = int(
            self.training_cfg.get("early_stopping_patience", 20)
        )
        if self.early_stopping_patience < 1:
            raise ValueError(
                "TrainingConfig.early_stopping_patience must be at least 1."
            )
        self.tracking_metric = str(
            self.training_cfg.get("tracking_metric", "val_loss")
        ).strip()
        if not self.tracking_metric:
            raise ValueError("TrainingConfig.tracking_metric cannot be empty.")

    def _resolve_resume_checkpoints(self) -> list[Path]:
        """Resolve checkpoints for Chemprop's weights-only continuation mode."""
        if not self.resume:
            return []
        checkpoints = list(self.resume_checkpoints)
        if not checkpoints:
            ensemble_size = int(self.training_cfg.get("ensemble_size", 1))
            checkpoints = [
                self.workdir / f"model_{index}" / "checkpoints" / "last.ckpt"
                for index in range(ensemble_size)
            ]
        missing = [path for path in checkpoints if not path.is_file()]
        if missing:
            formatted = "\n  ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "Chemprop resume checkpoint(s) not found:\n  " + formatted
            )
        return checkpoints

    def _validate_frame(self, frame: pd.DataFrame, path: Path, *, split: bool) -> None:
        required = {self.smiles_column, *self.targets}
        if split:
            required.add(self.split_column)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"Dataset {path} is missing columns {missing}. "
                f"Available columns: {list(frame.columns)}"
            )
        invalid_smiles = frame[self.smiles_column].isna() | frame[
            self.smiles_column
        ].astype(str).str.strip().eq("")
        if invalid_smiles.any():
            raise ValueError(
                f"Dataset {path} contains {int(invalid_smiles.sum())} empty SMILES."
            )
        if frame[self.targets].notna().sum().sum() == 0:
            raise ValueError(f"Dataset {path} contains no finite target labels.")

    def _prepare_predefined_files(
        self, frame: pd.DataFrame, test_frame: pd.DataFrame | None
    ) -> list[Path]:
        labels = (
            frame[self.split_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(_SPLIT_ALIASES)
        )
        if labels.isna().any():
            values = sorted(frame.loc[labels.isna(), self.split_column].astype(str).unique())
            raise ValueError(f"Unsupported predefined split labels: {values}")
        counts = labels.value_counts().to_dict()
        if not counts.get("train") or not counts.get("val"):
            raise ValueError("A predefined split requires nonempty train and val rows.")
        if test_frame is not None and counts.get("test", 0):
            raise ValueError(
                "Do not combine predefined test rows with test_dataset_path; "
                "use only the independent test dataset."
            )

        prepared = self.workdir / "prepared_data"
        prepared.mkdir(parents=True, exist_ok=True)
        columns = [self.smiles_column, *self.targets]
        partitions: list[pd.DataFrame] = []
        for name in ("train", "val"):
            path = prepared / f"{name}.csv"
            partition = frame.loc[labels.eq(name), columns].copy()
            partition.to_csv(path, index=False)
            partition["split"] = name
            partitions.append(partition)
        if test_frame is not None:
            path = prepared / "test.csv"
            partition = test_frame[columns].copy()
            partition.to_csv(path, index=False)
            partition["split"] = "test"
            partitions.append(partition)
        elif counts.get("test", 0):
            path = prepared / "test.csv"
            partition = frame.loc[labels.eq("test"), columns].copy()
            partition.to_csv(path, index=False)
            partition["split"] = "test"
            partitions.append(partition)
        else:
            raise ValueError(
                "Chemprop requires a test partition. Supply test_dataset_path or "
                "include test rows in the predefined split."
            )
        combined_path = prepared / "data_with_splits.csv"
        pd.concat(partitions, ignore_index=True).to_csv(combined_path, index=False)
        return [combined_path]

    def _prepare_generated_split(
        self, frame: pd.DataFrame, test_frame: pd.DataFrame | None
    ) -> list[Path]:
        """Use Chemprop's splitter, then encode every partition in one CSV."""
        from chemprop import data as chemprop_data

        molecules = []
        for index, value in enumerate(frame[self.smiles_column]):
            molecule = Chem.MolFromSmiles(str(value))
            if molecule is None:
                raise ValueError(
                    f"Invalid SMILES in {self.dataset_path} at row {index}: {value}"
                )
            molecules.append(molecule)
        internal_test_fraction = 0.0 if test_frame is not None else self.test_fraction
        sizes = (
            1.0 - self.val_fraction - internal_test_fraction,
            self.val_fraction,
            internal_test_fraction,
        )
        split_indices = chemprop_data.make_split_indices(
            molecules,
            split=self.split_type,
            sizes=sizes,
            seed=int(self.training_cfg.get("seed", 42)),
        )
        train_indices = list(split_indices[0][0])
        val_indices = list(split_indices[1][0])
        test_indices = list(split_indices[2][0]) if len(split_indices) > 2 else []
        if not train_indices or not val_indices:
            raise ValueError("Generated split produced an empty train or val partition.")

        columns = [self.smiles_column, *self.targets]
        partitions = []
        for name, indices in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        ):
            if not indices:
                continue
            partition = frame.iloc[indices][columns].copy()
            partition["split"] = name
            partitions.append(partition)
        if test_frame is not None:
            external = test_frame[columns].copy()
            external["split"] = "test"
            partitions.append(external)
        if not any(partition["split"].eq("test").any() for partition in partitions):
            raise ValueError(
                "Chemprop requires a test partition. Increase test_fraction or "
                "supply test_dataset_path."
            )

        prepared = self.workdir / "prepared_data"
        prepared.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(partitions, ignore_index=True)
        combined_path = prepared / "data_with_splits.csv"
        combined.to_csv(combined_path, index=False)
        for name in ("train", "val", "test"):
            combined.loc[combined["split"].eq(name), columns].to_csv(
                prepared / f"{name}.csv", index=False
            )
        return [combined_path]

    def prepare_data(self) -> tuple[list[Path], dict[str, Any]]:
        frame = _read_csv(self.dataset_path)
        self._validate_frame(
            frame, self.dataset_path, split=self.split_type == "predefined"
        )
        test_frame = None
        if self.test_dataset_path is not None:
            test_frame = _read_csv(self.test_dataset_path)
            self._validate_frame(test_frame, self.test_dataset_path, split=False)

        if self.split_type == "predefined":
            data_paths = self._prepare_predefined_files(frame, test_frame)
        else:
            data_paths = self._prepare_generated_split(frame, test_frame)

        audit = {
            "dataset_path": str(self.dataset_path),
            "test_dataset_path": (
                str(self.test_dataset_path) if self.test_dataset_path else None
            ),
            "rows": len(frame),
            "test_rows": len(test_frame) if test_frame is not None else None,
            "smiles_column": self.smiles_column,
            "targets": self.targets,
            "target_label_counts": {
                target: int(frame[target].notna().sum()) for target in self.targets
            },
            "split_type": self.split_type,
            "prepared_data_paths": [str(path) for path in data_paths],
        }
        return data_paths, audit

    def build_command(
        self,
        data_paths: list[Path],
        resume_checkpoints: list[Path] | None = None,
    ) -> list[str]:
        executable = str(self.training_cfg.get("executable", "chemprop")).strip()
        command = [executable, "train", "--data-path", *(str(path) for path in data_paths)]
        command.extend(["--output-dir", str(self.workdir)])
        command.extend(["--smiles-columns", self.smiles_column])
        command.extend(["--target-columns", *self.targets])
        command.extend(["--task-type", self.task])
        if resume_checkpoints:
            command.extend(["--checkpoint", *(str(path) for path in resume_checkpoints)])

        if data_paths[0].name == "data_with_splits.csv":
            command.extend(["--splits-column", "split"])
        else:
            command.extend(["--split-type", self.split_type])
            train_fraction = 1.0 - self.val_fraction - self.test_fraction
            command.extend(
                [
                    "--split-sizes",
                    str(train_fraction),
                    str(self.val_fraction),
                    str(self.test_fraction),
                ]
            )

        model = self.model_cfg
        command.extend(["--message-hidden-dim", str(int(model.get("message_hidden_dim", 300)))])
        command.extend(["--depth", str(int(model.get("depth", 3)))])
        command.extend(["--dropout", str(float(model.get("dropout", 0.0)))])
        command.extend(["--activation", str(model.get("activation", "RELU"))])
        command.extend(["--aggregation", str(model.get("aggregation", "norm"))])
        command.extend(["--aggregation-norm", str(float(model.get("aggregation_norm", 100.0)))])
        command.extend(["--ffn-hidden-dim", str(int(model.get("ffn_hidden_dim", 300)))])
        command.extend(["--ffn-num-layers", str(int(model.get("ffn_num_layers", 1)))])
        if bool(model.get("batch_norm", False)):
            command.append("--batch-norm")
        featurizers = _items(
            model.get("molecule_featurizers", []), name="ChempropConfig.molecule_featurizers"
        )
        if featurizers:
            command.extend(["--molecule-featurizers", *featurizers])
        if bool(model.get("no_descriptor_scaling", False)):
            command.append("--no-descriptor-scaling")

        training = self.training_cfg
        scalar_options = {
            "--batch-size": int(training.get("batch_size", 64)),
            "--num-workers": int(training.get("num_workers", 0)),
            "--epochs": int(training.get("num_epochs", 100)),
            "--patience": self.early_stopping_patience,
            "--warmup-epochs": int(training.get("warmup_epochs", 2)),
            "--init-lr": float(training.get("init_lr", 1e-4)),
            "--max-lr": float(training.get("max_lr", 1e-3)),
            "--final-lr": float(training.get("final_lr", 1e-4)),
            "--grad-clip": float(training.get("gradient_clip_val", 0.0)),
            "--ensemble-size": int(training.get("ensemble_size", 1)),
            "--num-replicates": int(training.get("num_replicates", 1)),
            "--data-seed": int(training.get("seed", 42)),
            "--pytorch-seed": int(training.get("pytorch_seed", 42)),
            "--accelerator": str(training.get("accelerator", "auto")),
            "--devices": str(training.get("devices", "auto")),
        }
        for option, value in scalar_options.items():
            command.extend([option, str(value)])
        command.extend(["--metrics", *self.metrics])
        command.extend(["--tracking-metric", self.tracking_metric])
        command.append("--show-individual-scores")
        if bool(training.get("save_data_splits", True)):
            version = self._chemprop_version()
            command.append(
                "--save-data-splits"
                if _version_tuple(version) >= (2, 3)
                else "--save-smiles-splits"
            )
        command.extend(self.extra_args)
        return command

    def _best_model_paths(self) -> list[Path]:
        """Return the validation-selected model from every ensemble member."""
        ensemble_size = int(self.training_cfg.get("ensemble_size", 1))
        paths = [
            self.workdir / f"model_{index}" / "best.pt"
            for index in range(ensemble_size)
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            formatted = "\n  ".join(str(path) for path in missing)
            raise FileNotFoundError("Chemprop best model(s) not found:\n  " + formatted)
        return paths

    def _export_training_history(self) -> Path | None:
        """Export TensorBoard or CSV scalar logs as one epoch-level CSV."""
        histories: list[pd.DataFrame] = []
        ensemble_size = int(self.training_cfg.get("ensemble_size", 1))
        for model_index in range(ensemble_size):
            log_root = self.workdir / f"model_{model_index}" / "trainer_logs"
            csv_paths = sorted(log_root.glob("version_*/metrics.csv"))
            if csv_paths:
                csv_path = max(csv_paths, key=lambda path: path.stat().st_mtime_ns)
                frame = pd.read_csv(csv_path)
            else:
                event_paths = sorted(log_root.glob("version_*/events.out.tfevents.*"))
                if not event_paths:
                    continue
                try:
                    from tensorboard.backend.event_processing.event_accumulator import (
                        EventAccumulator,
                    )
                except ImportError:
                    print(
                        "Chemprop history export skipped: tensorboard is not installed.",
                        flush=True,
                    )
                    return None
                event_path = max(event_paths, key=lambda path: path.stat().st_mtime_ns)
                accumulator = EventAccumulator(str(event_path))
                accumulator.Reload()
                scalar_tags = accumulator.Tags().get("scalars", [])
                series = []
                for tag in scalar_tags:
                    values = accumulator.Scalars(tag)
                    series.append(
                        pd.DataFrame(
                            {
                                "step": [item.step for item in values],
                                tag: [item.value for item in values],
                            }
                        ).drop_duplicates("step", keep="last")
                    )
                if not series:
                    continue
                frame = series[0]
                for values in series[1:]:
                    frame = frame.merge(values, on="step", how="outer")
            if "step" in frame:
                frame = frame.sort_values("step").groupby("step", as_index=False).first()
                frame.insert(0, "epoch", frame["step"].astype(int) + 1)
            frame.insert(0, "model", model_index)
            histories.append(frame)
        if not histories:
            return None
        output = self.workdir / "training_history.csv"
        pd.concat(histories, ignore_index=True).to_csv(output, index=False)
        return output

    def _evaluate_best_models(
        self,
        executable: str,
        process_environment: dict[str, str],
    ) -> tuple[Path, Path]:
        """Evaluate the independent test set using best validation models."""
        test_path = self.workdir / "prepared_data" / "test.csv"
        truth = pd.read_csv(test_path)
        prediction_input = self.workdir / "prepared_data" / "test_smiles.csv"
        truth[[self.smiles_column]].to_csv(prediction_input, index=False)
        raw_output = self.workdir / "best_model_predictions.csv"
        best_models = self._best_model_paths()
        command = [
            executable,
            "predict",
            "--test-path",
            str(prediction_input),
            "--preds-path",
            str(raw_output),
            "--smiles-columns",
            self.smiles_column,
            "--model-paths",
            *(str(path) for path in best_models),
            "--num-workers",
            str(int(self.training_cfg.get("num_workers", 0))),
            "--batch-size",
            str(int(self.training_cfg.get("batch_size", 64))),
            "--accelerator",
            str(self.training_cfg.get("accelerator", "auto")),
            "--devices",
            str(self.training_cfg.get("devices", "auto")),
        ]
        subprocess.run(
            command,
            cwd=self.workdir,
            env=process_environment,
            check=True,
        )
        predicted = pd.read_csv(raw_output)
        if len(predicted) != len(truth):
            raise ValueError(
                "Best-model prediction row count does not match the test dataset: "
                f"{len(predicted)} != {len(truth)}."
            )

        results = pd.DataFrame({self.smiles_column: truth[self.smiles_column]})
        metric_summary: dict[str, Any] = {}
        for target in self.targets:
            if target not in predicted:
                raise ValueError(
                    f"Chemprop prediction output is missing target {target!r}; "
                    f"available columns: {list(predicted.columns)}"
                )
            true_values = pd.to_numeric(truth[target], errors="coerce")
            predicted_values = pd.to_numeric(predicted[target], errors="coerce")
            results[f"{target}_true"] = true_values
            results[f"{target}_prediction"] = predicted_values
            results[f"{target}_error"] = predicted_values - true_values
            metric_summary[target] = _regression_metrics(true_values, predicted_values)

        official_output = self.workdir / "model_0" / "test_predictions.csv"
        if official_output.is_file():
            shutil.copy2(
                official_output,
                self.workdir / "final_model_test_predictions.csv",
            )
        prediction_path = self.workdir / "test_predictions.csv"
        metrics_path = self.workdir / "test_metrics.json"
        results.to_csv(prediction_path, index=False)
        metrics_path.write_text(
            json.dumps(
                {
                    "selection": "best_validation_checkpoint",
                    "models": [str(path) for path in best_models],
                    "targets": metric_summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("Independent test metrics — best validation checkpoint", flush=True)
        for target, metrics in metric_summary.items():
            values = "  ".join(
                f"{name}={metrics[name]:.4f}"
                for name in ("rmse", "mae", "r2", "pearson", "spearman", "kendall")
                if metrics[name] is not None
            )
            print(f"  {target} (n={metrics['n']}): {values}", flush=True)
        return prediction_path, metrics_path

    @staticmethod
    def _chemprop_version() -> str:
        try:
            return importlib.metadata.version("chemprop")
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    @staticmethod
    def _resolve_executable(value: str) -> str | None:
        """Find Chemprop even when ChemFlow was launched by absolute path."""
        candidate = Path(value).expanduser()
        if candidate.parent != Path("."):
            return str(candidate.resolve()) if candidate.is_file() else None
        discovered = shutil.which(value)
        if discovered:
            return discovered
        sibling = Path(sys.executable).resolve().with_name(value)
        return str(sibling) if sibling.is_file() else None

    def train(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        data_paths, audit = self.prepare_data()
        resume_checkpoints = self._resolve_resume_checkpoints()
        command = self.build_command(data_paths, resume_checkpoints)
        version = self._chemprop_version()
        resolved_executable = self._resolve_executable(command[0])
        if resolved_executable is not None:
            command[0] = resolved_executable
        process_environment = os.environ.copy()
        compatibility_environment: dict[str, str] = {}
        if _version_tuple(version) < (2, 3):
            # Chemprop 2.2 asks Lightning to reload the checkpoint it just
            # created without explicitly setting weights_only. PyTorch 2.6+
            # changed that default to True, which rejects Chemprop metric
            # objects. This official PyTorch override is safe here because the
            # checkpoint is created locally by this same training process.
            process_environment.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            compatibility_environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = (
                process_environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]
            )
        manifest = {
            "backend": "Chemprop v2",
            "chemprop_version": version,
            "config_path": str(self.config_path),
            "workdir": str(self.workdir),
            "data_audit": audit,
            "command": command,
            "command_text": shlex.join(command),
            "compatibility_environment": compatibility_environment,
            "early_stopping": {
                "implementation": "Chemprop v2 native Lightning callback",
                "monitor": self.tracking_metric,
                "patience": self.early_stopping_patience,
                "min_delta": 0.0,
                "state_restored_on_resume": False,
            },
            "resume": {
                "enabled": self.resume,
                "mode": "weights_only" if self.resume else None,
                "checkpoints": [
                    {
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                    for path in resume_checkpoints
                ],
                "optimizer_scheduler_epoch_restored": False,
            },
        }
        manifest_path = self.workdir / "chemprop_launch_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        print("Chemprop v2 training audit", flush=True)
        print(f"  Version: {version}", flush=True)
        print(f"  Training rows: {audit['rows']:,}", flush=True)
        if audit["test_rows"] is not None:
            print(f"  Independent test rows: {audit['test_rows']:,}", flush=True)
        print(f"  Targets: {', '.join(self.targets)}", flush=True)
        print(f"  Split: {self.split_type}", flush=True)
        print(f"  Output: {self.workdir}", flush=True)
        print(f"  Manifest: {manifest_path}", flush=True)
        print(
            "  Early stopping: enabled "
            f"(monitor={self.tracking_metric}, "
            f"patience={self.early_stopping_patience}, min_delta=0)",
            flush=True,
        )
        if resume_checkpoints:
            print("  Resume: enabled (weights-only continuation)", flush=True)
            for index, path in enumerate(resume_checkpoints):
                print(f"  Resume checkpoint [{index}]: {path}", flush=True)
                print(f"  Resume checkpoint SHA-256 [{index}]: {_sha256(path)}", flush=True)
            print(
                "  Resume state: model weights restored; optimizer, scheduler, "
                "and epoch counter start fresh",
                flush=True,
            )
            print(
                "  Epochs: num_epochs is the number of additional epochs",
                flush=True,
            )
            print(
                "  Early-stopping state: reset; a new patience window starts",
                flush=True,
            )
        else:
            print("  Resume: disabled", flush=True)
        if compatibility_environment:
            print(
                "  Compatibility: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "
                "for Chemprop 2.2 checkpoint reload",
                flush=True,
            )
        print(f"  Command: {shlex.join(command)}", flush=True)
        if self.dry_run:
            print("Dry run requested; Chemprop was not launched.", flush=True)
            return
        if version == "not installed" or resolved_executable is None:
            raise RuntimeError(
                "Chemprop v2 is not installed in this environment. Install the "
                "project dependencies, then rerun this command."
            )
        if not version.startswith("2."):
            raise RuntimeError(f"ChemFlow requires Chemprop v2; found {version}.")
        subprocess.run(
            command,
            cwd=self.workdir,
            env=process_environment,
            check=True,
        )
        history_path = self._export_training_history()
        if history_path is not None:
            print(f"Chemprop epoch history: {history_path}", flush=True)
        prediction_path, metrics_path = self._evaluate_best_models(
            command[0], process_environment
        )
        print(f"Chemprop best-model test predictions: {prediction_path}", flush=True)
        print(f"Chemprop best-model test metrics: {metrics_path}", flush=True)
