# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

"""
ChemFlow model training runner.

Supports:
1. Hyperparameter tuning with RandomizedSearchCV.
2. Regular training with custom model_params from YAML/JSON config.
"""

import json
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from src.chemflow.machine_learning.data import DataSplitter
from src.chemflow.machine_learning.data.data_pipeline import featurize_array

from src.chemflow.machine_learning.train.utils import (
    load_training_data,
    _json_safe,
)

from src.chemflow.machine_learning.train.regular_training import (
    regular_training_multiple_models,
)

from src.chemflow.machine_learning.train.hyperparameter_tuning import (
    tune_parameters_multiple_model,
)


# ============================================================
# Logging utilities
# ============================================================

def write_status(job_dir, status, progress=0, extra=None):
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "status": status,
        "progress": progress,
    }

    if extra:
        data.update(extra)

    with open(job_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=4)


def write_log(job_dir, message):
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    with open(job_dir / "training.log", "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")
        f.flush()


# ============================================================
# Data utilities
# ============================================================

def clean_raw_dataframe(df, smiles_col, target_col):
    if smiles_col not in df.columns:
        raise ValueError(f"smiles_col/X_col '{smiles_col}' not found in dataframe.")

    if target_col not in df.columns:
        raise ValueError(f"target_col/y_col '{target_col}' not found in dataframe.")

    clean_df = df.copy()
    clean_df = clean_df.loc[pd.notna(clean_df[smiles_col])].copy()
    clean_df = clean_df.loc[pd.notna(clean_df[target_col])].copy()

    clean_df[smiles_col] = clean_df[smiles_col].astype(str)
    clean_df = clean_df.reset_index(drop=True)

    if len(clean_df) == 0:
        raise ValueError("No usable rows after removing missing SMILES/target values.")

    return clean_df


def get_feature_types(featurization_config):
    feature_types = featurization_config.get(
        "features",
        featurization_config.get("representations"),
    )

    if feature_types is None:
        raise ValueError(
            "Missing molecular representation config. "
            "Expected 'features' or 'representations' in featurization config."
        )

    return feature_types


def get_split_npz_path(split_config):
    save_split_data = bool(
        split_config.get(
            "save_split_data",
            split_config.get("save_dataset", True),
        )
    )

    if not save_split_data:
        return None

    split_method = split_config["split_method"]
    random_seed = split_config.get("random_seed", 42)

    save_dir = Path(split_config.get("save_dir", "split_data"))
    save_dir.mkdir(parents=True, exist_ok=True)

    prefix_name = split_config.get(
        "split_name",
        split_config.get("prefix_name", f"{split_method}_split_seed{random_seed}"),
    )

    return save_dir / f"{prefix_name}_split_data.npz"


def load_or_create_split_features(
    training_data_file,
    smiles_col,
    y_col,
    feature_types,
    split_config,
    split_npz_path,
    job_dir,
):
    if split_npz_path is not None:
        write_log(job_dir, f"Checking split feature data: {split_npz_path}")

    if split_npz_path is not None and split_npz_path.exists():
        write_log(job_dir, "Existing featurized split data found. Loading NPZ...")

        data = np.load(split_npz_path, allow_pickle=False)

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

        X_valid = data["X_valid"] if "X_valid" in data.files else None

        write_log(job_dir, f"Loaded X_train shape: {X_train.shape}")
        write_log(job_dir, f"Loaded X_test shape: {X_test.shape}")

        if X_valid is not None:
            write_log(job_dir, f"Loaded X_valid shape: {X_valid.shape}")

        return X_train, y_train, X_test, y_test, X_valid

    write_log(job_dir, "No existing split data found. Loading raw training data...")
    write_log(job_dir, f"Training data file: {training_data_file}")

    payload = load_training_data(training_data_file)
    df = payload["data"]

    clean_df = clean_raw_dataframe(
        df=df,
        smiles_col=smiles_col,
        target_col=y_col,
    )

    smiles = clean_df[smiles_col].to_numpy()
    y_raw = clean_df[y_col].to_numpy()

    splitter = DataSplitter(split_config)

    write_status(job_dir, "running", 25)
    write_log(job_dir, f"Splitting raw SMILES using {split_config['split_method']}...")

    split_result = splitter.split_data(
        X=smiles,
        y=y_raw,
        smiles=smiles,
    )

    X_train_raw = split_result.X_train
    X_test_raw = split_result.X_test
    X_valid_raw = split_result.X_valid

    y_train_raw = split_result.y_train
    y_test_raw = split_result.y_test
    y_valid_raw = split_result.y_valid

    train_indices = split_result.train_indices
    test_indices = split_result.test_indices
    valid_indices = split_result.valid_indices

    write_status(job_dir, "running", 40)
    write_log(job_dir, "Featurizing train/test/valid arrays...")

    X_train, y_train, _ = featurize_array(
        X_train_raw,
        y_train_raw,
        feature_types,
    )

    X_test, y_test, _ = featurize_array(
        X_test_raw,
        y_test_raw,
        feature_types,
    )

    X_valid = None
    y_valid = None

    if X_valid_raw is not None and y_valid_raw is not None and len(X_valid_raw) > 0:
        X_valid, y_valid, _ = featurize_array(
            X_valid_raw,
            y_valid_raw,
            feature_types,
        )

    write_log(job_dir, f"X_train shape after featurization: {X_train.shape}")
    write_log(job_dir, f"X_test shape after featurization: {X_test.shape}")

    if X_valid is not None:
        write_log(job_dir, f"X_valid shape after featurization: {X_valid.shape}")

    if split_npz_path is not None:
        arrays_to_save = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_indices": train_indices,
            "test_indices": test_indices,
        }

        if X_valid is not None:
            arrays_to_save["X_valid"] = X_valid

        if y_valid is not None:
            arrays_to_save["y_valid"] = y_valid

        if valid_indices is not None:
            arrays_to_save["valid_indices"] = valid_indices

        np.savez_compressed(split_npz_path, **arrays_to_save)
        write_log(job_dir, f"Saved featurized split NPZ to: {split_npz_path}")

    return X_train, y_train, X_test, y_test, X_valid


# ============================================================
# Model loading helper
# ============================================================

def load_pickle_model(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "model" in obj:
        return obj

    return {
        "model": obj,
        "model_name": None,
        "task_type": None,
        "feature_config": None,
        "training_config": None,
        "metrics": None,
    }


def get_loaded_model_info(model_package):
    feature_config = model_package.get("feature_config") or {}
    training_config = model_package.get("training_config") or {}
    metrics = model_package.get("metrics") or {}

    return {
        "model_name": model_package.get("model_name") or training_config.get("model_name"),
        "task_type": model_package.get("task_type") or training_config.get("task_type"),
        "feature_types": feature_config.get("feature_types"),
        "features": feature_config.get("features"),
        "n_bits": feature_config.get("n_bits"),
        "fp_bits": feature_config.get("fp_bits"),
        "smiles_col": feature_config.get("smiles_col"),
        "X_col": training_config.get("data", {}).get("X_col"),
        "target_col": feature_config.get("target_col"),
        "y_col": training_config.get("data", {}).get("y_col"),
        "split_method": feature_config.get("split_method"),
        "split_name": feature_config.get("split_name"),
        "feature_array_shapes": feature_config.get("feature_array_shapes"),
        "best_params": metrics.get("best_params"),
        "model_params": metrics.get("model_params"),
        "refit_metric": metrics.get("refit_metric"),
    }


# ============================================================
# Training runner
# ============================================================

def train(training_config):
    job_dir = Path(training_config.get("workdir", "training_job")).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    write_status(job_dir, "running", 0)
    write_log(job_dir, "Training started.")
    write_log(job_dir, f"Config: {json.dumps(_json_safe(training_config), indent=2)}")

    try:
        if "models" not in training_config or not training_config["models"]:
            raise ValueError("No models specified in training_config['models'].")

        if "data" not in training_config:
            raise ValueError("Missing 'data' section in training_config.")

        if "featurization" not in training_config:
            raise ValueError("Missing 'featurization' section in training_config.")

        if "data_split" not in training_config:
            raise ValueError("Missing 'data_split' section in training_config.")

        data_config = training_config["data"]
        featurization_config = training_config["featurization"]
        split_config = training_config["data_split"]

        task_type = str(training_config.get("task_type", data_config.get("task_type", ""))).lower()

        if task_type not in ["classification", "regression"]:
            raise ValueError(f"Invalid task_type: {task_type}")

        training_data_file = data_config["training_data_file"]
        smiles_col = data_config.get("X_col", data_config.get("smiles_col"))
        y_col = data_config.get("y_col", data_config.get("target_col"))

        if smiles_col is None:
            raise ValueError("Missing X_col/smiles_col in data config.")

        if y_col is None:
            raise ValueError("Missing y_col/target_col in data config.")

        n_classes = data_config.get("n_classes")

        if n_classes is not None:
            n_classes = int(n_classes)
            write_log(job_dir, f"Number of classes: {n_classes}")

        feature_types = get_feature_types(featurization_config)

        feature_n_bits = featurization_config.get("fp_bits", featurization_config.get("n_bits"))

        if feature_n_bits is not None:
            featurization_config["fp_bits"] = int(feature_n_bits)

        write_status(job_dir, "running", 10)

        split_npz_path = get_split_npz_path(split_config)

        X_train, y_train, X_test, y_test, X_valid = load_or_create_split_features(
            training_data_file=training_data_file,
            smiles_col=smiles_col,
            y_col=y_col,
            feature_types=feature_types,
            split_config=split_config,
            split_npz_path=split_npz_path,
            job_dir=job_dir,
        )

        feature_config_to_save = {
            **featurization_config,
            "feature_types": feature_types,
            "representations": feature_types,
            "features": feature_types,
            "fp_bits": featurization_config.get("fp_bits"),
            "smiles_col": smiles_col,
            "target_col": y_col,
            "y_col": y_col,
            "split_method": split_config.get("split_method"),
            "split_name": split_config.get("split_name", split_config.get("prefix_name")),
            "feature_array_shapes": {
                "X_train": list(X_train.shape),
                "X_test": list(X_test.shape),
                "X_valid": None if X_valid is None else list(X_valid.shape),
            },
        }

        with open(job_dir / "feature_config.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(feature_config_to_save), f, indent=4)

        write_status(job_dir, "running", 60)

        hyperparameter_tuning = bool(
            training_config.get("hyperparameter_tuning", True)
        )

        if hyperparameter_tuning:
            write_log(job_dir, "Starting hyperparameter tuning...")

            results = tune_parameters_multiple_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cfgs=training_config["models"],
                parent_config=training_config,
                n_classes=n_classes,
                output_dir=job_dir,
                feature_config=feature_config_to_save,
            )

        else:
            write_log(job_dir, "Starting regular training with custom model_params...")

            results = regular_training_multiple_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_configs=training_config["models"],
                parent_config=training_config,
                output_dir=job_dir,
                feature_config=feature_config_to_save,
            )

        write_status(job_dir, "completed", 100, {"metrics": _json_safe(results)})
        write_log(job_dir, "Training completed.")

    except Exception as e:
        write_log(job_dir, "Training failed.")
        write_log(job_dir, traceback.format_exc())
        write_status(job_dir, "failed", 0, {"error": str(e)})
        raise e


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Missing config path. Usage: python train_runner.py path/to/config.json"
        )

    config_path = Path(sys.argv[1]).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        training_config = json.load(f)

    train(training_config)


if __name__ == "__main__":
    main()