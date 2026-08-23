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
import yaml
from rdkit import Chem
import logging
from threading import Lock

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

from src.chemflow.machine_learning.data import DataSplitter
from src.chemflow.machine_learning.data.data_pipeline import featurize_array

from src.chemflow.machine_learning.train.utils import (
    build_split_config,
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
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def write_log(job_dir, message):
    """Append a timestamped message to the job log file."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    log_path = job_dir / "training.log"
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def write_status(job_dir, status, progress, extra=None):
    """Write the current training status to ``status.json``."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": status,
        "progress": int(progress),
    }

    if extra:
        payload.update(_json_safe(extra))

    status_path = job_dir / "status.json"
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)


def _log_and_write(job_dir, level, message, *args):
    """Send a message to both the Python logger and the job log file."""
    log_method = getattr(logger, level)
    log_method(message, *args)

    if job_dir is not None:
        rendered = message % args if args else message
        write_log(job_dir, f"{level.upper()}: {rendered}")


# ============================================================
# Data utilities
# ============================================================

def clean_raw_dataframe(df, smiles_col, target_col, job_dir=None, split_label=None):
    """
    Remove rows with missing targets, missing/empty SMILES, or invalid SMILES.

    Cleaning messages are written both to the standard Python logger and, when
    ``job_dir`` is provided, to ``training.log`` in the job directory.
    """
    label = f"[{split_label}] " if split_label else ""

    if smiles_col not in df.columns:
        message = f"smiles_col/X_col '{smiles_col}' not found in dataframe."
        _log_and_write(job_dir, "error", "%s%s", label, message)
        raise ValueError(message)

    if target_col not in df.columns:
        message = f"target_col/y_col '{target_col}' not found in dataframe."
        _log_and_write(job_dir, "error", "%s%s", label, message)
        raise ValueError(message)

    clean_df = df.copy()
    n_initial = len(clean_df)

    _log_and_write(
        job_dir,
        "info",
        "%sStarting dataframe cleaning with %d rows.",
        label,
        n_initial,
    )

    # Remove missing SMILES
    missing_smiles_mask = clean_df[smiles_col].isna()
    n_missing_smiles = int(missing_smiles_mask.sum())

    if n_missing_smiles:
        _log_and_write(
            job_dir,
            "warning",
            "%sRemoving %d rows with missing SMILES.",
            label,
            n_missing_smiles,
        )

    clean_df = clean_df.loc[~missing_smiles_mask].copy()

    # Remove missing target values
    missing_target_mask = clean_df[target_col].isna()
    n_missing_target = int(missing_target_mask.sum())

    if n_missing_target:
        _log_and_write(
            job_dir,
            "warning",
            "%sRemoving %d rows with missing target values.",
            label,
            n_missing_target,
        )

    clean_df = clean_df.loc[~missing_target_mask].copy()

    # Convert SMILES to strings and strip surrounding whitespace
    clean_df[smiles_col] = clean_df[smiles_col].astype(str).str.strip()

    # Remove empty / whitespace-only SMILES
    empty_smiles_mask = clean_df[smiles_col].eq("")
    n_empty_smiles = int(empty_smiles_mask.sum())

    if n_empty_smiles:
        _log_and_write(
            job_dir,
            "warning",
            "%sRemoving %d rows with empty SMILES.",
            label,
            n_empty_smiles,
        )

    clean_df = clean_df.loc[~empty_smiles_mask].copy()

    # Validate SMILES with RDKit
    valid_smiles_mask = clean_df[smiles_col].apply(
        lambda smiles: Chem.MolFromSmiles(smiles) is not None
    )
    n_invalid_smiles = int((~valid_smiles_mask).sum())

    if n_invalid_smiles:
        invalid_smiles = clean_df.loc[~valid_smiles_mask, smiles_col].tolist()
        invalid_examples = invalid_smiles[:10]

        _log_and_write(
            job_dir,
            "warning",
            "%sRemoving %d rows with invalid SMILES. Examples: %s",
            label,
            n_invalid_smiles,
            invalid_examples,
        )

    clean_df = clean_df.loc[valid_smiles_mask].copy()
    clean_df = clean_df.reset_index(drop=True)

    if clean_df.empty:
        message = "No usable rows after removing missing/invalid SMILES and target values."
        _log_and_write(job_dir, "error", "%s%s", label, message)
        raise ValueError(message)

    n_removed = n_initial - len(clean_df)
    _log_and_write(
        job_dir,
        "info",
        "%sDataframe cleaning complete: %d -> %d rows (%d removed).",
        label,
        n_initial,
        len(clean_df),
        n_removed,
    )

    return clean_df


def get_feature_types(featurization_config):
    feature_types = featurization_config.get("features", None)
    if feature_types is None:
        logger.error("Missing molecular representation config. Expected 'features' in featurization config.")
        raise ValueError("Missing molecular representation config. Expected 'features' in featurization config.")
    return list(feature_types)


def get_split_npz_path(split_config):
    save_split_data = bool(split_config.get("save_split_data", True))

    if not save_split_data:
        return None

    split_method = split_config["split_method"]
    random_seed = split_config.get("random_seed", 42)

    save_dir = Path(split_config.get("save_dir", None))
    if save_dir is None:
        logger.error("Missing 'save_dir' in split_config for saving split data.")
        raise ValueError("Missing 'save_dir' in split_config for saving split data.")
    save_dir.mkdir(parents=True, exist_ok=True)

    prefix_name = split_config.get("split_name", split_config.get("prefix_name", f"{split_method}_seed{random_seed}"))

    return save_dir / f"{prefix_name}_split_data.npz"


def save_cached_split_arrays(
    training_data_file,
    split_config,
    job_dir,
    X_train,
    y_train,
    X_test,
    y_test,
    X_valid,
    y_valid=None,
    split_source=None,
):
    if not split_config.get("save_split_data", True):
        return None

    save_dir = Path(split_config.get("save_dir", "split_data"))
    save_dir.mkdir(parents=True, exist_ok=True)

    split_name = split_config.get("split_name", split_config.get("prefix_name", Path(training_data_file).stem))

    cache_path = save_dir / f"{split_name}_split_data.npz"

    arrays_to_save = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    if X_valid is not None:
        arrays_to_save["X_valid"] = X_valid

    if y_valid is not None:
        arrays_to_save["y_valid"] = y_valid

    if split_source is not None:
        arrays_to_save["split_source"] = np.asarray([split_source])

    np.savez_compressed(cache_path, **arrays_to_save)
    logger.info("Saved cached split NPZ to: %s", cache_path)
    return cache_path


def load_pre_split_features(training_data_file, job_dir):
    logger.info("Loading pre-split dataset: %s", training_data_file)

    payload = load_training_data(training_data_file)
    split_data = payload["data"]

    if not isinstance(split_data, dict):
        logger.error("Pre-split dataset must be a dictionary containing X_train, X_test, y_train, and y_test.")
        raise ValueError(
            "Pre-split datasets must be stored as an .npz file containing "
            "X_train, X_test, y_train, and y_test arrays."
        )

    required_keys = ["X_train", "X_test", "y_train", "y_test"]
    missing_keys = [key for key in required_keys if key not in split_data]

    if missing_keys:
        logger.error(
            "Pre-split dataset is missing required arrays: %s. Expected keys: %s.",
            ", ".join(missing_keys),
            required_keys,
        )
        raise ValueError(
            "Pre-split dataset is missing required arrays: "
            f"{', '.join(missing_keys)}. Expected keys: {required_keys}."
        )

    has_x_valid = "X_valid" in split_data
    has_y_valid = "y_valid" in split_data

    if has_x_valid != has_y_valid:
        logger.error(
            "Pre-split dataset must provide both X_valid and y_valid, or neither."
        )
        raise ValueError(
            "Pre-split dataset must provide both X_valid and y_valid, or neither."
        )

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    X_valid = split_data.get("X_valid") if has_x_valid else None

    logger.info("Loaded pre-split X_train shape: %s", X_train.shape)
    logger.info("Loaded pre-split y_train shape: %s", y_train.shape)
    logger.info("Loaded pre-split X_test shape: %s", X_test.shape)
    logger.info("Loaded pre-split y_test shape: %s", y_test.shape)

    if X_valid is not None:
        logger.info("Loaded pre-split X_valid shape: %s", X_valid.shape)

    y_valid = split_data.get("y_valid") if has_y_valid else None
    
    return X_train, y_train, X_test, y_test, X_valid, y_valid, "pre_split_npz"


def load_split_column_features(
    training_data_file,
    smiles_col,
    y_col,
    feature_types,
    split_column,
    split_config,
    job_dir,
):
    payload = load_training_data(training_data_file)
    df = payload["data"]

    if not isinstance(df, pd.DataFrame):
        return None

    if split_column not in df.columns:
        return None

    logger.info("Using existing split column '%s' from training data.", split_column)

    clean_df = clean_raw_dataframe(
        df=df,
        smiles_col=smiles_col,
        target_col=y_col,
        job_dir=job_dir,
        split_label="split_column",
    )

    clean_df = clean_df.loc[pd.notna(clean_df[split_column])].copy()

    if len(clean_df) == 0:
        logger.warning(
            "Split column '%s' had no usable rows after filtering; falling back to raw split.",
            split_column,
        )
        return None

    split_values = clean_df[split_column].astype(str).str.strip().str.lower()
    train_mask = split_values.isin({"train", "training"})
    test_mask = split_values.isin({"test", "testing"})
    valid_mask = split_values.isin({"val", "valid", "validation", "dev"})

    if not train_mask.any() or not test_mask.any():
        logger.warning(
            "Split column '%s' does not contain both train and test rows; falling back to raw split.",
            split_column,
        )
        return None

    train_df = clean_df.loc[train_mask].copy()
    test_df = clean_df.loc[test_mask].copy()
    valid_df = clean_df.loc[valid_mask].copy()

    X_train, y_train, _ = featurize_array(
        train_df[smiles_col].to_numpy(),
        train_df[y_col].to_numpy(),
        feature_types,
    )

    X_test, y_test, _ = featurize_array(
        test_df[smiles_col].to_numpy(),
        test_df[y_col].to_numpy(),
        feature_types,
    )

    X_valid = None
    y_valid = None

    if len(valid_df) > 0:
        X_valid, y_valid, _ = featurize_array(
            valid_df[smiles_col].to_numpy(),
            valid_df[y_col].to_numpy(),
            feature_types,
        )

    x_train_shape = getattr(X_train, "shape", None)
    x_test_shape = getattr(X_test, "shape", None)
    x_valid_shape = getattr(X_valid, "shape", None) if X_valid is not None else None

    logger.info("Loaded split-column X_train shape: %s", x_train_shape)
    logger.info("Loaded split-column X_test shape: %s", x_test_shape)

    if X_valid is not None:
        logger.info("Loaded split-column X_valid shape: %s", x_valid_shape)

    save_cached_split_arrays(
        training_data_file=training_data_file,
        split_config=split_config or {},
        job_dir=job_dir,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_valid=X_valid,
        y_valid=y_valid,
        split_source="pre_split_column",
    )

    return X_train, y_train, X_test, y_test, X_valid, y_valid, "pre_split_column"


def load_explicit_split_features(
    train_data_file,
    test_data_file,
    validation_data_file,
    smiles_col,
    y_col,
    feature_types,
    split_config,
    job_dir,
):
    logger.info("Loading explicit split datasets: train=%s, test=%s, valid=%s", train_data_file, test_data_file, validation_data_file)

    def load_split_frame(split_file, split_label):
        if split_file is None:
            return None

        payload = load_training_data(split_file)
        df = payload["data"]

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Explicit split file for '{split_label}' must contain tabular data."
            )

        return clean_raw_dataframe(
            df=df,
            smiles_col=smiles_col,
            target_col=y_col,
            job_dir=job_dir,
            split_label=split_label,
        )

    train_df = load_split_frame(train_data_file, "train")
    test_df = load_split_frame(test_data_file, "test")
    valid_df = load_split_frame(validation_data_file, "valid")

    if train_df is None:
        raise ValueError("train_data_file is required when using explicit split files.")

    if test_df is None:
        raise ValueError("test_data_file is required when using explicit split files.")

    X_train, y_train, _ = featurize_array(
        train_df[smiles_col].to_numpy(),
        train_df[y_col].to_numpy(),
        feature_types,
    )

    X_test, y_test, _ = featurize_array(
        test_df[smiles_col].to_numpy(),
        test_df[y_col].to_numpy(),
        feature_types,
    )

    X_valid = None
    y_valid = None

    if valid_df is not None and len(valid_df) > 0:
        X_valid, y_valid, _ = featurize_array(
            valid_df[smiles_col].to_numpy(),
            valid_df[y_col].to_numpy(),
            feature_types,
        )

    x_train_shape = getattr(X_train, "shape", None)
    x_test_shape = getattr(X_test, "shape", None)

    logger.info("Loaded explicit split X_train shape: %s", x_train_shape)
    logger.info("Loaded explicit split X_test shape: %s", x_test_shape)

    if X_valid is not None:
        logger.info("Loaded explicit split X_valid shape: %s", getattr(X_valid, "shape", None))

    save_cached_split_arrays(
        training_data_file=train_data_file,
        split_config=split_config or {},
        job_dir=job_dir,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_valid=X_valid,
        y_valid=y_valid,
        split_source="explicit_split_files",
    )

    return X_train, y_train, X_test, y_test, X_valid, y_valid, "explicit_split_files"


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
        logger.info("Checking split feature data: %s", split_npz_path)

    if split_npz_path is not None and split_npz_path.exists():
        logger.info("Existing featurized split data found. Loading NPZ...")

        data = np.load(split_npz_path, allow_pickle=False)

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

        X_valid = data["X_valid"] if "X_valid" in data.files else None
        y_valid = data["y_valid"] if "y_valid" in data.files else None

        logger.info("Loaded X_train shape: %s", X_train.shape)
        logger.info("Loaded X_test shape: %s", X_test.shape)

        if X_valid is not None:
            logger.info("Loaded X_valid shape: %s", X_valid.shape)

        return X_train, y_train, X_test, y_test, X_valid, y_valid, "split"

    logger.info("No existing split data found. Loading raw training data...")
    logger.info("Training data file: %s", training_data_file)

    payload = load_training_data(training_data_file)
    df = payload["data"]

    if isinstance(df, dict):
        try:
            return load_pre_split_features(training_data_file=training_data_file, job_dir=job_dir)
        except ValueError:
            pass

    split_method = str(split_config.get("split_method", "random")).strip().lower()

    if split_method == "splitted":
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "split_method='splitted' requires a tabular dataset containing a split column."
            )

        split_column = split_config.get("split_column", split_config.get("split_col", "split"))

        if not split_column:
            raise ValueError(
                "split_method='splitted' requires 'split_column' in data_split config."
            )

        if split_column not in df.columns:
            raise ValueError(
                f"split_method='splitted' requested split_column '{split_column}', "
                f"but that column was not found in the dataset."
            )

        logger.info(
            "split_method='splitted': using column '%s' to define train/test/validation sets.",
            split_column,
        )
        write_log(
            job_dir,
            f"Using existing split column '{split_column}' because split_method='splitted'.",
        )

        split_column_result = load_split_column_features(
            training_data_file=training_data_file,
            smiles_col=smiles_col,
            y_col=y_col,
            feature_types=feature_types,
            split_column=split_column,
            split_config=split_config,
            job_dir=job_dir,
        )

        if split_column_result is None:
            raise ValueError(
                f"Unable to create train/test sets from split column '{split_column}'. "
                "The column must contain both train/training and test/testing labels."
            )

        return split_column_result

    clean_df = clean_raw_dataframe(
        df=df,
        smiles_col=smiles_col,
        target_col=y_col,
        job_dir=job_dir,
        split_label="raw",
    )

    smiles = clean_df[smiles_col].to_numpy()
    y_raw = clean_df[y_col].to_numpy()

    splitter = DataSplitter(split_config)

    write_status(job_dir, "running", 25)
    logger.info("Splitting raw SMILES using %s", split_config["split_method"])

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
    logger.info("Featurizing train/test/valid arrays...")

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

    x_train_shape = getattr(X_train, "shape", None)
    x_test_shape = getattr(X_test, "shape", None)
    logger.info("X_train shape after featurization: %s", x_train_shape)
    logger.info("X_test shape after featurization: %s", x_test_shape)

    if X_valid is not None:
        logger.info("X_valid shape after featurization: %s", X_valid.shape)

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
        logger.info("Saved featurized split NPZ to: %s", split_npz_path)

    return X_train, y_train, X_test, y_test, X_valid, y_valid, "split"


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
        "split_mode": feature_config.get("split_mode"),
        "split_source": feature_config.get("split_source"),
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

        data_config = training_config["data"]
        featurization_config = training_config["featurization"]

        task_type = str(training_config.get("task_type", data_config.get("task_type", ""))).lower()

        if task_type not in ["classification", "regression"]:
            raise ValueError(f"Invalid task_type: {task_type}")

        training_data_file = data_config["data_file"]
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
        write_log(job_dir, f"Feature types: {feature_types}")

        feature_n_bits = featurization_config.get("fp_bits", featurization_config.get("n_bits"))

        if feature_n_bits is not None:
            featurization_config["fp_bits"] = int(feature_n_bits)

        write_status(job_dir, "running", 10)

        split_config = (
            build_split_config(training_config["data_split"])
            if "data_split" in training_config
            else {}
        )

        explicit_train_data_file = data_config.get("train_data_file")
        explicit_test_data_file = data_config.get("test_data_file")
        explicit_validation_data_file = data_config.get("validation_data_file", data_config.get("valid_data_file"))

        if explicit_train_data_file or explicit_test_data_file or explicit_validation_data_file:
            X_train, y_train, X_test, y_test, X_valid, y_valid, split_source = load_explicit_split_features(
                train_data_file=explicit_train_data_file or training_data_file,
                test_data_file=explicit_test_data_file,
                validation_data_file=explicit_validation_data_file,
                smiles_col=smiles_col,
                y_col=y_col,
                feature_types=feature_types,
                split_config=split_config,
                job_dir=job_dir,
            )
        else:
            split_npz_path = get_split_npz_path(split_config) if split_config else None

            X_train, y_train, X_test, y_test, X_valid, y_valid, split_source = load_or_create_split_features(
                training_data_file=training_data_file,
                smiles_col=smiles_col,
                y_col=y_col,
                feature_types=feature_types,
                split_config=split_config,
                split_npz_path=split_npz_path,
                job_dir=job_dir,
            )

        x_train_shape = getattr(X_train, "shape", None)
        x_test_shape = getattr(X_test, "shape", None)
        x_valid_shape = getattr(X_valid, "shape", None) if X_valid is not None else None

        feature_config_to_save = {
            **featurization_config,
            "feature_types": feature_types,
            "representations": feature_types,
            "features": feature_types,
            "fp_bits": featurization_config.get("fp_bits"),
            "smiles_col": smiles_col,
            "target_col": y_col,
            "y_col": y_col,
            "split_method": split_config.get("split_method") if split_source == "split" else None,
            "split_mode": split_source,
            "split_name": (
                split_config.get("split_name", split_config.get("prefix_name"))
                if split_source == "split"
                else split_config.get("split_name", Path(training_data_file).stem)
            ),
            "pre_split_data": split_source != "split",
            "split_source": split_source,
            "feature_array_shapes": {
                "X_train": None if x_train_shape is None else list(x_train_shape),
                "X_test": None if x_test_shape is None else list(x_test_shape),
                "X_valid": None if x_valid_shape is None else list(x_valid_shape),
            },
        }

        with open(job_dir / "feature_config.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(feature_config_to_save), f, indent=4)

        write_status(job_dir, "running", 60)

        hyperparameter_tuning = bool(
            training_config.get("hyperparameter_tuning", True)
        )

        model_cfgs = training_config["models"]
        if isinstance(model_cfgs, dict):
            model_cfgs = list(model_cfgs.values())

        total_models = len(model_cfgs)
        if total_models == 0:
            raise ValueError("No models specified in training_config['models'].")

        if hyperparameter_tuning:
            total_steps = total_models
        else:
            total_steps = 0
            for model_cfg in model_cfgs:
                merged_cfg = dict(training_config)
                merged_cfg.pop("models", None)
                merged_cfg.update(model_cfg)
                seeds = merged_cfg.get("seeds", merged_cfg.get("eval_seeds", [42]))
                total_steps += len(seeds)

        total_steps = max(total_steps, 1)
        progress_state = {"completed": 0}
        progress_lock = Lock()

        def on_training_step_complete():
            with progress_lock:
                progress_state["completed"] += 1
                ratio = progress_state["completed"] / total_steps
                current_progress = 60 + int(round(ratio * 39))

            write_status(
                job_dir,
                "running",
                min(current_progress, 99),
                {
                    "completed_steps": progress_state["completed"],
                    "total_steps": total_steps,
                },
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
                output_dir=str(job_dir),
                feature_config=feature_config_to_save,
                progress_callback=on_training_step_complete,
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
                output_dir=str(job_dir),
                feature_config=feature_config_to_save,
                progress_callback=on_training_step_complete,
            )

        write_status(job_dir, "completed", 100, {"metrics": _json_safe(results)})
        write_log(job_dir, "Training completed.")

    except Exception as e:
        write_log(job_dir, "Training failed.")
        write_log(job_dir, traceback.format_exc())
        write_status(job_dir, "failed", 0, {"error": str(e)})
        raise e


def load_training_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if suffix in {".yaml", ".yml"}:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            return {}
        return cfg

    if suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("Python 3.11+ is required to load TOML config files.")
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    raise ValueError(
        "Unsupported config format. Use .json, .yaml, .yml, or .toml. "
        f"Got: {config_path}"
    )


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Missing config path. Usage: python train_runner.py path/to/config.json|yaml|toml"
        )

    config_path = Path(sys.argv[1]).resolve()
    training_config = load_training_config(config_path)
    train(training_config)


if __name__ == "__main__":
    main()