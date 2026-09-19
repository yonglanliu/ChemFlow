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
from rdkit import Chem, rdBase
import logging
from threading import Lock

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

from chemflow.machine_learning.data import DataSplitter
from chemflow.machine_learning.data.data_pipeline import featurize_array
from chemflow.featurization import (
    DESC_NAMES,
    DESC_TYPES,
    RDKIT2D_DESC_NAMES,
    RDKIT2D_DESC_TYPES,
)

from chemflow.machine_learning.train.utils import (
    build_split_config,
    load_training_data,
    _json_safe,
)

from chemflow.machine_learning.train.regular_training import (
    regular_training_multiple_models,
)

from chemflow.machine_learning.train.hyperparameter_tuning import (
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
    if isinstance(feature_types, str):
        feature_types = [feature_types]
    return [str(feature).strip().lower() for feature in feature_types]


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


def get_split_csv_path(split_config):
    """Return the human-readable split path when split caching is enabled."""
    if not split_config.get("save_split_data", True):
        return None

    save_dir = Path(split_config.get("save_dir", "split_data"))
    save_dir.mkdir(parents=True, exist_ok=True)
    split_method = split_config.get("split_method", "split")
    random_seed = split_config.get("random_seed", 42)
    split_name = split_config.get(
        "split_name",
        split_config.get("prefix_name", f"{split_method}_seed{random_seed}"),
    )
    return save_dir / f"{split_name}_split_data.csv"


def save_split_dataframe(
    frame,
    split_config,
    smiles_col=None,
    target_col=None,
    train_indices=None,
    valid_indices=None,
    test_indices=None,
    split_labels=None,
):
    """Save source columns plus a canonical train/validation/test column."""
    csv_path = get_split_csv_path(split_config)
    if csv_path is None:
        return None

    output = frame.copy().reset_index(drop=True)
    normalized_columns = {
        str(column).strip().lower().replace("_", " "): column
        for column in output.columns
    }
    name_column = next(
        (
            normalized_columns[candidate]
            for candidate in (
                "molecule name",
                "compound name",
                "molecule id",
                "compound id",
                "name",
            )
            if candidate in normalized_columns
        ),
        None,
    )
    selected_columns = []
    if name_column is not None:
        selected_columns.append(name_column)
    for column in (smiles_col, target_col):
        if column is not None and column not in selected_columns:
            if column not in output.columns:
                raise ValueError(
                    f"Cannot export split CSV because column '{column}' is missing."
                )
            selected_columns.append(column)
    if selected_columns:
        output = output.loc[:, selected_columns].copy()
    if split_labels is not None:
        labels = np.asarray(split_labels, dtype=str)
        if len(labels) != len(output):
            raise ValueError("Split labels must have the same length as the dataframe.")
        output["split"] = labels
    else:
        partition_frames = []
        partitions = (
            (train_indices, "train"),
            (valid_indices, "validation"),
            (test_indices, "test"),
        )
        for indices, label in partitions:
            if indices is None:
                continue
            indices = np.asarray(indices, dtype=int)
            if indices.size:
                partition = output.iloc[indices].copy()
                partition["split"] = label
                partition_frames.append(partition)
        output = (
            pd.concat(partition_frames, ignore_index=True)
            if partition_frames
            else output.iloc[0:0].assign(split=pd.Series(dtype=str))
        )

    # Only rows actually used by a partition belong in the exported split.
    output = output.loc[output["split"] != "unassigned"].copy()
    output.to_csv(csv_path, index=False)
    logger.info("Saved human-readable split CSV to: %s", csv_path)
    return csv_path


def load_test_metadata(split_config, smiles_col, expected_rows):
    """Load aligned molecule names and SMILES from the readable split CSV."""
    csv_path = get_split_csv_path(split_config)
    if csv_path is None or not csv_path.exists():
        return None

    frame = pd.read_csv(csv_path)
    if "split" not in frame.columns or smiles_col not in frame.columns:
        return None
    test_frame = frame.loc[
        frame["split"].astype(str).str.strip().str.lower().isin({"test", "testing"})
    ].reset_index(drop=True)
    if len(test_frame) != int(expected_rows):
        logger.warning(
            "Cannot attach test metadata: split CSV has %d test rows but predictions have %d.",
            len(test_frame),
            expected_rows,
        )
        return None

    normalized_columns = {
        str(column).strip().lower().replace("_", " "): column
        for column in test_frame.columns
    }
    name_column = next(
        (
            normalized_columns[candidate]
            for candidate in (
                "molecule name",
                "compound name",
                "molecule id",
                "compound id",
                "name",
            )
            if candidate in normalized_columns
        ),
        None,
    )
    names = (
        test_frame[name_column].astype(str).to_numpy()
        if name_column is not None
        else np.asarray([f"Test_{index + 1}" for index in range(len(test_frame))])
    )
    return {
        "Molecule Name": names,
        "SMILES": test_frame[smiles_col].astype(str).to_numpy(),
    }


def ensure_split_csv_from_cache(
    training_data_file,
    smiles_col,
    y_col,
    split_config,
    cached_data,
    job_dir,
):
    """Create a missing CSV companion from cached indices or source labels."""
    csv_path = get_split_csv_path(split_config)
    if csv_path is None or training_data_file is None:
        return csv_path

    payload = load_training_data(training_data_file)
    frame = payload["data"]
    if not isinstance(frame, pd.DataFrame):
        return None

    source_normalized = {
        str(column).strip().lower().replace("_", " "): column
        for column in frame.columns
    }
    source_name_column = next(
        (
            source_normalized[candidate]
            for candidate in (
                "molecule name",
                "compound name",
                "molecule id",
                "compound id",
                "name",
            )
            if candidate in source_normalized
        ),
        None,
    )
    if csv_path.exists():
        existing_columns = set(pd.read_csv(csv_path, nrows=0).columns)
        required_columns = {smiles_col, y_col, "split"}
        if source_name_column is not None:
            required_columns.add(source_name_column)
        if required_columns.issubset(existing_columns):
            return csv_path

    clean_frame = clean_raw_dataframe(
        df=frame,
        smiles_col=smiles_col,
        target_col=y_col,
        job_dir=job_dir,
        split_label="split_csv_export",
    )
    method = str(split_config.get("split_method", "")).strip().lower()
    if method in {"predefined", "splitted"}:
        split_column = split_config.get("split_column", "split")
        if split_column not in clean_frame.columns:
            return None
        values = clean_frame[split_column].astype(str).str.strip().str.lower()
        labels = np.full(len(clean_frame), "unassigned", dtype="<U10")
        labels[values.isin({"train", "training"}).to_numpy()] = "train"
        labels[values.isin({"val", "valid", "validation", "dev"}).to_numpy()] = "validation"
        labels[values.isin({"test", "testing"}).to_numpy()] = "test"
        return save_split_dataframe(
            clean_frame,
            split_config,
            smiles_col=smiles_col,
            target_col=y_col,
            split_labels=labels,
        )

    keys = set(cached_data.files)
    if not {"train_indices", "test_indices"}.issubset(keys):
        return None
    return save_split_dataframe(
        clean_frame,
        split_config,
        smiles_col=smiles_col,
        target_col=y_col,
        train_indices=np.asarray(cached_data["train_indices"], dtype=int),
        valid_indices=(
            np.asarray(cached_data["valid_indices"], dtype=int)
            if "valid_indices" in keys
            else None
        ),
        test_indices=np.asarray(cached_data["test_indices"], dtype=int),
    )


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
    train_smiles=None,
    feature_types=None,
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

    if train_smiles is not None:
        arrays_to_save["train_smiles"] = np.asarray(train_smiles, dtype=str)

    if feature_types is not None:
        arrays_to_save["feature_types"] = np.asarray(feature_types, dtype=str)

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
    
    train_smiles = split_data.get("train_smiles")
    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_valid,
        y_valid,
        "pre_split_npz",
        train_smiles,
    )


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

    used_mask = train_mask | valid_mask | test_mask
    canonical_labels = np.where(
        train_mask.loc[used_mask],
        "train",
        np.where(valid_mask.loc[used_mask], "validation", "test"),
    )
    save_split_dataframe(
        frame=clean_df.loc[used_mask],
        split_config=split_config or {},
        smiles_col=smiles_col,
        target_col=y_col,
        split_labels=canonical_labels,
    )

    X_train, y_train, train_valid_indices = featurize_array(
        train_df[smiles_col].to_numpy(),
        train_df[y_col].to_numpy(),
        feature_types,
    )
    train_smiles = train_df[smiles_col].to_numpy()[train_valid_indices]

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
        train_smiles=train_smiles,
        feature_types=feature_types,
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_valid,
        y_valid,
        "pre_split_column",
        train_smiles,
    )


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

    X_train, y_train, train_valid_indices = featurize_array(
        train_df[smiles_col].to_numpy(),
        train_df[y_col].to_numpy(),
        feature_types,
    )
    train_smiles = train_df[smiles_col].to_numpy()[train_valid_indices]

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

    split_frames = [train_df.assign(split="train")]
    if valid_df is not None and len(valid_df) > 0:
        split_frames.append(valid_df.assign(split="validation"))
    split_frames.append(test_df.assign(split="test"))
    split_export = pd.concat(split_frames, ignore_index=True)
    save_split_dataframe(
        frame=split_export.drop(columns="split"),
        split_config=split_config or {},
        smiles_col=smiles_col,
        target_col=y_col,
        split_labels=split_export["split"].to_numpy(),
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
        train_smiles=train_smiles,
        feature_types=feature_types,
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_valid,
        y_valid,
        "explicit_split_files",
        train_smiles,
    )


def rebuild_cached_features_from_indices(
    training_data_file,
    smiles_col,
    y_col,
    feature_types,
    split_npz_path,
    cached_data,
    job_dir,
):
    """Rebuild features while preserving indices from an existing split cache."""
    required_indices = {"train_indices", "test_indices"}
    available_keys = set(cached_data.files)
    missing = required_indices.difference(available_keys)
    if missing:
        raise ValueError(
            "The cached split cannot be reused safely because it contains "
            f"legacy feature arrays and is missing {sorted(missing)}."
        )

    cached_indices = {
        key: np.asarray(cached_data[key], dtype=int)
        for key in ("train_indices", "test_indices", "valid_indices")
        if key in available_keys
    }
    # The upgraded cache replaces the legacy file. Close its zip handle first
    # so this also works on platforms that prohibit replacing an open file.
    cached_data.close()

    payload = load_training_data(training_data_file)
    frame = payload["data"]
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(
            "A legacy split cache can only be rebuilt from its original "
            "tabular source dataset."
        )
    clean_frame = clean_raw_dataframe(
        df=frame,
        smiles_col=smiles_col,
        target_col=y_col,
        job_dir=job_dir,
        split_label="cached_split_rebuild",
    )
    all_smiles = clean_frame[smiles_col].to_numpy()
    all_targets = clean_frame[y_col].to_numpy()

    save_split_dataframe(
        frame=clean_frame,
        split_config={
            "save_split_data": True,
            "save_dir": str(split_npz_path.parent),
            "split_name": split_npz_path.stem.removesuffix("_split_data"),
        },
        smiles_col=smiles_col,
        target_col=y_col,
        train_indices=cached_indices["train_indices"],
        valid_indices=cached_indices.get("valid_indices"),
        test_indices=cached_indices["test_indices"],
    )

    def rebuild_partition(indices_key):
        indices = cached_indices[indices_key]
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(clean_frame)):
            raise ValueError(
                f"Cached {indices_key} are incompatible with the current source dataset."
            )
        features, targets, valid_local_indices = featurize_array(
            all_smiles[indices],
            all_targets[indices],
            feature_types,
        )
        if features is None:
            raise ValueError(f"No valid molecules remain in cached {indices_key}.")
        aligned_smiles = np.asarray(all_smiles[indices], dtype=str)[valid_local_indices]
        return features, targets, indices, aligned_smiles

    X_train, y_train, train_indices, train_smiles = rebuild_partition(
        "train_indices"
    )
    X_test, y_test, test_indices, _ = rebuild_partition("test_indices")

    X_valid = None
    y_valid = None
    valid_indices = None
    if "valid_indices" in cached_indices:
        valid_indices = cached_indices["valid_indices"]
        if valid_indices.size:
            X_valid, y_valid, valid_indices, _ = rebuild_partition("valid_indices")

    arrays_to_save = {
        "X_train": np.asarray(X_train, dtype=np.float32),
        "X_test": np.asarray(X_test, dtype=np.float32),
        "y_train": np.asarray(y_train),
        "y_test": np.asarray(y_test),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_smiles": np.asarray(train_smiles, dtype=str),
        "feature_types": np.asarray(feature_types, dtype=str),
    }
    if X_valid is not None:
        arrays_to_save["X_valid"] = np.asarray(X_valid, dtype=np.float32)
        arrays_to_save["y_valid"] = np.asarray(y_valid)
        arrays_to_save["valid_indices"] = valid_indices

    np.savez_compressed(split_npz_path, **arrays_to_save)
    logger.info(
        "Upgraded cached split without repartitioning: %s",
        split_npz_path,
    )
    write_log(
        job_dir,
        "Reused cached split indices and rebuilt its feature arrays; no data "
        "were repartitioned.",
    )
    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_valid,
        y_valid,
        "split",
        train_smiles,
    )


def load_or_create_split_features(
    training_data_file,
    smiles_col,
    y_col,
    feature_types,
    split_config,
    split_npz_path,
    job_dir,
    require_train_smiles=False,
):
    if split_npz_path is not None:
        logger.info("Checking split feature data: %s", split_npz_path)

    if split_npz_path is not None and split_npz_path.exists():
        logger.info("Existing featurized split data found. Loading NPZ...")
        with np.load(split_npz_path, allow_pickle=False) as data:
            requested_features = [str(name).strip().lower() for name in feature_types]
            cached_features = (
                [str(name).strip().lower() for name in data["feature_types"].tolist()]
                if "feature_types" in data.files
                else None
            )
            if cached_features != requested_features:
                logger.warning(
                    "Cached features %s do not match requested features %s. "
                    "Reusing split indices and rebuilding feature arrays.",
                    cached_features,
                    requested_features,
                )
                if "train_indices" not in data.files or "test_indices" not in data.files:
                    raise ValueError(
                        "The feature cache does not match the requested representations "
                        "and has no split indices. Use a new split_name or remove the "
                        "old NPZ cache."
                    )
                return rebuild_cached_features_from_indices(
                    training_data_file=training_data_file,
                    smiles_col=smiles_col,
                    y_col=y_col,
                    feature_types=feature_types,
                    split_npz_path=split_npz_path,
                    cached_data=data,
                    job_dir=job_dir,
                )
            try:
                X_train = data["X_train"]
                X_test = data["X_test"]
                y_train = data["y_train"]
                y_test = data["y_test"]
                X_valid = data["X_valid"] if "X_valid" in data.files else None
                y_valid = data["y_valid"] if "y_valid" in data.files else None
                train_smiles = (
                    data["train_smiles"] if "train_smiles" in data.files else None
                )
            except ValueError as exc:
                if "Object arrays cannot be loaded" not in str(exc):
                    raise
                logger.warning(
                    "Legacy object-based feature cache detected. Reusing its "
                    "stored split indices and rebuilding numeric features."
                )
                return rebuild_cached_features_from_indices(
                    training_data_file=training_data_file,
                    smiles_col=smiles_col,
                    y_col=y_col,
                    feature_types=feature_types,
                    split_npz_path=split_npz_path,
                    cached_data=data,
                    job_dir=job_dir,
                )

            logger.info("Loaded X_train shape: %s", X_train.shape)
            logger.info("Loaded X_test shape: %s", X_test.shape)

            if X_valid is not None:
                logger.info("Loaded X_valid shape: %s", X_valid.shape)

            ensure_split_csv_from_cache(
                training_data_file=training_data_file,
                smiles_col=smiles_col,
                y_col=y_col,
                split_config=split_config,
                cached_data=data,
                job_dir=job_dir,
            )

            if not require_train_smiles or train_smiles is not None:
                return (
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    X_valid,
                    y_valid,
                    "split",
                    train_smiles,
                )
            if "train_indices" in data.files and "test_indices" in data.files:
                logger.warning(
                    "Cached split has no train_smiles required for "
                    "scaffold-grouped CV. Reusing its stored indices and "
                    "rebuilding the cache."
                )
                return rebuild_cached_features_from_indices(
                    training_data_file=training_data_file,
                    smiles_col=smiles_col,
                    y_col=y_col,
                    feature_types=feature_types,
                    split_npz_path=split_npz_path,
                    cached_data=data,
                    job_dir=job_dir,
                )
            logger.warning(
                "Cached split has no train_smiles or reusable indices; "
                "regenerating it from the source dataset."
            )

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

    if split_method in {"splitted", "predefined"}:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "split_method='predefined' requires a tabular dataset containing a split column."
            )

        split_column = split_config.get("split_column", split_config.get("split_col", "split"))

        if not split_column:
            raise ValueError(
                "split_method='predefined' requires 'split_column' in data_split config."
            )

        if split_column not in df.columns:
            raise ValueError(
                f"Predefined split requested split_column '{split_column}', "
                f"but that column was not found in the dataset."
            )

        logger.info(
            "Using predefined column '%s' to define train/test/validation sets.",
            split_column,
        )
        write_log(
            job_dir,
            f"Using existing split column '{split_column}' without repartitioning.",
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

    save_split_dataframe(
        frame=clean_df,
        split_config=split_config,
        smiles_col=smiles_col,
        target_col=y_col,
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
    )

    write_status(job_dir, "running", 40)
    logger.info("Featurizing train/test/valid arrays...")

    X_train, y_train, train_valid_feature_indices = featurize_array(
        X_train_raw,
        y_train_raw,
        feature_types,
    )
    train_smiles = np.asarray(X_train_raw, dtype=str)[train_valid_feature_indices]

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
            "train_smiles": train_smiles,
            "feature_types": np.asarray(feature_types, dtype=str),
        }

        if X_valid is not None:
            arrays_to_save["X_valid"] = X_valid

        if y_valid is not None:
            arrays_to_save["y_valid"] = y_valid

        if valid_indices is not None:
            arrays_to_save["valid_indices"] = valid_indices

        np.savez_compressed(split_npz_path, **arrays_to_save)
        logger.info("Saved featurized split NPZ to: %s", split_npz_path)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_valid,
        y_valid,
        "split",
        train_smiles,
    )


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
        "cv_strategy": metrics.get("cv_strategy"),
        "cv_folds": metrics.get("cv_folds"),
        "cv_unique_scaffolds": metrics.get("cv_unique_scaffolds"),
        "feature_reduction": metrics.get("feature_reduction"),
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

        training_data_file = data_config.get("data_file")
        explicit_train_data_file = data_config.get("train_data_file")
        if training_data_file is None and explicit_train_data_file is None:
            raise ValueError(
                "Data config requires either 'data_file' or 'train_data_file'."
            )
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
        cv_strategy = str(
            training_config.get("cv_strategy", "cv")
        ).strip().lower()
        require_train_smiles = bool(
            training_config.get("hyperparameter_tuning", True)
        ) and cv_strategy in {
            "scaffold",
            "scaffold_grouped",
            "scaffold-grouped",
        }

        explicit_test_data_file = data_config.get("test_data_file")
        explicit_validation_data_file = data_config.get("validation_data_file", data_config.get("valid_data_file"))

        if explicit_train_data_file or explicit_test_data_file or explicit_validation_data_file:
            (
                X_train,
                y_train,
                X_test,
                y_test,
                X_valid,
                y_valid,
                split_source,
                train_smiles,
            ) = load_explicit_split_features(
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

            (
                X_train,
                y_train,
                X_test,
                y_test,
                X_valid,
                y_valid,
                split_source,
                train_smiles,
            ) = load_or_create_split_features(
                training_data_file=training_data_file,
                smiles_col=smiles_col,
                y_col=y_col,
                feature_types=feature_types,
                split_config=split_config,
                split_npz_path=split_npz_path,
                job_dir=job_dir,
                require_train_smiles=require_train_smiles,
            )

        x_train_shape = getattr(X_train, "shape", None)
        x_test_shape = getattr(X_test, "shape", None)
        x_valid_shape = getattr(X_valid, "shape", None) if X_valid is not None else None

        test_metadata = load_test_metadata(
            split_config=split_config,
            smiles_col=smiles_col,
            expected_rows=len(y_test),
        )

        feature_config_to_save = {
            **featurization_config,
            "feature_types": feature_types,
            "representations": feature_types,
            "features": feature_types,
            "fp_bits": featurization_config.get("fp_bits"),
            "desc_names": (
                list(RDKIT2D_DESC_NAMES)
                if any(name in RDKIT2D_DESC_TYPES for name in feature_types)
                else list(DESC_NAMES)
                if any(name in DESC_TYPES for name in feature_types)
                else []
            ),
            "descriptor_names_by_feature": {
                name: (
                    list(RDKIT2D_DESC_NAMES)
                    if name in RDKIT2D_DESC_TYPES
                    else list(DESC_NAMES)
                )
                for name in feature_types
                if name in (*DESC_TYPES, *RDKIT2D_DESC_TYPES)
            },
            "rdkit_version": rdBase.rdkitVersion,
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
                train_smiles=train_smiles,
                test_metadata=test_metadata,
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
                test_metadata=test_metadata,
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
