# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import pickle
from pathlib import Path
import pandas as pd
from typing import Any, Dict
import json
import numpy as np


def load_training_data(training_data_file):
    training_data_file = Path(training_data_file)
    ext = training_data_file.suffix.lower()

    if ext in [".pkl", ".pickle"]:
        with open(training_data_file, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and "data" in payload:
            return payload

        if isinstance(payload, pd.DataFrame):
            return {"data": payload}

        raise ValueError("Pickle file must contain a DataFrame or dict with key 'data'.")

    if ext == ".parquet":
        return {"data": pd.read_parquet(training_data_file)}

    if ext == ".csv":
        return {"data": pd.read_csv(training_data_file)}

    raise ValueError(f"Unsupported training data format: {ext}")



def build_split_config(data_split_cfg: Dict[str, Any]) -> Dict[str, Any]:
    result_conf: Dict[str, Any] = {}

    split_method = str(data_split_cfg.get("split_method")).lower()

    if split_method not in ["random", "scaffold", "butina", "cluster"]:
        raise ValueError(f"Invalid split_method: {split_method}")

    result_conf["split_method"] = split_method

    if "test_size" not in data_split_cfg:
        raise ValueError("Missing 'test_size' in data_split config.")

    result_conf["test_size"] = float(data_split_cfg["test_size"])
    result_conf["validation_size"] = data_split_cfg.get("validation_size")

    if result_conf["validation_size"] is not None:
        result_conf["validation_size"] = float(result_conf["validation_size"])

    split_seed = int(data_split_cfg.get("random_seed", 42))
    result_conf["random_seed"] = split_seed

    if split_method in ["butina", "cluster"]:
        if "fp_radius" not in data_split_cfg:
            raise ValueError("Missing 'fp_radius' for butina/cluster split.")

        if "fp_n_bits" not in data_split_cfg:
            raise ValueError("Missing 'fp_n_bits' for butina/cluster split.")

        result_conf["fp_radius"] = int(data_split_cfg["fp_radius"])
        result_conf["fp_n_bits"] = int(data_split_cfg["fp_n_bits"])

        if split_method == "butina":
            if "butina_cutoff" not in data_split_cfg:
                raise ValueError("Missing 'butina_cutoff' for butina split.")

            result_conf["butina_cutoff"] = float(data_split_cfg["butina_cutoff"])

        elif split_method == "cluster":
            if "n_clusters" not in data_split_cfg:
                raise ValueError("Missing 'n_clusters' for cluster split.")

            result_conf["n_clusters"] = int(data_split_cfg["n_clusters"])

    save_split_data = bool(data_split_cfg.get("save_split_data", True))
    result_conf["save_split_data"] = save_split_data

    save_dir = data_split_cfg.get("save_dir", "split_data")
    result_conf["save_dir"] = str(Path(save_dir).resolve())

    split_name = data_split_cfg.get(
        "split_name",
        data_split_cfg.get("prefix_name", f"{split_method}_split_seed{split_seed}"),
    )

    result_conf["split_name"] = split_name
    result_conf["prefix_name"] = split_name

    return result_conf

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, Path):
        return str(obj)

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

def _merge_parent_config(parent_config, model_config):
    merged = dict(parent_config)
    merged.pop("models", None)
    merged.update(model_config)
    return merged


def write_model_info(
    output_dir,
    model_name,
    task_type,
    feature_config,
    training_config,
    metrics,
):
    output_dir = Path(output_dir)
    model_tag = model_name

    model_info = {
        "model_name": model_name,
        "task_type": task_type,
        "feature_types": feature_config.get("feature_types"),
        "n_bits": feature_config.get("n_bits"),
        "desc_names": feature_config.get("desc_names"),
        "smiles_col": feature_config.get("smiles_col"),
        "target_col": feature_config.get("target_col"),
        "split_method": feature_config.get("split_method"),
        "split_name": feature_config.get("split_name"),
        "feature_array_shapes": feature_config.get("feature_array_shapes"),
        "best_params": metrics.get("best_params"),
        "model_params": metrics.get("model_params"),
        "refit_metric": metrics.get("refit_metric"),
        "metrics_summary": {
            k: v
            for k, v in metrics.items()
            if k.startswith("test_") or k.startswith("best_cv_")
        },
        "training_config": _json_safe(training_config),
    }

    with open(output_dir / f"{model_tag}_model_info.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(model_info), f, indent=4)

def safe_name(name: str) -> str:
    return str(name).lower().replace(" ", "_").replace("/", "_").replace("-", "_")