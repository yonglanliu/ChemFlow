# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

"""
ChemFlow model training runner.

Pipeline:
1. Load raw dataframe.
2. Clean SMILES and target columns.
3. Split raw SMILES array.
4. Featurize X_train / X_test / X_valid using featurize_array().
5. Save feature metadata with model package.
6. Train models.
"""

import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from src.chemflow.machine_learning import (
    MODEL_OPTIONS,
    get_default_param_grid,
    get_model,
    get_refit_metrics,
    get_scoring,
)
from src.chemflow.machine_learning.data.data_pipeline import (
    featurize_array,
    make_scaled_pipeline,
)
from src.chemflow.featurization import DESC_NAMES
from src.chemflow.machine_learning.data import DataSplitter


# ============================================================
# Utilities
# ============================================================

def safe_name(name: str) -> str:
    return str(name).lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def write_status(
    job_dir: str | Path,
    status: str,
    progress: int = 0,
    extra: Optional[dict] = None,
) -> None:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "status": status,
        "progress": progress,
    }

    if extra:
        data.update(extra)

    with open(job_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def write_log(job_dir: str | Path, message: str) -> None:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    with open(job_dir / "training.log", "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")
        f.flush()


def _to_json_safe_list(x):
    return pd.Series(x).tolist()


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


def _merge_parent_config(
    parent_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(parent_config)
    merged.pop("models", None)
    merged.update(model_config)
    return merged


def _prefix_param_grid_for_pipeline(
    param_grid: Dict[str, Any],
    estimator,
) -> Dict[str, Any]:
    if not isinstance(estimator, Pipeline):
        return param_grid

    return {
        key if str(key).startswith("model__") else f"model__{key}": value
        for key, value in param_grid.items()
    }


def _clean_raw_dataframe(
    df: pd.DataFrame,
    smiles_col: str,
    target_col: str,
) -> pd.DataFrame:
    if smiles_col not in df.columns:
        raise ValueError(f"smiles_col '{smiles_col}' not found in dataframe.")

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe.")

    clean_df = df.copy()
    clean_df = clean_df.loc[pd.notna(clean_df[smiles_col])].copy()
    clean_df = clean_df.loc[pd.notna(clean_df[target_col])].copy()

    clean_df[smiles_col] = clean_df[smiles_col].astype(str)
    clean_df = clean_df.reset_index(drop=True)

    if len(clean_df) == 0:
        raise ValueError("No usable rows after removing missing SMILES/target values.")

    return clean_df


def _build_split_config(data_split_cfg: Dict[str, Any]) -> Dict[str, Any]:
    split_method = data_split_cfg["split_method"]

    split_name = data_split_cfg.get(
        "split_name",
        data_split_cfg.get("prefix_name", f"{safe_name(split_method)}_split"),
    )

    return {
        "task": data_split_cfg.get("task", data_split_cfg.get("task_type")),
        "task_type": data_split_cfg.get("task_type", data_split_cfg.get("task")),
        "split_method": split_method,
        "test_size": data_split_cfg["test_size"],
        "validation_size": data_split_cfg.get("validation_size"),
        "random_seed": data_split_cfg.get("random_seed", 42),
        "n_clusters": data_split_cfg.get("n_clusters"),
        "butina_cutoff": data_split_cfg.get("butina_cutoff"),
        "fp_radius": data_split_cfg.get("fp_radius"),
        "fp_n_bits": data_split_cfg.get("fp_n_bits"),
        "save_split_data": False,
        "save_dir": data_split_cfg["save_dir"],
        "split_name": split_name,
        "prefix_name": split_name,
    }


def load_training_data(training_data_file: str | Path) -> Dict[str, Any]:
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


def build_feature_config(
    mol_cfg: Dict[str, Any],
    split_config: Dict[str, Any],
    smiles_col: str,
    y_col: str,
    feature_types,
    n_bits: Optional[int],
    X_train=None,
    X_test=None,
    X_valid=None,
) -> Dict[str, Any]:
    return {
        "feature_types": feature_types,
        "representations": feature_types,
        "n_bits": n_bits,
        "fp_bits": n_bits,
        "desc_names": DESC_NAMES,
        "smiles_col": smiles_col,
        "target_col": y_col,
        "y_col": y_col,
        "split_method": split_config.get("split_method"),
        "split_name": split_config.get("split_name"),
        "feature_array_shapes": {
            "X_train": None if X_train is None else list(X_train.shape),
            "X_test": None if X_test is None else list(X_test.shape),
            "X_valid": None if X_valid is None else list(X_valid.shape),
        },
        "featurizer": "featurize_array",
        "mol_config": _json_safe(mol_cfg),
    }


def write_model_info(
    output_dir: str | Path,
    model_name: str,
    task_type: str,
    feature_config: Dict[str, Any],
    training_config: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    output_dir = Path(output_dir)

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
        "refit_metric": metrics.get("refit_metric"),
        "metrics_summary": {
            k: v
            for k, v in metrics.items()
            if k.startswith("test_") or k.startswith("best_cv_")
        },
        "training_config": _json_safe(training_config),
    }

    model_tag = safe_name(model_name)

    with open(output_dir / f"{model_tag}_model_info.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(model_info), f, indent=4)


# ============================================================
# Model loading helper
# ============================================================

def load_pickle_model(model_path: str | Path) -> Dict[str, Any]:
    """
    Load either:
    1. A ChemFlow model package dict, or
    2. A raw sklearn/XGBoost pickle model.

    Returns a normalized package dict.
    """
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


def get_loaded_model_info(model_package: Dict[str, Any]) -> Dict[str, Any]:
    feature_config = model_package.get("feature_config") or {}
    training_config = model_package.get("training_config") or {}
    metrics = model_package.get("metrics") or {}

    return {
        "model_name": model_package.get("model_name") or training_config.get("model_name"),
        "task_type": model_package.get("task_type") or training_config.get("task_type"),
        "feature_types": feature_config.get("feature_types"),
        "n_bits": feature_config.get("n_bits"),
        "desc_names": feature_config.get("desc_names"),
        "smiles_col": feature_config.get("smiles_col"),
        "target_col": feature_config.get("target_col"),
        "split_method": feature_config.get("split_method"),
        "split_name": feature_config.get("split_name"),
        "feature_array_shapes": feature_config.get("feature_array_shapes"),
        "best_params": metrics.get("best_params"),
        "refit_metric": metrics.get("refit_metric"),
    }


# ============================================================
# Model tuning
# ============================================================

def tune_hyperparameters(
    X_train,
    y_train,
    X_test,
    y_test,
    config: Dict[str, Any],
    n_classes: Optional[int] = None,
    output_dir: str | Path = "model_tuning_results",
    feature_config: Dict[str, Any] | None = None,
):
    model_name = config["model_name"]

    if model_name not in MODEL_OPTIONS:
        raise ValueError(f"Invalid model_name '{model_name}'. Expected one of: {MODEL_OPTIONS}")

    task_type = config.get("task", config.get("task_type"))

    if task_type not in ["classification", "regression"]:
        raise ValueError("Invalid task. Expected 'classification' or 'regression'.")

    search_seed = int(config.get("search_seed", 42))
    n_iter = int(config.get("n_iter", 50))
    cv = int(config.get("cv", 5))
    n_jobs = int(config.get("n_jobs", -1))

    scoring = get_scoring(config, n_classes=n_classes)
    refit_metric = get_refit_metrics(config)

    if refit_metric not in scoring:
        raise ValueError(f"refit_metric '{refit_metric}' must be one of {list(scoring.keys())}")

    base_model = get_model(
        model_name=model_name,
        task_type=task_type,
        search_seed=search_seed,
    )

    if feature_config is not None:
        model = make_scaled_pipeline(
            model=base_model,
            feature_types=feature_config["feature_types"],
        )
    else:
        model = base_model

    param_grid = config.get("param_grid")

    if param_grid is None:
        param_grid = get_default_param_grid(model_name, task_type)

    param_grid = _prefix_param_grid_for_pipeline(param_grid, model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = safe_name(model_name)
    summary_path = output_dir / f"{model_tag}_summary.json"
    cv_path = output_dir / f"{model_tag}_cv_results.csv"
    model_path = output_dir / f"{model_tag}_best_model.pkl"
    package_path = output_dir / f"{model_tag}_model_package.pkl"
    error_path = output_dir / f"{model_tag}_error.json"

    start_time = time.time()

    try:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            random_state=search_seed,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        cv_results = pd.DataFrame(search.cv_results_)
        best_idx = search.best_index_

        results = {
            "model_name": model_name,
            "task_type": task_type,
            "best_params": search.best_params_,
            "refit_metric": refit_metric,
            "best_cv_score_raw": float(search.best_score_),
            "runtime_seconds": float(time.time() - start_time),
        }

        for metric in config.get("scoring_metrics", []):
            col = f"mean_test_{metric}"

            if col in cv_results.columns:
                value = float(cv_results.loc[best_idx, col])

                if metric in ["root_mean_squared_error", "mean_absolute_error", "rmse", "mae"]:
                    value = abs(value)

                results[f"best_cv_{metric}"] = value

        if task_type == "regression":
            results.update(
                {
                    "test_rmse": float(root_mean_squared_error(y_test, y_pred)),
                    "test_mae": float(mean_absolute_error(y_test, y_pred)),
                    "test_r2": float(r2_score(y_test, y_pred)),
                    "y_test": _to_json_safe_list(y_test),
                    "y_pred": _to_json_safe_list(y_pred),
                }
            )

        else:
            labels = sorted(pd.Series(y_test).dropna().unique())

            results.update(
                {
                    "test_accuracy": float(accuracy_score(y_test, y_pred)),
                    "test_f1_macro": float(
                        f1_score(y_test, y_pred, average="macro", zero_division=0)
                    ),
                    "classification_report": classification_report(
                        y_test,
                        y_pred,
                        output_dict=True,
                        zero_division=0,
                    ),
                    "confusion_matrix": confusion_matrix(
                        y_test,
                        y_pred,
                        labels=labels,
                    ).tolist(),
                    "confusion_matrix_labels": [str(x) for x in labels],
                    "y_test": _to_json_safe_list(y_test),
                    "y_pred": _to_json_safe_list(y_pred),
                }
            )

            if hasattr(best_model, "predict_proba"):
                try:
                    y_prob = best_model.predict_proba(X_test)
                    results["y_proba"] = y_prob.tolist()

                    if y_prob.shape[1] == 2:
                        y_score = y_prob[:, 1]

                        fpr, tpr, roc_thresholds = roc_curve(y_test, y_score)
                        precision, recall, pr_thresholds = precision_recall_curve(
                            y_test,
                            y_score,
                        )

                        results["test_roc_auc"] = float(roc_auc_score(y_test, y_score))
                        results["test_average_precision"] = float(
                            average_precision_score(y_test, y_score)
                        )

                        results["roc_curve"] = {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist(),
                            "thresholds": roc_thresholds.tolist(),
                        }

                        results["pr_curve"] = {
                            "precision": precision.tolist(),
                            "recall": recall.tolist(),
                            "thresholds": pr_thresholds.tolist(),
                        }

                    else:
                        results["test_roc_auc"] = float(
                            roc_auc_score(
                                y_test,
                                y_prob,
                                multi_class="ovr",
                                average="macro",
                            )
                        )

                        y_test_bin = label_binarize(y_test, classes=labels)
                        roc_curves = {}
                        pr_curves = {}

                        for i, label in enumerate(labels):
                            fpr, tpr, roc_thresholds = roc_curve(
                                y_test_bin[:, i],
                                y_prob[:, i],
                            )
                            precision, recall, pr_thresholds = precision_recall_curve(
                                y_test_bin[:, i],
                                y_prob[:, i],
                            )
                            ap = average_precision_score(y_test_bin[:, i], y_prob[:, i])

                            roc_curves[str(label)] = {
                                "fpr": fpr.tolist(),
                                "tpr": tpr.tolist(),
                                "thresholds": roc_thresholds.tolist(),
                            }

                            pr_curves[str(label)] = {
                                "precision": precision.tolist(),
                                "recall": recall.tolist(),
                                "thresholds": pr_thresholds.tolist(),
                                "average_precision": float(ap),
                            }

                        results["roc_curves_ovr"] = roc_curves
                        results["pr_curves_ovr"] = pr_curves

                except Exception as e:
                    results["test_roc_auc"] = None
                    results["auc_error"] = str(e)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(results), f, indent=4)

        cv_results.to_csv(cv_path, index=False)

        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        model_package = {
            "model": best_model,
            "model_name": model_name,
            "task_type": task_type,
            "feature_config": feature_config,
            "training_config": config,
            "metrics": results,
            "chemflow_package_type": "model_package",
            "chemflow_version": "0.1.0",
            "created_at_unix": time.time(),
        }

        with open(package_path, "wb") as f:
            pickle.dump(model_package, f)

        write_model_info(
            output_dir=output_dir,
            model_name=model_name,
            task_type=task_type,
            feature_config=feature_config or {},
            training_config=config,
            metrics=results,
        )

        return best_model, results, cv_results

    except Exception as e:
        error_info = {
            "model_name": model_name,
            "task_type": task_type,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(error_info), f, indent=4)

        raise e


def tune_parameters_multiple_model(
    X_train,
    y_train,
    X_test,
    y_test,
    cfgs,
    parent_config: Dict[str, Any],
    n_classes: Optional[int] = None,
    output_dir: str | Path = "model_tuning_results",
    feature_config: Dict[str, Any] | None = None,
):
    all_results = []
    cfg_iter = cfgs.values() if isinstance(cfgs, dict) else cfgs

    for model_cfg in cfg_iter:
        cfg = _merge_parent_config(parent_config, model_cfg)

        _, results, _ = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=cfg,
            n_classes=n_classes,
            output_dir=output_dir,
            feature_config=feature_config,
        )

        all_results.append(results)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(Path(output_dir) / "all_model_summary.csv", index=False)

    return all_results


# ============================================================
# Training runner
# ============================================================

def train(config: Dict[str, Any]) -> None:
    job_dir = Path(config.get("workdir", "training_job")).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    write_status(job_dir, "running", 0)
    write_log(job_dir, "Training started.")
    write_log(job_dir, f"Config: {json.dumps(_json_safe(config), indent=2)}")

    try:
        mol_cfg = config["featurization"]
        split_config = _build_split_config(config["data_split"])

        n_classes = mol_cfg.get("n_classes")

        if n_classes is not None:
            n_classes = int(n_classes)
            write_log(job_dir, f"Number of classes: {n_classes}")

        training_data_file = mol_cfg["training_data_file"]
        smiles_col = mol_cfg["smiles_col"]
        y_col = mol_cfg["y_col"]
        feature_types = mol_cfg["representations"]

        if smiles_col is None:
            raise ValueError("Missing smiles_col in featurization config.")

        if y_col is None:
            raise ValueError("Missing y_col in featurization config.")

        n_bits = mol_cfg.get("fp_bits")

        if n_bits is not None:
            n_bits = int(n_bits)

        split_save_dir = Path(split_config["save_dir"])
        split_save_dir.mkdir(parents=True, exist_ok=True)

        split_npz_path = split_save_dir / f"{split_config['split_name']}_split_data.npz"

        write_status(job_dir, "running", 10)
        write_log(job_dir, f"Checking split feature data: {split_npz_path}")

        if split_npz_path.exists():
            write_log(job_dir, "Existing featurized split data found. Loading NPZ...")

            data = np.load(split_npz_path, allow_pickle=False)

            X_train = data["X_train"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]

            X_valid = data["X_valid"] if "X_valid" in data.files else None
            y_valid = data["y_valid"] if "y_valid" in data.files else None

            write_log(job_dir, f"Loaded X_train shape: {X_train.shape}")
            write_log(job_dir, f"Loaded X_test shape: {X_test.shape}")

            if X_valid is not None:
                write_log(job_dir, f"Loaded X_valid shape: {X_valid.shape}")

        else:
            write_log(job_dir, "No existing split data found. Loading raw training data...")
            write_log(job_dir, f"Training data file: {training_data_file}")

            payload = load_training_data(training_data_file)
            df = payload["data"]

            write_log(job_dir, "Cleaning raw dataframe before split...")
            clean_df = _clean_raw_dataframe(
                df=df,
                smiles_col=smiles_col,
                target_col=y_col,
            )

            write_log(job_dir, f"Clean dataframe shape: {clean_df.shape}")

            smiles = clean_df[smiles_col].to_numpy()
            y_raw = clean_df[y_col].to_numpy()

            splitter = DataSplitter(split_config)

            write_status(job_dir, "running", 25)
            write_log(job_dir, f"Splitting raw SMILES using {split_config['split_method']}...")
            write_log(job_dir, f"Split config: {json.dumps(_json_safe(split_config), indent=2)}")

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

            write_log(job_dir, f"Raw train SMILES: {len(X_train_raw)}")
            write_log(job_dir, f"Raw test SMILES: {len(X_test_raw)}")
            write_log(
                job_dir,
                f"Raw valid SMILES: {0 if X_valid_raw is None else len(X_valid_raw)}",
            )

            write_status(job_dir, "running", 40)
            write_log(job_dir, "Featurizing train/test/valid arrays with featurize_array()...")

            X_train, y_train, train_clean_idx = featurize_array(
                X_train_raw,
                y_train_raw,
                feature_types,
            )

            X_test, y_test, test_clean_idx = featurize_array(
                X_test_raw,
                y_test_raw,
                feature_types,
            )

            X_valid = None
            y_valid = None

            if X_valid_raw is not None and y_valid_raw is not None and len(X_valid_raw) > 0:
                X_valid, y_valid, valid_clean_idx = featurize_array(
                    X_valid_raw,
                    y_valid_raw,
                    feature_types,
                )

            write_log(job_dir, f"X_train shape after featurization: {X_train.shape}")
            write_log(job_dir, f"X_test shape after featurization: {X_test.shape}")

            if X_valid is not None:
                write_log(job_dir, f"X_valid shape after featurization: {X_valid.shape}")

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

        feature_config = build_feature_config(
            mol_cfg=mol_cfg,
            split_config=split_config,
            smiles_col=smiles_col,
            y_col=y_col,
            feature_types=feature_types,
            n_bits=n_bits,
            X_train=X_train,
            X_test=X_test,
            X_valid=X_valid,
        )

        with open(job_dir / "feature_config.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(feature_config), f, indent=4)

        write_status(job_dir, "running", 60)
        write_log(job_dir, "Starting model tuning...")

        if config.get("models"):
            results = tune_parameters_multiple_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cfgs=config["models"],
                parent_config=config,
                n_classes=n_classes,
                output_dir=job_dir,
                feature_config=feature_config,
            )
        else:
            _, results, _ = tune_hyperparameters(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                config=config,
                n_classes=n_classes,
                output_dir=job_dir,
                feature_config=feature_config,
            )

        write_status(job_dir, "completed", 100, {"metrics": _json_safe(results)})
        write_log(job_dir, "Training completed.")

    except Exception as e:
        write_log(job_dir, "Training failed.")
        write_log(job_dir, traceback.format_exc())
        write_status(job_dir, "failed", 0, {"error": str(e)})
        raise e


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Missing config path. Usage: python train_runner.py path/to/config.json")

    config_path = Path(sys.argv[1]).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    train(config)


if __name__ == "__main__":
    main()