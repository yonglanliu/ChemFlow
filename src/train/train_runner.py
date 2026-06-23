# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

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

from src.models import (
    MODEL_OPTIONS,
    get_default_param_grid,
    get_model,
    get_refit_metrics,
    get_scoring,
)
from src.data.data_pipeline import featurize_dataframe, make_scaled_pipeline
from src.data import DESC_NAMES, split_data


# ============================================================
# Utilities
# ============================================================

def safe_name(name: str) -> str:
    return (
        str(name)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def write_status(job_dir: str | Path, status: str, progress: int = 0, extra: Optional[dict] = None) -> None:
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


def _merge_parent_config(parent_config: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Each entry under config['models'] usually only contains model-specific fields.
    This copies global fields such as task, scoring_metrics, search_seed, cv, etc.
    The model-specific config overrides the parent config.
    """
    merged = dict(parent_config)
    merged.pop("models", None)
    merged.update(model_config)
    return merged


def _prefix_param_grid_for_pipeline(param_grid: Dict[str, Any], estimator) -> Dict[str, Any]:
    """
    RandomizedSearchCV must receive model__param when the estimator is a Pipeline.
    Do not double-prefix keys that already start with model__.
    """
    if not isinstance(estimator, Pipeline):
        return param_grid

    return {
        key if str(key).startswith("model__") else f"model__{key}": value
        for key, value in param_grid.items()
    }


# ============================================================
# Load saved training data
# ============================================================

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

        raise ValueError("Pickle file must contain a DataFrame or a dict with key 'data'.")

    if ext == ".parquet":
        return {"data": pd.read_parquet(training_data_file)}

    if ext == ".csv":
        return {"data": pd.read_csv(training_data_file)}

    raise ValueError(f"Unsupported training data format: {ext}")


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
        raise ValueError(
            f"Invalid model_name '{model_name}'. Expected one of: {MODEL_OPTIONS}"
        )

    task_type = config.get("task")

    if task_type not in ["classification", "regression"]:
        raise ValueError("Invalid task. Expected 'classification' or 'regression'.")

    search_seed = int(config.get("search_seed", 42))
    n_iter = int(config.get("n_iter", 50))
    cv = int(config.get("cv", 5))
    n_jobs = int(config.get("n_jobs", -1))

    scoring = get_scoring(config, n_classes=n_classes)
    refit_metric = get_refit_metrics(config)

    if refit_metric not in scoring:
        raise ValueError(
            f"refit_metric '{refit_metric}' must be one of: {list(scoring.keys())}"
        )

    base_model = get_model(
        model_name=model_name,
        task_type=task_type,
        search_seed=search_seed,
    )

    if feature_config is not None:
        model = make_scaled_pipeline(
            model=base_model,
            feature_types=feature_config["feature_types"],
            n_bits=feature_config.get("n_bits", 2048),
            desc_names=feature_config.get("desc_names", []),
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
            json.dump(results, f, indent=4)

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
        }

        with open(package_path, "wb") as f:
            pickle.dump(model_package, f)

        return best_model, results, cv_results

    except Exception as e:
        error_info = {
            "model_name": model_name,
            "task_type": task_type,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_info, f, indent=4)

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

    if isinstance(cfgs, dict):
        cfg_iter = cfgs.values()
    else:
        cfg_iter = cfgs

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

def train(config: Dict[str, Any], job_dir: str | Path) -> None:
    write_status(job_dir, "running", 0)
    write_log(job_dir, "Training started.")
    write_log(job_dir, f"Config: {json.dumps(config, indent=2)}")

    try:
        job_dir = Path(job_dir)

        mol_cfg = config["molecular_representation"]
        n_classes = mol_cfg.get("n_classes")
        training_data_file = mol_cfg["training_data_file"]

        if n_classes is not None:
            n_classes = int(n_classes)
            write_log(job_dir, f"Number of classes: {n_classes}")

        write_log(job_dir, f"Loading training data: {training_data_file}")
        payload = load_training_data(training_data_file)
        df = payload["data"]

        smiles_col = mol_cfg.get("smiles_col", payload.get("structure_col"))
        target_col = mol_cfg.get("target_col", payload.get("target_col"))

        if smiles_col is None:
            raise ValueError("Missing smiles_col in molecular_representation config.")

        if target_col is None:
            raise ValueError("Missing target_col in molecular_representation config.")

        feature_types = mol_cfg.get("representations", ["ECFP4"])
        n_bits = int(mol_cfg.get("fp_bits", 2048))

        write_status(job_dir, "running", 15)
        write_log(job_dir, "Featurizing molecules...")

        X, clean_df = featurize_dataframe(
            df=df,
            smiles_col=smiles_col,
            feature_types=feature_types,
            n_bits=n_bits,
            desc_names=DESC_NAMES,
        )

        if X is None or clean_df is None or len(clean_df) == 0:
            raise ValueError("Feature generation failed. No valid molecules found.")

        y = clean_df[target_col].to_numpy()

        valid_y = pd.notna(y)
        X = X[valid_y]
        clean_df = clean_df.loc[valid_y].reset_index(drop=True)
        y = clean_df[target_col].to_numpy()

        if len(y) == 0:
            raise ValueError(f"No valid target values found in target_col='{target_col}'.")

        write_log(job_dir, f"Feature matrix shape: {X.shape}")
        write_log(job_dir, f"Target length: {len(y)}")

        split_df = clean_df.copy()
        split_df["_row_id"] = np.arange(len(split_df))

        split_config = {
            "task": config["task"],
            "split_method": config.get("split_method", "random"),
            "test_size": config.get("test_size", 0.2),
            "random_seed": config.get("data_split_seed", config.get("search_seed", 42)),
            "n_clusters": config.get("n_clusters", 20),
            "butina_cutoff": config.get("butina_cutoff", 0.4),
            "fp_radius": config.get("fp_radius", 2),
            "fp_n_bits": config.get("fp_n_bits", 2048),
        }

        write_status(job_dir, "running", 35)
        write_log(job_dir, f"Splitting data using {split_config['split_method']}...")

        train_df, test_df = split_data(
            df=split_df,
            config=split_config,
            target_col=target_col,
            smiles_col=smiles_col,
        )

        train_idx = train_df["_row_id"].to_numpy()
        test_idx = test_df["_row_id"].to_numpy()

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        feature_config = {
            "feature_types": feature_types,
            "n_bits": n_bits,
            "desc_names": DESC_NAMES,
            "smiles_col": smiles_col,
            "target_col": target_col,
        }

        write_status(job_dir, "running", 55)
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

        write_status(job_dir, "completed", 100, {"metrics": results})
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
    job_dir = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    train(config, job_dir)


if __name__ == "__main__":
    main()
