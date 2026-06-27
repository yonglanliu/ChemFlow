# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from src.chemflow.machine_learning import (
    MODEL_OPTIONS,
    get_default_param_grid,
    get_model,
    get_refit_metrics,
    get_scoring,
)
from src.chemflow.featurization import DESC_NAMES
from src.chemflow.machine_learning.data.data_pipeline import (
    featurize_array,
    make_scaled_pipeline,
)
import time
from pathlib import Path
import json
import pickle
import traceback
from src.chemflow.machine_learning.train.utils import _json_safe, write_model_info, _merge_parent_config, safe_name
from src.chemflow.machine_learning.eval.eval_ml import evaluate_model   
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd

# ============================================================
# Hyperparameter tuning
# ============================================================

def _prefix_param_grid_for_pipeline(param_grid, estimator):
    if not isinstance(estimator, Pipeline):
        return param_grid

    return {
        key if str(key).startswith("model__") else f"model__{key}": value
        for key, value in param_grid.items()
    }

def tune_hyperparameters(
    X_train,
    y_train,
    X_test,
    y_test,
    config,
    n_classes=None,
    output_dir="model_tuning_results",
    feature_config=None,
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
        seed=search_seed,
        tune_hyperparameter=True,
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

        results.update(evaluate_model(best_model, X_test, y_test, task_type))

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
    parent_config,
    n_classes=None,
    output_dir="model_tuning_results",
    feature_config=None,
):
    all_results = []
    cfg_iter = cfgs.values() if isinstance(cfgs, dict) else cfgs

    for model_cfg in cfg_iter:
        cfg = _merge_parent_config(parent_config, model_cfg)

        model_output_dir = Path(output_dir) / safe_name(cfg["model_name"])
        model_output_dir.mkdir(parents=True, exist_ok=True)

        _, results, _ = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=cfg,
            n_classes=n_classes,
            output_dir=model_output_dir,
            feature_config=feature_config,
        )

        all_results.append(results)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(Path(output_dir) / "all_model_summary.csv", index=False)

    return all_results