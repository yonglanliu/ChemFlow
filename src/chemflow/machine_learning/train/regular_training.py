# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from src.chemflow.machine_learning.data.data_pipeline import (
    featurize_array,
    make_scaled_pipeline,
)

from src.chemflow.machine_learning import (
    MODEL_OPTIONS,
    get_default_param_grid,
    get_model,
    get_refit_metrics,
    get_scoring,
)

import time

from pathlib import Path
import json
import pickle
import traceback
from src.chemflow.machine_learning.train.utils import _json_safe, write_model_info, _merge_parent_config, safe_name
from src.chemflow.machine_learning.eval.eval_ml import evaluate_model
import pandas as pd

# ============================================================
# Regular training
# ============================================================
def regular_training(
    X_train,
    y_train,
    X_test,
    y_test,
    model_config,
    output_dir="model_training_results",
    feature_config=None,
):
    model_name = model_config["model_name"]
    task_type = str(model_config.get("task_type", model_config.get("task"))).lower()

    if model_name not in MODEL_OPTIONS:
        raise ValueError(f"Invalid model_name '{model_name}'. Expected one of: {MODEL_OPTIONS}")

    if task_type not in ["classification", "regression"]:
        raise ValueError("Invalid task. Expected 'classification' or 'regression'.")

    seeds = model_config.get("seeds", model_config.get("eval_seeds", [42]))
    seeds = [int(seed) for seed in seeds]

    model_params = model_config.get("model_params", {})
    if not model_params:
        raise ValueError("model_params must be provided in the model_config.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = safe_name(model_name)
    all_results = []

    for seed in seeds:
        seed_output_dir = output_dir / f"{model_tag}" / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        single_seed_config = dict(model_config)
        single_seed_config["seed"] = seed
        single_seed_config.pop("seeds", None)
        single_seed_config.pop("eval_seeds", None)

        summary_path = seed_output_dir / f"{model_tag}_summary.json"
        model_path = seed_output_dir / f"{model_tag}_trained_model.pkl"
        package_path = seed_output_dir / f"{model_tag}_model_package.pkl"
        error_path = seed_output_dir / f"{model_tag}_error.json"

        start_time = time.time()

        try:
            base_model = get_model(
                model_name=model_name,
                task_type=task_type,
                seed=seed,
                tune_hyperparameter=False,
                model_params=model_params,
            )

            if feature_config is not None:
                model = make_scaled_pipeline(
                    model=base_model,
                    feature_types=feature_config["feature_types"],
                )
            else:
                model = base_model

            model.fit(X_train, y_train)

            results = {
                "model_name": model_name,
                "task_type": task_type,
                "seed": seed,
                "model_params": model_params,
                "runtime_seconds": float(time.time() - start_time),
            }

            results.update(evaluate_model(model, X_test, y_test, task_type))

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(_json_safe(results), f, indent=4)

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            model_package = {
                "model": model,
                "model_name": model_name,
                "task_type": task_type,
                "feature_config": feature_config,
                "training_config": single_seed_config,
                "metrics": results,
                "chemflow_package_type": "model_package",
                "chemflow_version": "0.1.0",
                "created_at_unix": time.time(),
            }

            with open(package_path, "wb") as f:
                pickle.dump(model_package, f)

            write_model_info(
                output_dir=seed_output_dir,
                model_name=model_name,
                task_type=task_type,
                feature_config=feature_config or {},
                training_config=single_seed_config,
                metrics=results,
            )

            all_results.append(results)

        except Exception as e:
            error_info = {
                "model_name": model_name,
                "task_type": task_type,
                "seed": seed,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(_json_safe(error_info), f, indent=4)

            raise e

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(output_dir / f"{model_tag}_all_seed_summary.csv", index=False)

    return all_results



def regular_training_multiple_models(
    X_train,
    y_train,
    X_test,
    y_test,
    model_configs,
    parent_config=None,
    output_dir="model_training_results",
    feature_config=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    cfg_iter = model_configs.values() if isinstance(model_configs, dict) else model_configs

    for model_cfg in cfg_iter:
        if parent_config is not None:
            cfg = _merge_parent_config(parent_config, model_cfg)
        else:
            cfg = dict(model_cfg)

        model_name = cfg["model_name"]

        model_output_dir = output_dir / safe_name(model_name)
        model_output_dir.mkdir(parents=True, exist_ok=True)

        model_results = regular_training(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_config=cfg,
            output_dir=model_output_dir,
            feature_config=feature_config,
        )

        all_results.extend(model_results)

    all_summary_df = pd.DataFrame(all_results)
    all_summary_df.to_csv(output_dir / "all_regular_model_summary.csv", index=False)

    return all_results