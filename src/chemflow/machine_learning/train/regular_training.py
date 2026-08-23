# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import json
import pickle
import time
import traceback
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.chemflow.machine_learning import (
    MODEL_OPTIONS,
    get_model,
)
from src.chemflow.machine_learning.data.data_pipeline import (
    make_scaled_pipeline,
)
from src.chemflow.machine_learning.eval.eval_ml import evaluate_model
from src.chemflow.machine_learning.train.utils import (
    _json_safe,
    _merge_parent_config,
    safe_name,
    write_model_info,
)


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
    progress_callback=None,
):
    """
    Train a single model across one or more random seeds.

    If model_params is not provided, is None, or is empty,
    the estimator defaults defined in get_model() are used.
    """

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    model_name = model_config["model_name"]

    task_type = str(
        model_config.get(
            "task_type",
            model_config.get("task"),
        )
    ).lower()

    if model_name not in MODEL_OPTIONS:
        raise ValueError(
            f"Invalid model_name '{model_name}'. "
            f"Expected one of: {MODEL_OPTIONS}"
        )

    if task_type not in ["classification", "regression"]:
        raise ValueError(
            "Invalid task. Expected 'classification' or 'regression'."
        )

    # --------------------------------------------------------
    # Seeds
    # --------------------------------------------------------

    seeds = model_config.get(
        "seeds",
        model_config.get(
            "eval_seeds",
            [42],
        ),
    )

    seeds = [
        int(seed)
        for seed in seeds
    ]

    # --------------------------------------------------------
    # Model parameters
    # --------------------------------------------------------
    #
    # Missing:
    #   model_params
    #
    # or:
    #   model_params: null
    #
    # or:
    #   model_params: {}
    #
    # all become {}, allowing get_model() to use model defaults.
    # --------------------------------------------------------

    model_params = model_config.get("model_params") or {}

    # Defensive copy so this function does not mutate config.
    model_params = dict(model_params)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_tag = safe_name(model_name)

    all_results = []

    # --------------------------------------------------------
    # Logging helper
    # --------------------------------------------------------

    def log(
        message,
        log_name="training.log",
    ):
        print(message)

        from src.chemflow.machine_learning.train.train_runner import (
            write_log,
        )

        write_log(
            output_dir,
            message,
        )

    # --------------------------------------------------------
    # Seed progress bar
    # --------------------------------------------------------

    seed_iterator = tqdm(
        seeds,
        desc=f"{model_name} seeds",
        unit="seed",
        leave=False,
        dynamic_ncols=True,
    )

    # ========================================================
    # Train each seed
    # ========================================================

    for seed in seed_iterator:

        seed_iterator.set_postfix(
            seed=seed,
        )

        # ----------------------------------------------------
        # Per-seed output directory
        # ----------------------------------------------------

        seed_output_dir = (
            output_dir
            / model_tag
            / f"seed_{seed}"
        )

        seed_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_log_name = "training.log"

        # ----------------------------------------------------
        # Per-seed config
        # ----------------------------------------------------

        single_seed_config = dict(
            model_config
        )

        single_seed_config["seed"] = seed

        single_seed_config.pop(
            "seeds",
            None,
        )

        single_seed_config.pop(
            "eval_seeds",
            None,
        )

        # Normalize model_params in saved config as well.
        single_seed_config["model_params"] = dict(
            model_params
        )

        # ----------------------------------------------------
        # Output files
        # ----------------------------------------------------

        summary_path = (
            seed_output_dir
            / f"{model_tag}_summary.json"
        )

        model_path = (
            seed_output_dir
            / f"{model_tag}_trained_model.pkl"
        )

        package_path = (
            seed_output_dir
            / f"{model_tag}_model_package.pkl"
        )

        error_path = (
            seed_output_dir
            / f"{model_tag}_error.json"
        )

        start_time = time.time()

        # ====================================================
        # Training
        # ====================================================

        try:

            log(
                f"[regular][{model_name}][seed={seed}] "
                "starting training",
                model_log_name,
            )

            # ------------------------------------------------
            # Model parameter logging
            # ------------------------------------------------

            if model_params:

                log(
                    f"[regular][{model_name}][seed={seed}] "
                    f"model params: {_json_safe(model_params)}",
                    model_log_name,
                )

            else:

                log(
                    f"[regular][{model_name}][seed={seed}] "
                    "model params: using model defaults",
                    model_log_name,
                )

            # ------------------------------------------------
            # Construct base estimator
            # ------------------------------------------------

            base_model = get_model(
                model_name=model_name,
                task_type=task_type,
                seed=seed,
                tune_hyperparameter=False,
                model_params=model_params,
            )

            log(
                f"[regular][{model_name}][seed={seed}] "
                "base model constructed",
                model_log_name,
            )

            # ------------------------------------------------
            # Optional preprocessing / scaling pipeline
            # ------------------------------------------------

            if feature_config is not None:

                model = make_scaled_pipeline(
                    model=base_model,
                    feature_types=feature_config[
                        "feature_types"
                    ],
                )

            else:
                model = base_model

            log(
                f"[regular][{model_name}][seed={seed}] "
                "feature pipeline ready",
                model_log_name,
            )

            # ------------------------------------------------
            # Dataset information
            # ------------------------------------------------

            log(
                f"[regular][{model_name}][seed={seed}] "
                f"X_train shape: "
                f"{getattr(X_train, 'shape', None)}, "
                f"y_train shape: "
                f"{getattr(y_train, 'shape', None)}",
                model_log_name,
            )

            # ------------------------------------------------
            # Fit
            # ------------------------------------------------

            log(
                f"[regular][{model_name}][seed={seed}] "
                "starting fit",
                model_log_name,
            )

            model.fit(
                X_train,
                y_train,
            )

            log(
                f"[regular][{model_name}][seed={seed}] "
                "fit complete",
                model_log_name,
            )

            # ------------------------------------------------
            # Basic results
            # ------------------------------------------------

            results = {
                "model_name": model_name,
                "task_type": task_type,
                "seed": seed,
                "model_params": model_params,
                "runtime_seconds": float(
                    time.time() - start_time
                ),
            }

            # ------------------------------------------------
            # Test-set evaluation
            # ------------------------------------------------

            log(
                f"[regular][{model_name}][seed={seed}] "
                "evaluating on test set",
                model_log_name,
            )

            evaluation_results = evaluate_model(
                model,
                X_test,
                y_test,
                task_type,
            )

            results.update(
                evaluation_results
            )

            log(
                f"[regular][{model_name}][seed={seed}] "
                f"evaluation metrics: "
                f"{_json_safe(results)}",
                model_log_name,
            )

            # =================================================
            # Save summary
            # =================================================

            with open(
                summary_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    _json_safe(results),
                    f,
                    indent=4,
                )

            log(
                f"[regular][{model_name}][seed={seed}] "
                f"wrote summary to {summary_path}",
                model_log_name,
            )

            # =================================================
            # Save trained model
            # =================================================

            with open(
                model_path,
                "wb",
            ) as f:

                pickle.dump(
                    model,
                    f,
                )

            log(
                f"[regular][{model_name}][seed={seed}] "
                f"saved model to {model_path}",
                model_log_name,
            )

            # =================================================
            # Save ChemFlow model package
            # =================================================

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

            with open(
                package_path,
                "wb",
            ) as f:

                pickle.dump(
                    model_package,
                    f,
                )

            log(
                f"[regular][{model_name}][seed={seed}] "
                f"saved model package to {package_path}",
                model_log_name,
            )

            # =================================================
            # Model metadata
            # =================================================

            write_model_info(
                output_dir=seed_output_dir,
                model_name=model_name,
                task_type=task_type,
                feature_config=feature_config or {},
                training_config=single_seed_config,
                metrics=results,
            )

            log(
                f"[regular][{model_name}][seed={seed}] "
                "wrote model info json",
                model_log_name,
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            all_results.append(
                results
            )

            log(
                f"[regular][{model_name}][seed={seed}] "
                f"training complete in "
                f"{results['runtime_seconds']:.2f}s",
                model_log_name,
            )

            # ------------------------------------------------
            # Optional external callback
            # ------------------------------------------------

            if progress_callback is not None:
                progress_callback()

        # ====================================================
        # Error handling
        # ====================================================

        except Exception as e:

            error_info = {
                "model_name": model_name,
                "task_type": task_type,
                "seed": seed,
                "model_params": model_params,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

            with open(
                error_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    _json_safe(error_info),
                    f,
                    indent=4,
                )

            log(
                f"[regular][{model_name}][seed={seed}] "
                f"failed: {e}",
                model_log_name,
            )

            # Preserve the original traceback.
            raise

    # ========================================================
    # Save all-seed summary
    # ========================================================

    summary_df = pd.DataFrame(
        all_results
    )

    summary_path = (
        output_dir
        / f"{model_tag}_all_seed_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    return all_results


# ============================================================
# Regular training for multiple models
# ============================================================

def regular_training_multiple_models(
    X_train,
    y_train,
    X_test,
    y_test,
    model_configs,
    parent_config=None,
    output_dir="model_training_results",
    feature_config=None,
    progress_callback=None,
):
    """
    Train multiple models using regular training.

    Each model may define its own:
        - model_name
        - task_type / task
        - model_params
        - seeds

    If model_params is omitted, estimator defaults are used.
    """

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    # --------------------------------------------------------
    # Normalize configuration container
    # --------------------------------------------------------

    cfg_iter = (
        model_configs.values()
        if isinstance(model_configs, dict)
        else model_configs
    )

    model_cfgs = list(
        cfg_iter
    )

    # --------------------------------------------------------
    # Overall model progress bar
    # --------------------------------------------------------

    model_iterator = tqdm(
        model_cfgs,
        desc="Regular training models",
        unit="model",
        dynamic_ncols=True,
    )

    # ========================================================
    # Train models
    # ========================================================

    for model_cfg in model_iterator:

        # ----------------------------------------------------
        # Merge global + per-model config
        # ----------------------------------------------------

        if parent_config is not None:

            cfg = _merge_parent_config(
                parent_config,
                model_cfg,
            )

        else:

            cfg = dict(
                model_cfg
            )

        model_name = cfg[
            "model_name"
        ]

        model_iterator.set_postfix(
            current=model_name
        )

        # ----------------------------------------------------
        # Model-specific output directory
        # ----------------------------------------------------

        model_output_dir = (
            output_dir
            / safe_name(model_name)
        )

        model_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model_results = regular_training(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_config=cfg,
            output_dir=model_output_dir,
            feature_config=feature_config,
            progress_callback=progress_callback,
        )

        all_results.extend(
            model_results
        )

    # ========================================================
    # Save combined summary
    # ========================================================

    all_summary_df = pd.DataFrame(
        all_results
    )

    all_summary_df.to_csv(
        output_dir
        / "all_regular_model_summary.csv",
        index=False,
    )

    return all_results