# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import json
import logging
import os
import pickle
import platform
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union, cast

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from chemflow.machine_learning import (
    MODEL_OPTIONS,
    get_default_param_grid,
    get_model,
    get_refit_metrics,
    get_scoring,
)
from chemflow.machine_learning.data.data_pipeline import (
    fitted_feature_dimensions,
    make_scaled_pipeline,
)
from chemflow.machine_learning.eval.eval_ml import evaluate_model
from chemflow.machine_learning.train.utils import (
    _json_safe,
    _merge_parent_config,
    safe_name,
    write_model_info,
    write_test_predictions,
)


logger = logging.getLogger(__name__)


def _scaffold_group_labels(smiles):
    labels = []
    scaffold_ids = {}
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"Cannot generate a scaffold for SMILES {value!r}.")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule,
            includeChirality=False,
        )
        if scaffold not in scaffold_ids:
            scaffold_ids[scaffold] = len(scaffold_ids)
        labels.append(scaffold_ids[scaffold])
    return np.asarray(labels, dtype=int)


def build_cv_splitter(config, task_type, train_smiles=None):
    """Build ordinary or scaffold-grouped inner cross-validation."""
    n_splits = int(config.get("cv", 5))
    if n_splits < 2:
        raise ValueError("cv must be at least 2.")
    strategy = str(config.get("cv_strategy", "cv")).strip().lower()
    seed = int(config.get("cv_random_seed", config.get("search_seed", 42)))

    if strategy in {"kfold", "cv", "standard"}:
        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            if task_type == "classification"
            else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        )
        return splitter, None, "cv"

    if strategy in {"scaffold", "scaffold_grouped", "scaffold-grouped"}:
        if train_smiles is None:
            raise ValueError(
                "cv_strategy='scaffold-grouped' requires training SMILES. "
                "Regenerate an old cached split so it includes train_smiles, "
                "or use cv_strategy='cv'."
            )
        groups = _scaffold_group_labels(train_smiles)
        unique_groups = int(np.unique(groups).size)
        if unique_groups < n_splits:
            raise ValueError(
                f"Scaffold-grouped CV requested {n_splits} folds but the "
                f"training set contains only {unique_groups} unique scaffolds."
            )
        splitter = (
            StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            if task_type == "classification"
            else GroupKFold(n_splits=n_splits)
        )
        return splitter, groups, "scaffold-grouped"

    raise ValueError(
        "cv_strategy must be 'cv' or 'scaffold-grouped'. "
        f"Got {strategy!r}."
    )


def _materialize_cv_splits(cv, X, y, groups, task_type, scoring_metrics):
    """Create deterministic folds and validate classification coverage."""
    splits = list(cv.split(X, y, groups=groups))
    if not splits:
        raise ValueError("Cross-validation did not produce any folds.")

    needs_all_classes = task_type == "classification" and any(
        metric in {"roc_auc", "average_precision"}
        for metric in scoring_metrics
    )
    if needs_all_classes:
        all_classes = set(np.unique(y))
        for fold_number, (train_indices, validation_indices) in enumerate(
            splits,
            start=1,
        ):
            train_classes = set(np.unique(np.asarray(y)[train_indices]))
            validation_classes = set(np.unique(np.asarray(y)[validation_indices]))
            if train_classes != all_classes or validation_classes != all_classes:
                raise ValueError(
                    f"CV fold {fold_number} does not contain every class in both "
                    "its training and validation portions. Reduce cv, use "
                    "cv_strategy='cv', or collect more examples/scaffolds for "
                    "the missing class before using ROC-AUC or average precision."
                )
    return splits


def _filter_size_dependent_grid(param_grid, model_name, min_fold_train_size, n_features):
    """Remove candidates that cannot be fitted within the smallest CV fold."""
    grid = dict(param_grid)

    def filter_values(parameter, maximum):
        for key in (parameter, f"model__{parameter}"):
            if key not in grid:
                continue
            values = [value for value in grid[key] if int(value) <= maximum]
            if not values:
                raise ValueError(
                    f"No valid {parameter} candidates remain for {model_name}; "
                    f"the maximum allowed by the smallest CV training fold is {maximum}."
                )
            grid[key] = values

    if model_name == "KNN":
        filter_values("n_neighbors", max(1, int(min_fold_train_size)))
    elif model_name == "PLS":
        maximum = max(1, min(int(min_fold_train_size) - 1, int(n_features)))
        filter_values("n_components", maximum)

    return grid


# ============================================================
# Progress bar
# ============================================================

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Connect joblib parallel execution with a tqdm progress bar.

    This allows sklearn RandomizedSearchCV/GridSearchCV jobs to update
    tqdm whenever one or more CV fitting jobs finish.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# ============================================================
# Hyperparameter tuning helpers
# ============================================================

def _prefix_param_grid_for_pipeline(param_grid, estimator):
    """
    Prefix hyperparameters with ``model__`` when the estimator is
    wrapped inside an sklearn Pipeline.
    """

    if not isinstance(estimator, Pipeline):
        return param_grid

    return {
        key if str(key).startswith("model__") else f"model__{key}": value
        for key, value in param_grid.items()
    }


def _is_xgboost_model(model_name: str) -> bool:
    return str(model_name).strip().lower() == "xgboost"


def _set_safe_thread_env_for_tuning() -> None:
    """
    Keep thread usage conservative during CV.

    This reduces the risk of platform-specific crashes caused by
    OpenMP/BLAS/XGBoost thread oversubscription.
    """

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


# ============================================================
# Tune one model
# ============================================================

def tune_hyperparameters(
    X_train,
    y_train,
    X_test,
    y_test,
    config,
    n_classes=None,
    output_dir: Union[str, Path] = "model_tuning_results",
    feature_config=None,
    train_smiles=None,
    test_metadata=None,
):
    """
    Perform RandomizedSearchCV hyperparameter tuning for one model.

    Parameters
    ----------
    X_train
        Training features.

    y_train
        Training labels/targets.

    X_test
        Test features.

    y_test
        Test labels/targets.

    config
        Model training/tuning configuration.

    n_classes
        Number of classes for classification tasks.

    output_dir
        Directory where tuning results and the trained model are saved.

    feature_config
        Feature configuration used to construct preprocessing/scaling
        pipelines.

    Returns
    -------
    best_model
        Best fitted estimator.

    results
        Dictionary containing tuning and test metrics.

    cv_results
        Full RandomizedSearchCV results as a DataFrame.
    """

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    model_name = config["model_name"]

    if model_name not in MODEL_OPTIONS:
        raise ValueError(
            f"Invalid model_name '{model_name}'. "
            f"Expected one of: {MODEL_OPTIONS}"
        )

    task_type = config.get("task", config.get("task_type"))

    if task_type not in ["classification", "regression"]:
        raise ValueError(
            "Invalid task. Expected 'classification' or 'regression'."
        )

    search_seed = int(config.get("search_seed", 42))
    requested_n_iter = int(config.get("n_iter", 50))
    if requested_n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    cv_folds = int(config.get("cv", 5))
    n_jobs = int(config.get("n_jobs", -1))
    cv, cv_groups, cv_strategy = build_cv_splitter(
        config,
        task_type,
        train_smiles=train_smiles,
    )
    if cv_groups is not None and len(cv_groups) != len(X_train):
        raise ValueError(
            "Training SMILES and feature rows are misaligned: "
            f"{len(cv_groups)} SMILES versus {len(X_train)} feature rows."
        )

    # --------------------------------------------------------
    # XGBoost safety settings
    # --------------------------------------------------------

    if _is_xgboost_model(model_name):
        _set_safe_thread_env_for_tuning()

        if n_jobs != 1:
            logger.warning(
                "Forcing RandomizedSearchCV n_jobs=1 for XGBoost "
                "to improve stability (requested n_jobs=%s).",
                n_jobs,
            )

            n_jobs = 1

    # --------------------------------------------------------
    # Scoring
    # --------------------------------------------------------

    scoring = get_scoring(
        config,
        n_classes=n_classes,
    )

    refit_metric = get_refit_metrics(config)

    if refit_metric not in scoring:
        raise ValueError(
            f"refit_metric '{refit_metric}' must be one of "
            f"{list(scoring.keys())}"
        )

    cv_splits = _materialize_cv_splits(
        cv,
        X_train,
        y_train,
        cv_groups,
        task_type,
        config.get("scoring_metrics", []),
    )
    cv = cv_splits
    cv_folds = len(cv_splits)
    min_fold_train_size = min(len(train_indices) for train_indices, _ in cv_splits)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    base_model = get_model(
        model_name=model_name,
        task_type=task_type,
        seed=search_seed,
        tune_hyperparameter=True,
        model_params=config.get("model_params"),
    )

    if model_name == "SVM_RBF" and len(X_train) >= 10_000:
        logger.warning(
            "RBF SVM tuning is expensive for %d training rows. "
            "This run requests %d candidates and %d folds (%d fits). "
            "Consider model-specific n_iter/cv overrides or a tree model.",
            len(X_train),
            requested_n_iter,
            cv_folds,
            requested_n_iter * cv_folds,
        )

    # --------------------------------------------------------
    # Optional preprocessing pipeline
    # --------------------------------------------------------

    if feature_config is not None:
        model = make_scaled_pipeline(
            model=base_model,
            feature_types=feature_config["feature_types"],
            descriptor_correlation_threshold=feature_config.get(
                "descriptor_correlation_threshold", 0.95
            ),
            descriptor_model_selection=feature_config.get(
                "descriptor_model_selection", False
            ),
            descriptor_selection_threshold=feature_config.get(
                "descriptor_selection_threshold", "mean"
            ),
            descriptor_selection_estimators=feature_config.get(
                "descriptor_selection_estimators", 128
            ),
        )

    else:
        model = base_model

    # --------------------------------------------------------
    # Hyperparameter search space
    # --------------------------------------------------------

    param_grid = config.get("param_grid")

    if not param_grid:
        param_grid = get_default_param_grid(
            model_name,
            task_type,
        )

    param_grid = _filter_size_dependent_grid(
        param_grid,
        model_name=model_name,
        min_fold_train_size=min_fold_train_size,
        n_features=X_train.shape[1],
    )

    param_grid = _prefix_param_grid_for_pipeline(
        param_grid,
        model,
    )

    try:
        available_candidates = len(ParameterGrid(param_grid))
    except TypeError:
        available_candidates = requested_n_iter
    n_iter = min(requested_n_iter, available_candidates)

    # --------------------------------------------------------
    # XGBoost-specific tuning safety
    # --------------------------------------------------------

    if _is_xgboost_model(model_name):

        xgb_n_jobs_key = (
            "model__n_jobs"
            if isinstance(model, Pipeline)
            else "n_jobs"
        )

        xgb_nthread_key = (
            "model__nthread"
            if isinstance(model, Pipeline)
            else "nthread"
        )

        # Keep XGBoost itself single-threaded during CV
        param_grid[xgb_n_jobs_key] = [1]
        param_grid[xgb_nthread_key] = [1]

        if platform.system() == "Darwin":
            logger.warning(
                "Using safe XGBoost tuning settings on macOS "
                "(n_jobs=1, nthread=1)."
            )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_tag = safe_name(model_name)

    summary_path = (
        output_dir /
        f"{model_tag}_summary.json"
    )

    cv_path = (
        output_dir /
        f"{model_tag}_cv_results.csv"
    )

    model_path = (
        output_dir /
        f"{model_tag}_best_model.pkl"
    )

    package_path = (
        output_dir /
        f"{model_tag}_model_package.pkl"
    )

    error_path = (
        output_dir /
        f"{model_tag}_error.json"
    )

    start_time = time.time()

    # --------------------------------------------------------
    # Hyperparameter tuning
    # --------------------------------------------------------

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

            # tqdm handles the output
            verbose=0,

            return_train_score=True,
            # A single estimator-specific numerical failure should not abort
            # every other candidate. RandomizedSearchCV still raises when all
            # candidates fail.
            error_score=np.nan,
        )

        # Number of CV model-fitting jobs
        total_fits = n_iter * cv_folds

        logger.info(
            "Starting hyperparameter tuning for %s with %s: "
            "%d candidates × %d folds = %d fits",
            model_name,
            cv_strategy,
            n_iter,
            cv_folds,
            total_fits,
        )

        # ----------------------------------------------------
        # CV progress bar
        # ----------------------------------------------------

        progress_bar = tqdm(
            total=total_fits,
            desc=f"Tuning {model_name}",
            unit="fit",
            leave=True,
            dynamic_ncols=True,
            postfix={
                "candidates": n_iter,
                "cv": cv_folds,
            },
        )

        with tqdm_joblib(progress_bar):
            search.fit(X_train, y_train)

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        best_model = search.best_estimator_

        cv_results = pd.DataFrame(
            search.cv_results_
        )

        best_idx = search.best_index_

        # ----------------------------------------------------
        # Basic tuning results
        # ----------------------------------------------------

        results = {
            "model_name": model_name,
            "task_type": task_type,
            "best_params": search.best_params_,
            "refit_metric": refit_metric,
            "best_cv_score_raw": float(
                search.best_score_
            ),
            "runtime_seconds": float(
                time.time() - start_time
            ),
            "search_candidates_requested": requested_n_iter,
            "search_candidates_evaluated": n_iter,
            "cv_strategy": cv_strategy,
            "cv_folds": cv_folds,
            "cv_unique_scaffolds": (
                int(np.unique(cv_groups).size) if cv_groups is not None else None
            ),
            "feature_reduction": fitted_feature_dimensions(
                best_model,
                X_train.shape[1],
            ),
        }

        # ----------------------------------------------------
        # Extract requested CV metrics
        # ----------------------------------------------------

        for metric in config.get(
            "scoring_metrics",
            [],
        ):

            col = f"mean_test_{metric}"

            if col in cv_results.columns:

                raw_value = cv_results.loc[
                    best_idx,
                    col,
                ]

                value = float(
                    cast(
                        Any,
                        raw_value,
                    )
                )

                # sklearn uses negative loss metrics
                if metric in [
                    "root_mean_squared_error",
                    "mean_absolute_error",
                    "rmse",
                    "mae",
                ]:
                    value = abs(value)

                results[
                    f"best_cv_{metric}"
                ] = value

        # ----------------------------------------------------
        # Test-set evaluation
        # ----------------------------------------------------

        test_results = evaluate_model(
            best_model,
            X_test,
            y_test,
            task_type,
        )

        results.update(
            test_results
        )

        write_test_predictions(
            output_dir=output_dir,
            model_name=model_name,
            task_type=task_type,
            evaluation_results=test_results,
            test_metadata=test_metadata,
        )

        # ----------------------------------------------------
        # Save summary
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Save full CV results
        # ----------------------------------------------------

        cv_results.to_csv(
            cv_path,
            index=False,
        )

        # ----------------------------------------------------
        # Save best estimator
        # ----------------------------------------------------

        with open(
            model_path,
            "wb",
        ) as f:

            pickle.dump(
                best_model,
                f,
            )

        # ----------------------------------------------------
        # Save ChemFlow model package
        # ----------------------------------------------------

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

        with open(
            package_path,
            "wb",
        ) as f:

            pickle.dump(
                model_package,
                f,
            )

        # ----------------------------------------------------
        # Write human-readable model metadata
        # ----------------------------------------------------

        write_model_info(
            output_dir=output_dir,
            model_name=model_name,
            task_type=task_type,
            feature_config=feature_config or {},
            training_config=config,
            metrics=results,
        )

        logger.info(
            "Finished tuning %s in %.2f seconds.",
            model_name,
            results["runtime_seconds"],
        )

        return (
            best_model,
            results,
            cv_results,
        )

    # --------------------------------------------------------
    # Error handling
    # --------------------------------------------------------

    except Exception as e:

        error_info = {
            "model_name": model_name,
            "task_type": task_type,
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

        logger.exception(
            "Hyperparameter tuning failed for %s.",
            model_name,
        )

        raise


# ============================================================
# Tune multiple models
# ============================================================

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
    train_smiles=None,
    test_metadata=None,
    progress_callback=None,
):
    """
    Tune multiple machine-learning models.

    Provides two levels of progress reporting:

    1. Overall model progress.
    2. CV fitting progress for the current model.
    """

    all_results = []

    # --------------------------------------------------------
    # Normalize model configs
    # --------------------------------------------------------

    cfg_iter = (
        cfgs.values()
        if isinstance(cfgs, dict)
        else cfgs
    )

    model_cfgs = list(cfg_iter)

    # --------------------------------------------------------
    # Overall model progress
    # --------------------------------------------------------

    model_progress = tqdm(
        model_cfgs,
        desc="Hyperparameter tuning models",
        unit="model",
        dynamic_ncols=True,
        position=0,
    )

    for model_cfg in model_progress:

        cfg = _merge_parent_config(
            parent_config,
            model_cfg,
        )

        model_name = cfg["model_name"]

        model_progress.set_postfix(
            current=model_name
        )

        # ----------------------------------------------------
        # Model-specific output directory
        # ----------------------------------------------------

        model_output_dir = (
            Path(output_dir) /
            safe_name(model_name)
        )

        model_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Tune model
        # ----------------------------------------------------

        _, results, _ = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=cfg,
            n_classes=n_classes,
            output_dir=model_output_dir,
            feature_config=feature_config,
            train_smiles=train_smiles,
            test_metadata=test_metadata,
        )

        all_results.append(
            results
        )

        # ----------------------------------------------------
        # Optional external progress callback
        # ----------------------------------------------------

        if progress_callback is not None:
            progress_callback()

    # --------------------------------------------------------
    # Save summary for all models
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        all_results
    )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        output_dir /
        "all_model_summary.csv",
        index=False,
    )

    return all_results
