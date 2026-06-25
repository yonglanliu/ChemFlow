# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

# Tree models
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

# XGBoost
from xgboost import (
    XGBClassifier,
    XGBRegressor,
)

# SVM
from sklearn.svm import (
    SVC,
    SVR,
)

# KNN
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)

# Neural Network
from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor,
)

# Linear Models
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    Lasso,
)

# Partial Least Squares
from sklearn.cross_decomposition import PLSRegression
from src.chemflow.machine_learning.utils.scoring import get_scoring_config


def get_base_model_config(
    model_id,
    model_name,
    task_type,
    search_seed,
    eval_seeds,
    hyperparameter_tuning=True,
    multi_seed_evaluation=False,
):
    task = task_type.lower()
    if task not in ["classification", "regression"]:
        raise ValueError(
            f"Invalid task_type: {task_type}. "
            "Expected classification or regression."
        )
    scoring_cfg = get_scoring_config(task)

    return {
        "model_id": model_id,
        "model_name": model_name,
        "task_type": task,

        "hyperparameter_tuning": bool(hyperparameter_tuning),
        "tuning_method": "RandomizedSearchCV",
        "n_iter": 50,
        "cv": 5,
        "search_seed": search_seed,

        "multi_seed_evaluation": bool(multi_seed_evaluation),
        "eval_seeds": eval_seeds if multi_seed_evaluation else [],
        "n_runs": len(eval_seeds) if multi_seed_evaluation else 1,

        "scoring_metrics": scoring_cfg["scoring_metrics"],
        "refit_metric": scoring_cfg["refit_metric"],
        "n_jobs": -1,
    }

def get_model_config(
    model_id,
    model_name,
    task_type,
    search_seed,
    eval_seeds,
    hyperparameter_tuning=True,
    multi_seed_evaluation=False,
):
    task = task_type.lower()
    if task not in ["classification", "regression"]:
        raise ValueError(
            f"Invalid task_type: {task_type}. "
            "Expected classification or regression."
        )

    cfg = get_base_model_config(
        model_id=model_id,
        model_name=model_name,
        task_type=task,
        search_seed=search_seed,
        eval_seeds=eval_seeds,
        hyperparameter_tuning=hyperparameter_tuning,
        multi_seed_evaluation=multi_seed_evaluation,
    )

    if model_name == "Random Forest":
        cfg["estimator"] = (
            "RandomForestClassifier"
            if task == "classification"
            else "RandomForestRegressor"
        )
        cfg["param_grid"] = {
            "n_estimators": [100, 200, 300, 500, 800],
            "max_depth": [None, 5, 10, 20, 40],
            "max_features": ["sqrt", "log2", None],
            "min_samples_leaf": [1, 2, 4, 8],
            "min_samples_split": [2, 5, 10],
            "bootstrap": [True, False],
        }

    elif model_name == "Extra Trees":
        cfg["estimator"] = (
            "ExtraTreesClassifier"
            if task == "classification"
            else "ExtraTreesRegressor"
        )
        cfg["param_grid"] = {
            "n_estimators": [100, 200, 500, 800],
            "max_depth": [None, 5, 10, 20, 40],
            "max_features": ["sqrt", "log2", None],
            "min_samples_leaf": [1, 2, 4, 8],
            "min_samples_split": [2, 5, 10],
        }

    elif model_name == "Gradient Boosting":
        cfg["estimator"] = (
            "GradientBoostingClassifier"
            if task == "classification"
            else "GradientBoostingRegressor"
        )
        cfg["param_grid"] = {
            "n_estimators": [100, 200, 500],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.6, 0.8, 1.0],
        }

    elif model_name == "XGBoost":
        cfg["estimator"] = (
            "XGBClassifier"
            if task == "classification"
            else "XGBRegressor"
        )
        cfg["param_grid"] = {
            "n_estimators": [100, 300, 500, 1000],
            "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6, 8],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.1, 1.0, 5.0, 10.0],
        }

    elif model_name == "SVM_RBF":
        cfg["estimator"] = "SVC" if task == "classification" else "SVR"
        cfg["param_grid"] = {
            "C": [0.01, 0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "kernel": ["rbf"],
        }

    elif model_name == "KNN":
        cfg["estimator"] = (
            "KNeighborsClassifier"
            if task == "classification"
            else "KNeighborsRegressor"
        )
        cfg["param_grid"] = {
            "n_neighbors": [3, 5, 7, 9, 15, 25],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }

    elif model_name == "MLP":
        cfg["estimator"] = (
            "MLPClassifier"
            if task == "classification"
            else "MLPRegressor"
        )
        cfg["param_grid"] = {
            "hidden_layer_sizes": [(128,), (256,), (128, 64), (256, 128)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.0001, 0.001, 0.01],
            "max_iter": [500],
        }

    elif model_name == "Logistic Regression":
        if task == "regression":
            return None

        cfg["estimator"] = "LogisticRegression"
        cfg["param_grid"] = {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
            "max_iter": [1000],
        }

    elif model_name == "Ridge Regression":
        if task == "classification":
            return None

        cfg["estimator"] = "Ridge"
        cfg["param_grid"] = {
            "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
        }

    elif model_name == "Lasso Regression":
        if task == "classification":
            return None

        cfg["estimator"] = "Lasso"
        cfg["param_grid"] = {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
            "max_iter": [5000],
        }

    elif model_name == "PLS":
        if task == "classification":
            return None

        cfg["estimator"] = "PLSRegression"
        cfg["param_grid"] = {
            "n_components": [2, 3, 5, 8, 10, 15, 20],
            "scale": [True, False],
        }

    else:
        return None

    return cfg


def build_models_config(
    selected_models,
    task_type,
    search_seed,
    eval_seeds,
    hyperparameter_tuning=True,
    multi_seed_evaluation=False,
):
    models = {}
    skipped_models = []

    for i, model_name in enumerate(selected_models):
        model_cfg = get_model_config(
            model_id=i,
            model_name=model_name,
            task_type=task_type,
            search_seed=search_seed,
            eval_seeds=eval_seeds,
            hyperparameter_tuning=hyperparameter_tuning,
            multi_seed_evaluation=multi_seed_evaluation,
        )

        if model_cfg is None:
            skipped_models.append(model_name)
            continue

        models[model_name] = model_cfg

    return models, skipped_models


def get_model(model_name: str, task_type: str, search_seed: int = 42):
    if model_name == "Random Forest":
        return RandomForestClassifier(random_state=search_seed) if task_type == "classification" else RandomForestRegressor(random_state=search_seed)

    elif model_name == "Extra Trees":
        return ExtraTreesClassifier(random_state=search_seed) if task_type == "classification" else ExtraTreesRegressor(random_state=search_seed)

    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=search_seed) if task_type == "classification" else GradientBoostingRegressor(random_state=search_seed)

    elif model_name == "XGBoost":
        return XGBClassifier(random_state=search_seed, eval_metric="logloss", n_jobs=-1) if task_type == "classification" else XGBRegressor(random_state=search_seed, n_jobs=-1)

    elif model_name == "SVM_RBF":
        return SVC(kernel="rbf", probability=True, random_state=search_seed) if task_type == "classification" else SVR(kernel="rbf")

    elif model_name == "KNN":
        return KNeighborsClassifier() if task_type == "classification" else KNeighborsRegressor()

    elif model_name == "MLP":
        return MLPClassifier(max_iter=1000, random_state=search_seed) if task_type == "classification" else MLPRegressor(max_iter=1000, random_state=search_seed)

    elif model_name == "Logistic Regression":
        if task_type != "classification":
            raise ValueError("Logistic Regression only supports classification.")
        return LogisticRegression(max_iter=5000, random_state=search_seed)

    elif model_name == "Ridge Regression":
        if task_type != "regression":
            raise ValueError("Ridge Regression only supports regression.")
        return Ridge()

    elif model_name == "Lasso Regression":
        if task_type != "regression":
            raise ValueError("Lasso Regression only supports regression.")
        return Lasso(max_iter=5000)

    elif model_name == "PLS":
        if task_type != "regression":
            raise ValueError("PLS only supports regression.")
        return PLSRegression()

    else:
        raise ValueError(f"Unknown model_name: {model_name}")