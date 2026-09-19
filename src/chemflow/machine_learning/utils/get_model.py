import yaml
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression

from chemflow.machine_learning.utils.scoring import get_scoring_config
from chemflow.config import CURRENT_DIR as CONFIG_DIR

GRID_SEARCH_CONFIG_PATH = CONFIG_DIR / "grid_search_conf.yaml"
ML_CONFIG_PATH = CONFIG_DIR / "ml_model_conf.yaml"

ESTIMATOR_REGISTRY = {
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "XGBClassifier": XGBClassifier,
    "XGBRegressor": XGBRegressor,
    "LGBMClassifier": LGBMClassifier,
    "LGBMRegressor": LGBMRegressor,
    "SVC": SVC,
    "SVR": SVR,
    "KNeighborsClassifier": KNeighborsClassifier,
    "KNeighborsRegressor": KNeighborsRegressor,
    "MLPClassifier": MLPClassifier,
    "MLPRegressor": MLPRegressor,
    "LogisticRegression": LogisticRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "PLSRegression": PLSRegression,
}


def load_yaml_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_base_model_config(
    model_id,
    model_name,
    task_type,
    seeds,
    hyperparameter_tuning=True,
    defaults=None,
):
    task = task_type.lower()

    if task not in ["classification", "regression"]:
        raise ValueError("task_type must be classification or regression.")
    
    scoring_cfg = get_scoring_config(task)
    defaults = defaults or {}

    return {
        "model_id": model_id,
        "model_name": model_name,
        "task_type": task,

        "hyperparameter_tuning": bool(hyperparameter_tuning),
        "tuning_method": defaults.get("tuning_method", "RandomizedSearchCV") if hyperparameter_tuning else None,
        "n_iter": defaults.get("n_iter", 50) if hyperparameter_tuning else None,
        "cv": defaults.get("cv", 5) if hyperparameter_tuning else None,
        "cv_strategy": defaults.get("cv_strategy", "cv") if hyperparameter_tuning else None,
        "cv_random_seed": defaults.get("cv_random_seed", 42) if hyperparameter_tuning else None,
        "seeds": seeds,
        "n_runs": len(seeds),
        "scoring_metrics": scoring_cfg["scoring_metrics"],
        "refit_metric": scoring_cfg["refit_metric"],
        "n_jobs": defaults.get("n_jobs", -1),
    } if hyperparameter_tuning else {
        "model_id": model_id,
        "model_name": model_name,
        "task_type": task,

        "hyperparameter_tuning": False,
        "seeds": seeds,
        "n_runs": len(seeds),
        "scoring_metrics": scoring_cfg["scoring_metrics"],
        "refit_metric": scoring_cfg["refit_metric"],
        "n_jobs": defaults.get("n_jobs", -1),
    }


def get_model_config(
    model_id,
    model_cfg,
    task_type,
    seeds,
    defaults,
    hyperparameter_tuning=True,
):
    cfg = get_base_model_config(
        model_id=model_id,
        model_name=model_cfg["model_name"],
        task_type=task_type,
        seeds=seeds,
        hyperparameter_tuning=hyperparameter_tuning,
        defaults=defaults,
    )

    cfg["estimator"] = model_cfg["estimator"]
    if hyperparameter_tuning:
        cfg["param_grid"] = model_cfg.get("param_grid", {})
    else:
        cfg["model_params"] = model_cfg.get("model_params", {})

    return cfg


def build_models_config(
    selected_models,
    task_type,
    seeds,
    hyperparameter_tuning=True,
):
    if hyperparameter_tuning:
        yaml_cfg = load_yaml_config(GRID_SEARCH_CONFIG_PATH)
    else:
        yaml_cfg = load_yaml_config(ML_CONFIG_PATH)
    defaults = yaml_cfg.get("defaults", {})

    if task_type == "classification":
        available_model_cfgs = yaml_cfg["classification_models"]
    elif task_type == "regression":
        available_model_cfgs = yaml_cfg["regression_models"]
    else:
        raise ValueError("task_type must be classification or regression.")

    available_models = {
        cfg["model_name"]: cfg for cfg in available_model_cfgs
    }

    models = {}
    skipped_models = []

    for i, model_name in enumerate(selected_models):
        if model_name not in available_models:
            skipped_models.append(model_name)
            continue

        model_cfg = get_model_config(
            model_id=i,
            model_cfg=available_models[model_name],
            task_type=task_type,
            seeds=seeds,
            defaults=defaults,
            hyperparameter_tuning=hyperparameter_tuning,
        )

        models[model_name] = model_cfg

    return models, skipped_models


def get_model(
    model_name: str,
    task_type: str,
    seed: int = 42,
    tune_hyperparameter: bool = True,
    model_params: dict | None = None,
):
    task_type = task_type.lower()

    if task_type not in ["classification", "regression"]:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    params = {} if tune_hyperparameter else dict(model_params or {})

    def add_seed(params):
        params = dict(params)
        params.setdefault("random_state", seed)
        return params

    if model_name == "Random Forest":
        cls = RandomForestClassifier if task_type == "classification" else RandomForestRegressor
        params.setdefault("n_estimators", 500)
        params.setdefault("n_jobs", 1 if tune_hyperparameter else -1)
        if task_type == "classification" and not tune_hyperparameter:
            params.setdefault("class_weight", "balanced")
        return cls(**add_seed(params))

    elif model_name == "Extra Trees":
        cls = ExtraTreesClassifier if task_type == "classification" else ExtraTreesRegressor
        params.setdefault("n_estimators", 500)
        params.setdefault("n_jobs", 1 if tune_hyperparameter else -1)
        if task_type == "classification" and not tune_hyperparameter:
            params.setdefault("class_weight", "balanced")
        return cls(**add_seed(params))

    elif model_name == "Gradient Boosting":
        cls = GradientBoostingClassifier if task_type == "classification" else GradientBoostingRegressor
        if not tune_hyperparameter:
            params.setdefault("n_estimators", 300)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("max_depth", 3)
            params.setdefault("subsample", 0.8)
        return cls(**add_seed(params))

    elif model_name == "XGBoost":
        cls = XGBClassifier if task_type == "classification" else XGBRegressor
        params = add_seed(params)
        params["n_jobs"] = 1  # Avoid nested OpenMP parallelism during CV.
        params.setdefault("verbosity", 0)
        if task_type == "classification":
            params.setdefault("eval_metric", "logloss")
        elif not tune_hyperparameter:
            params.setdefault("objective", "reg:squarederror")
        if not tune_hyperparameter:
            params.setdefault("n_estimators", 500)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("max_depth", 5)
            params.setdefault("subsample", 0.8)
            params.setdefault("colsample_bytree", 0.8)
            params.setdefault("reg_alpha", 0.1)
            params.setdefault("reg_lambda", 1.0)
        return cls(**params)
    
    elif model_name == "LightGBM":
        cls = LGBMClassifier if task_type == "classification" else LGBMRegressor

        params = add_seed(params)
        params.setdefault("n_jobs", 1)
        params.setdefault("verbosity", -1)
        if not tune_hyperparameter:
            params.setdefault("n_estimators", 500)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("num_leaves", 31)
            params.setdefault("max_depth", 6)
            params.setdefault("min_child_samples", 20)
            params.setdefault("subsample", 0.8)
            params.setdefault("subsample_freq", 1)
            params.setdefault("colsample_bytree", 0.8)
            params.setdefault("reg_alpha", 0.1)
            params.setdefault("reg_lambda", 1.0)
            if task_type == "classification":
                params.setdefault("class_weight", "balanced")

        return cls(**params)

    elif model_name == "SVM_RBF":
        if task_type == "classification":
            params.setdefault("kernel", "rbf")
            params.setdefault("probability", True)
            params.setdefault("random_state", seed)
            return SVC(**params)
        else:
            params.setdefault("kernel", "rbf")
            return SVR(**params)

    elif model_name == "KNN":
        cls = KNeighborsClassifier if task_type == "classification" else KNeighborsRegressor
        params.setdefault("n_neighbors", 5)
        params.setdefault("weights", "distance")
        params.setdefault("algorithm", "brute")
        return cls(**params)

    elif model_name == "MLP":
        cls = MLPClassifier if task_type == "classification" else MLPRegressor
        params = add_seed(params)
        params.setdefault("max_iter", 1000)
        params.setdefault("early_stopping", True)
        params.setdefault("n_iter_no_change", 20)
        return cls(**params)

    elif model_name == "Logistic Regression":
        if task_type != "classification":
            raise ValueError("Logistic Regression only supports classification.")
        params = add_seed(params)
        params.setdefault("max_iter", 2000)
        if not tune_hyperparameter:
            params.setdefault("class_weight", "balanced")
        return LogisticRegression(**params)

    elif model_name == "Ridge Regression":
        if task_type != "regression":
            raise ValueError("Ridge Regression only supports regression.")
        return Ridge(**params)

    elif model_name == "Lasso Regression":
        if task_type != "regression":
            raise ValueError("Lasso Regression only supports regression.")
        params.setdefault("max_iter", 10000)
        return Lasso(**params)

    elif model_name == "PLS":
        if task_type != "regression":
            raise ValueError("PLS only supports regression.")
        params.setdefault("max_iter", 1000)
        return PLSRegression(**params)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")
