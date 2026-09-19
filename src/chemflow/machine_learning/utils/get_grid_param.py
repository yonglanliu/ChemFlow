# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

def get_default_param_grid(model_name: str, task_type: str):
    """Return a conservative randomized-search space for a supported model."""
    task_type = str(task_type).strip().lower()
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    if model_name == "Random Forest":
        grid = {
            "n_estimators": [300, 600, 1000],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.25, 0.5],
            "bootstrap": [True, False],
        }
        if task_type == "classification":
            grid["class_weight"] = [None, "balanced", "balanced_subsample"]
        return grid

    if model_name == "Extra Trees":
        grid = {
            "n_estimators": [300, 600, 1000],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.25, 0.5],
            "bootstrap": [False, True],
        }
        if task_type == "classification":
            grid["class_weight"] = [None, "balanced"]
        return grid

    if model_name == "Gradient Boosting":
        grid = {
            "n_estimators": [100, 300, 600],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [2, 5, 10, 20],
            "subsample": [0.6, 0.8, 1.0],
            "max_features": ["sqrt", 0.5, None],
        }
        if task_type == "regression":
            grid["loss"] = ["squared_error", "huber"]
        return grid

    if model_name == "XGBoost":
        return {
            "n_estimators": [200, 500, 800, 1200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.7, 0.9],
            "min_child_weight": [1, 3, 5, 10],
            "gamma": [0.0, 0.1, 0.5],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.1, 1.0, 5.0, 10.0],
        }

    if model_name == "LightGBM":
        grid = {
            "n_estimators": [200, 400, 800, 1200],
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [15, 31, 63],
            "max_depth": [4, 6, 8],
            "min_child_samples": [20, 40, 80],
            "subsample": [0.7, 0.8, 0.9],
            "subsample_freq": [1],
            "colsample_bytree": [0.6, 0.7, 0.8],
            "reg_alpha": [0.01, 0.1, 1.0],
            "reg_lambda": [1.0, 5.0, 10.0],
        }
        if task_type == "classification":
            grid["class_weight"] = [None, "balanced"]
        return grid

    if model_name == "SVM_RBF":
        grid = {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1],
        }
        if task_type == "classification":
            grid["class_weight"] = [None, "balanced"]
        else:
            grid["epsilon"] = [0.05, 0.1, 0.2, 0.5]
        return grid

    if model_name == "KNN":
        return {
            "n_neighbors": [3, 5, 7, 11, 21, 31],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
            "metric": ["minkowski"],
            "algorithm": ["brute"],
        }

    if model_name == "MLP":
        return {
            "hidden_layer_sizes": [(64,), (128,), (256,), (128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "learning_rate_init": [0.0001, 0.0005, 0.001],
            "early_stopping": [True],
            "max_iter": [1000],
        }

    if model_name == "Logistic Regression":
        if task_type != "classification":
            raise ValueError("Logistic Regression only supports classification.")
        return {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "class_weight": [None, "balanced"],
            "max_iter": [2000],
        }

    if model_name == "Ridge Regression":
        if task_type != "regression":
            raise ValueError("Ridge Regression only supports regression.")
        return {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

    if model_name == "Lasso Regression":
        if task_type != "regression":
            raise ValueError("Lasso Regression only supports regression.")
        return {
            "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            "max_iter": [10000],
            "selection": ["cyclic", "random"],
        }

    if model_name == "PLS":
        if task_type != "regression":
            raise ValueError("PLS only supports regression.")
        return {
            "n_components": [2, 3, 5, 8, 10, 15, 20],
            "scale": [True, False],
            "max_iter": [1000],
        }

    raise ValueError(f"No default parameter grid for {model_name}")
