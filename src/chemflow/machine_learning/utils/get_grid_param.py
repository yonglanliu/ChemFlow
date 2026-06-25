# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

def get_default_param_grid(model_name: str, task_type: str):
    if model_name in ["Random Forest", "Extra Trees"]:
        return {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }

    elif model_name == "Gradient Boosting":
        return {
            "n_estimators": [100, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 5],
            "subsample": [0.8, 1.0],
        }

    elif model_name == "XGBoost":
        return {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

    elif model_name == "SVM_RBF":
        return {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        }

    elif model_name == "KNN":
        return {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }

    elif model_name == "MLP":
        return {
            "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.0005],
        }

    elif model_name == "Logistic Regression":
        return {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        }

    elif model_name in ["Ridge Regression", "Lasso Regression"]:
        return {
            "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
        }

    elif model_name == "PLS":
        return {
            "n_components": [2, 3, 5, 10, 20],
        }

    else:
        raise ValueError(f"No default parameter grid for {model_name}")
