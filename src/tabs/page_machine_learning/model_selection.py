import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.design import temp_success
from src.models import MODEL_OPTIONS


def normalize_task(task_type: str) -> str:
    task = task_type.lower()

    if task not in ["classification", "regression"]:
        raise ValueError(
            f"Invalid task_type: {task_type}. "
            "Expected classification or regression."
        )

    return task


def get_scoring_config(task_type):
    task = normalize_task(task_type)

    if task == "classification":
        return {
            "scoring_metrics": [
                "roc_auc",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
            ],
            "refit_metric": "f1",
        }

    return {
        "scoring_metrics": [
            "r2",
            "root_mean_squared_error",
            "mean_absolute_error",
        ],
        "refit_metric": "root_mean_squared_error",
    }


def parse_eval_seeds(seed_text: str):
    try:
        return [
            int(x.strip())
            for x in seed_text.split(",")
            if x.strip()
        ]
    except ValueError:
        st.error("Evaluation seeds must be comma-separated integers.")
        st.stop()


def get_base_model_config(
    model_id,
    model_name,
    task_type,
    search_seed,
    eval_seeds,
    hyperparameter_tuning=True,
    multi_seed_evaluation=False,
):
    task = normalize_task(task_type)
    scoring_cfg = get_scoring_config(task)

    return {
        "model_id": model_id,
        "model_name": model_name,
        "task": task,

        "hyperparameter_tuning": bool(hyperparameter_tuning),
        "tuning_method": "RandomizedSearchCV",
        "n_iter": 50,
        "cv": 5,
        "search_seed": int(search_seed),

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
    task = normalize_task(task_type)

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


def models_to_dataframe(models):
    rows = []

    for model_name, cfg in models.items():
        rows.append(
            {
                "Model": model_name,
                "Estimator": cfg["estimator"],
                "Task": cfg["task"],
                "Search Seed": cfg["search_seed"],
                "Multi-seed": cfg["multi_seed_evaluation"],
                "Evaluation Runs": cfg["n_runs"],
                "Evaluation Seeds": str(cfg["eval_seeds"]),
                "CV": cfg["cv"],
                "Search Method": cfg["tuning_method"],
                "Iterations": cfg["n_iter"],
                "Refit Metric": cfg["refit_metric"],
                "Metrics": ", ".join(cfg["scoring_metrics"]),
                "Hyperparameters": len(cfg["param_grid"]),
            }
        )

    return pd.DataFrame(rows)


def hyperparameters_to_dataframe(models):
    rows = []

    for model_name, cfg in models.items():
        for hp_name, hp_values in cfg["param_grid"].items():
            rows.append(
                {
                    "Model": model_name,
                    "Hyperparameter": hp_name,
                    "Values": str(hp_values),
                }
            )

    return pd.DataFrame(rows)


def design(workdir):
    if "show_model_info" not in st.session_state:
        st.session_state["show_model_info"] = False

    st.subheader("Model Selection")

    selected_models = st.multiselect(
        "Algorithms",
        MODEL_OPTIONS,
        default=["Random Forest", "XGBoost"],
        key="ml_selected_models",
    )

    c1, c2, c3, c4, c5 = st.columns(5, vertical_alignment="bottom")

    with c1:
        task_type = st.selectbox(
            "Task",
            ["classification", "regression"],
            key="ml_task",
        )

    with c2:
        hyperparameter_tuning = st.checkbox(
            "Hyperparameter Tuning",
            value=True,
            key="ml_hyperparameter_tuning",
        )

    with c3:
        search_seed = st.number_input(
            "Hyperparameter Search Seed",
            value=42,
            step=1,
            key="ml_search_seed",
        )

    with c4:
        multi_seed_evaluation = st.checkbox(
            "Run Multiple Seeds",
            value=False,
            key="ml_multi_seed_evaluation",
        )

    eval_seeds = []

    with c5:
        if multi_seed_evaluation:
            eval_seeds_input = st.text_input(
                label="Evaluation Seeds",
                value="42,123,456,500,800",
                help="Comma-separated random seeds used for multi-seed evaluation.",
                key="ml_eval_seeds",
            )
            eval_seeds = parse_eval_seeds(eval_seeds_input)
        else:
            st.caption("Single-seed evaluation")

    models, skipped_models = build_models_config(
        selected_models=selected_models,
        task_type=task_type,
        search_seed=int(search_seed),
        eval_seeds=eval_seeds,
        hyperparameter_tuning=hyperparameter_tuning,
        multi_seed_evaluation=multi_seed_evaluation,
    )

    if skipped_models:
        st.warning("Skipped incompatible model(s): " + ", ".join(skipped_models))

    st.session_state["ml_models_config"] = models

    c1, c2, c3 = st.columns(3, vertical_alignment="bottom")

    with c1:
        if st.button(
            "Show Model Information",
            use_container_width=True,
            key="show_model_info_button",
        ):
            st.session_state["show_model_info"] = True

    with c2:
        if st.button(
            "Hide Model Information",
            use_container_width=True,
            key="hide_model_info_button",
        ):
            st.session_state["show_model_info"] = False

    with c3:
        if st.button(
            "Save Configuration",
            type="primary",
            use_container_width=True,
            key="save_model_config_button",
        ):
            output_dir = Path(workdir)
            output_dir.mkdir(parents=True, exist_ok=True)

            json_file = output_dir / "model_config.json"

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(models, f, indent=4)

            temp_success(f"Saved: {json_file}")

    if st.session_state["show_model_info"]:
        st.subheader("Model Summary")

        model_df = models_to_dataframe(models)

        st.dataframe(
            model_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Hyperparameter Search Space")

        hp_df = hyperparameters_to_dataframe(models)

        st.dataframe(
            hp_df,
            use_container_width=True,
            hide_index=True,
        )

    return models, task_type, int(search_seed), bool(hyperparameter_tuning)