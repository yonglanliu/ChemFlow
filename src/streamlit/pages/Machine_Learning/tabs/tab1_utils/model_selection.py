# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import yaml
from src.streamlit.utils.design import temp_success
from src.chemflow.machine_learning import MODEL_OPTIONS, build_models_config
from src.config import PROJECT_ROOT 
from src.streamlit.utils.select_file import file_picker

GRID_SEARCH_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "grid_search_conf.yaml"
ML_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "ml_model_conf.yaml"


def load_yaml_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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

def models_to_dataframe(models, hyperparameter_tuning=True):
    rows = []

    for model_name, cfg in models.items():
        if hyperparameter_tuning:
            rows.append(
                {
                    "Model": model_name,
                    "Estimator": cfg["estimator"],
                    "Task": cfg["task_type"],
                    "Seeds": str(cfg["seeds"]),
                    "Evaluation Runs": cfg["n_runs"],
                    "CV": cfg["cv"],
                    "Search Method": cfg["tuning_method"],
                    "Iterations": cfg["n_iter"],
                    "Refit Metric": cfg["refit_metric"],
                    "Metrics": ", ".join(cfg["scoring_metrics"]),
                    "No. Hyperparameters": len(cfg["param_grid"]),
                }
            )
        else:
            rows.append(
                {
                    "Model": model_name,
                    "Estimator": cfg["estimator"],
                    "Task": cfg["task_type"],
                    "Seeds": str(cfg["seeds"]),
                    "Evaluation Runs": cfg["n_runs"],      
                    "Refit Metric": cfg["refit_metric"],
                    "Metrics": ", ".join(cfg["scoring_metrics"]),
                    "model_params": cfg["model_params"],
                }
            )


    return pd.DataFrame(rows)


def hyperparameters_to_dataframe(models, hyperparameter_tuning=True):
    rows = []

    for model_name, cfg in models.items():
        if hyperparameter_tuning:
            for hp_name, hp_values in cfg["param_grid"].items():
                rows.append(
                    {
                        "Model": model_name,
                        "Hyperparameter": hp_name,
                        "Values": str(hp_values),
                }
            )
        else:
            for hp_name, hp_value in cfg["model_params"].items():
                rows.append(
                    {
                        "Model": model_name,
                        "Parameter": hp_name,
                        "Values": str(hp_value),
                    }
                )
    return pd.DataFrame(rows)


def design():
    default_config = load_yaml_config(GRID_SEARCH_CONFIG_PATH)["defaults"]
    if "show_model_info" not in st.session_state:
        st.session_state["show_model_info"] = False

    st.subheader("Model Selection")

    selected_models = st.multiselect(
        "Algorithms",
        MODEL_OPTIONS,
        default=["Random Forest", "XGBoost"],
        key="ml_selected_models",
    )

    c1, c2, c3 = st.columns(3, vertical_alignment="bottom")

    with c1:
        task_type = st.selectbox(
            "Task",
            ["classification", "regression"],
            key="ml_task",
        )

    with c2:
        hyperparameter_tuning = st.checkbox(
            "Hyperparameter Tuning",
            value=default_config.get("hyperparameter_tuning", True),
            key="ml_hyperparameter_tuning",
        )

    with c3:
        eval_seeds_input = st.text_input(
            label="Seeds",
            value="42",
            help="Comma-separated random seeds used for multi-seed evaluation.",
            key="ml_eval_seeds",
        )
        eval_seeds = parse_eval_seeds(eval_seeds_input)


    models, skipped_models = build_models_config(
        selected_models=selected_models,
        task_type=task_type,
        seeds=eval_seeds,
        hyperparameter_tuning=hyperparameter_tuning,
    )

    if skipped_models:
        st.warning("Skipped incompatible model(s): " + ", ".join(skipped_models))

    st.session_state["ml_models_config"] = models

    c1, c2 = st.columns(2, vertical_alignment="bottom")

    with c1:
        if st.button(
            "Show Model Information",
            use_container_width=True,
            type="primary",
            key="show_model_info_button",
        ):
            st.session_state["show_model_info"] = True

    with c2:
        if st.button(
            "Hide Model Information",
            use_container_width=True,
            type="primary",
            key="hide_model_info_button",
        ):
            st.session_state["show_model_info"] = False


    if st.session_state["show_model_info"]:
        st.subheader("Model Summary")

        model_df = models_to_dataframe(models, hyperparameter_tuning=hyperparameter_tuning)

        st.dataframe(
            model_df,
            use_container_width=True,
            hide_index=True,
        )

        if hyperparameter_tuning:
            st.subheader("Hyperparameter Search Space")

            hp_df = hyperparameters_to_dataframe(models, hyperparameter_tuning=hyperparameter_tuning)

            st.dataframe(
                hp_df,
                use_container_width=True,
                hide_index=True,
            )


    return models, task_type