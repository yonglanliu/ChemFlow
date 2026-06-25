# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.streamlit.utils.design import temp_success
from src.chemflow.machine_learning import MODEL_OPTIONS, build_models_config



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

def models_to_dataframe(models):
    rows = []

    for model_name, cfg in models.items():
        rows.append(
            {
                "Model": model_name,
                "Estimator": cfg["estimator"],
                "Task": cfg["task_type"],
                "Search Seed": cfg["search_seed"],
                "Multi-seed": cfg["multi_seed_evaluation"],
                "Evaluation Runs": cfg["n_runs"],
                "Evaluation Seeds": str(cfg["eval_seeds"]),
                "CV": cfg["cv"],
                "Search Method": cfg["tuning_method"],
                "Iterations": cfg["n_iter"],
                "Refit Metric": cfg["refit_metric"],
                "Metrics": ", ".join(cfg["scoring_metrics"]),
                "No. Hyperparameters": len(cfg["param_grid"]),
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
        if hyperparameter_tuning:
            search_seed = st.number_input(
                "Hyperparameter Search Seed",
                value=42,
                step=1,
                key="ml_search_seed",
            )
        else:
            search_seed = None
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
        search_seed=search_seed,
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

    return models, task_type