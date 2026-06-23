# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from src.utils.design import temp_success, temp_info

def make_json_safe(value):
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "n_rows": len(value),
            "n_columns": len(value.columns),
            "columns": value.columns.tolist(),
        }

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def config_to_dataframe(config):
    rows = []

    safe_config = make_json_safe(config)

    for key, value in safe_config.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value, indent=2, default=str)
        else:
            value = str(value)

        rows.append(
            {
                "Parameter": str(key),
                "Value": value,
            }
        )

    return pd.DataFrame(rows)


def create_run_dir(workdir):
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(workdir) / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def read_json(path):
    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def show_training_status(run_dir):
    run_dir = Path(run_dir)

    status_file = run_dir / "status.json"
    log_file = run_dir / "training.log"
    metrics_file = run_dir / "metrics.json"

    status = read_json(status_file)

    if status is None:
        st.info("No training status found yet.")
        return

    current_status = status.get("status", "unknown")
    progress = status.get("progress", 0)

    if current_status == "running":
        st.info("Training is running.")
        st.progress(progress / 100)

    elif current_status == "completed":
        st.success("Training completed.")
        st.progress(1.0)

    elif current_status == "failed":
        st.error("Training failed.")
        st.write(status.get("error", ""))

    else:
        st.warning(f"Unknown status: {current_status}")

    st.write(status)

    if metrics_file.exists():
        st.subheader("Metrics")
        st.json(read_json(metrics_file), expanded=False)

    if log_file.exists():
        st.subheader("Training Log")
        st.code(log_file.read_text(encoding="utf-8"))

def design():
    from src.utils.select_dir import directory_picker

    workdir = directory_picker(
        label="Select Working Directory",
        start_dir="./",
        key_prefix="traditional_ml_workdir",
    )

    if workdir is None:
        st.info("Please select a working directory.")
        return

    workdir = Path(workdir)

    st.divider()

    import src.streamlit.tabs.page_machine_learning.model_selection as model_selection

    models, task_type, search_seed, hyperparameter_tuning = model_selection.design(
        workdir
    )

    task_type = str(task_type).lower()

    st.divider()

    st.subheader("Molecular Representation")

    from src.utils.select_file import file_picker
    import src.streamlit.tabs.page_machine_learning.molecular_representation as mol_rep

    data_file = file_picker(
        start_dir=workdir,
        key_prefix="input_training_data",
        allowed_extensions=(
            ".csv",
            ".tsv",
            ".xlsx",
            ".xls",
            ".json",
            ".pkl",
            ".pickle",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".txt",
            ".parquet",
            ".feather",
        ),
    )

    mol_config = None

    if data_file:
        mol_config = mol_rep.design(
            data_file=data_file,
            workdir=workdir,
            task_type=task_type,
        )
        # st.text(f'n_classes: mol_config["n_classes"]') #debug
    else:
        temp_info("Please select a data file first.")

    st.divider()

    from src.streamlit.tabs.page_machine_learning import data_split

    st.subheader("Dataset Splitting")

    split_method, test_size, validation_size, split_config = data_split.design(
        task_type
    )

    training_config = {
        "models": models,
        "task": task_type,
        "task_type": task_type,
        "molecular_representation": mol_config,
        "data_file": str(data_file) if data_file else None,
        "split_method": split_method,
        "test_size": test_size,
        "validation_size": validation_size,
        **split_config,
        "search_seed": int(search_seed),
        "hyperparameter_tuning": bool(hyperparameter_tuning),
        "workdir": str(workdir),
    }

    training_config_safe = make_json_safe(training_config)

    st.subheader("Training Configuration Preview")

    config_df = config_to_dataframe(training_config_safe)

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True,
    )

    json_data = json.dumps(
        training_config_safe,
        indent=4,
        default=str,
    )

    with st.expander("View JSON", expanded=False):
        st.code(json_data, language="json")

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="Download JSON Config",
            data=json_data,
            file_name="training_config.json",
            mime="application/json",
            use_container_width=True,
            key="ml_download_json_config",
        )

    with c2:
        if st.button(
            "Save JSON Config",
            type="primary",
            use_container_width=True,
            key="ml_save_json_config",
        ):
            config_path = workdir / "config" / "training_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(training_config_safe, f, indent=4, default=str)

            temp_success(f"Saved config to {config_path}")

    st.divider()

    if "current_run_dir" not in st.session_state:
        st.session_state["current_run_dir"] = None

    if st.button(
        "Train Selected Models",
        type="primary",
        use_container_width=True,
        key="ml_train_models",
    ):
        if not models:
            st.warning("Please select at least one model.")
            return

        if mol_config is None:
            st.warning("Please select a data file and configure molecular representation.")
            return

        if mol_config.get("training_data_file") is None:
            st.warning("Training data file was not created. Please finish target/class setup.")
            return

        run_id, run_dir = create_run_dir(workdir)

        config_path = run_dir / "config.json"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(training_config_safe, f, indent=4, default=str)

        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.train.train_runner",
                str(config_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        st.session_state["current_run_dir"] = str(run_dir)

        st.success(f"Training started: {run_id}")

    st.divider()

    st.subheader("Training Status")

    if st.session_state["current_run_dir"]:
        run_dir = Path(st.session_state["current_run_dir"])

        if st.button(
            "Refresh Training Status",
            use_container_width=True,
            key="refresh_training_status",
        ):
            st.rerun()

        show_training_status(run_dir)

    else:
        st.info("Training has not started.")
    return workdir