# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


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

        rows.append({"Parameter": str(key), "Value": value})

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
    subprocess_log_file = run_dir / "subprocess.log"

    status = read_json(status_file)

    if status is None:
        st.info("No training status found yet.")

        if subprocess_log_file.exists():
            st.subheader("Subprocess Log")
            st.code(subprocess_log_file.read_text(encoding="utf-8"))

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

    with st.expander("Raw status.json", expanded=False):
        st.json(status)

    summary_files = sorted(run_dir.glob("*summary*.json"))
    csv_files = sorted(run_dir.glob("*summary*.csv"))

    if summary_files:
        st.subheader("Summary JSON Files")
        for path in summary_files:
            with st.expander(path.name, expanded=False):
                st.json(read_json(path))

    if csv_files:
        st.subheader("Summary CSV Files")
        for path in csv_files:
            st.write(path.name)
            st.dataframe(pd.read_csv(path), use_container_width=True)

    if log_file.exists():
        st.subheader("Training Log")
        st.code(log_file.read_text(encoding="utf-8"))

    if subprocess_log_file.exists():
        st.subheader("Subprocess Log")
        st.code(subprocess_log_file.read_text(encoding="utf-8"))


def design():
    from src.streamlit.utils.select_dir import directory_picker
    import src.streamlit.pages.Machine_Learning.tabs.tab1_utils.model_selection as model_selection
    from src.streamlit.utils.select_file import file_picker
    from src.streamlit.pages.Machine_Learning.tabs.tab1_utils import featurization
    from src.streamlit.pages.Machine_Learning.tabs.tab1_utils import data_split

    workdir = directory_picker(
        label="Select Working Directory",
        start_dir="./",
        key_prefix="traditional_ml_workdir",
    )

    if workdir is None:
        st.info("Please select a working directory.")
        return None

    workdir = Path(workdir)

    st.divider()

    models, task_type = model_selection.design()
    task_type = str(task_type).lower()

    st.divider()

    data_file = file_picker(
        start_dir=workdir,
        key_prefix="input_training_data",
        allowed_extensions=(
            ".csv",
            ".tsv",
            ".json",
            ".pkl",
            ".pickle",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".parquet",
            ".feather",
        ),
    )

    data = {}
    features = {}

    if data_file:
        data, features = featurization.design(
            data_file=data_file,
            workdir=workdir,
            task_type=task_type,
        )
    else:
        st.info("Please select a data file first.")

    st.divider()

    st.subheader("Dataset Splitting")
    split_config = data_split.design(workdir, task_type)

    training_config = {
        "models": models,
        "task_type": task_type,
        "hyperparameter_tuning": st.session_state.get(
            "ml_hyperparameter_tuning",
            True,
        ),
        "data": data,
        "featurization": features,
        "data_split": split_config,
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

            st.success(f"Saved config to: {config_path}")

    st.divider()

    if "current_run_dir" not in st.session_state:
        st.session_state["current_run_dir"] = None

    if "training_pid" not in st.session_state:
        st.session_state["training_pid"] = None

    if st.button(
        "Train Selected Models",
        type="primary",
        use_container_width=True,
        key="ml_train_models",
    ):
        if not models:
            st.warning("Please select at least one model.")
            return workdir

        if not data_file:
            st.warning("Please select a training data file.")
            return workdir

        if not features:
            st.warning("Please configure molecular representation.")
            return workdir

        if not data:
            st.warning("Please configure data settings.")
            return workdir

        if not split_config:
            st.warning("Please configure dataset splitting.")
            return workdir

        run_id, run_dir = create_run_dir(workdir)

        run_training_config = {
            "models": models,
            "hyperparameter_tuning": st.session_state.get(
                "ml_hyperparameter_tuning",
                True,
            ),
            "task_type": task_type,
            "data": data,
            "featurization": features,
            "data_split": split_config,
            "workdir": str(run_dir),
        }

        run_training_config_safe = make_json_safe(run_training_config)

        config_path = run_dir / "config.json"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(run_training_config_safe, f, indent=4, default=str)

        subprocess_log_path = run_dir / "subprocess.log"

        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path.cwd() / "src")

        with open(subprocess_log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "chemflow.machine_learning.train.train_runner",
                    str(config_path),
                ],
                stdout=log_f,
                stderr=log_f,
                cwd=Path.cwd(),
                env=env,
            )

        st.session_state["current_run_dir"] = str(run_dir)
        st.session_state["training_pid"] = process.pid

        st.success(f"Training started: {run_id}")
        st.info(f"PID: {process.pid}")
        st.info(f"Config saved to: {config_path}")
        st.info(f"Subprocess log: {subprocess_log_path}")

    st.divider()

    st.subheader("Training Status")

    if st.session_state["current_run_dir"]:
        run_dir = Path(st.session_state["current_run_dir"])

        st.write(f"Current run directory: `{run_dir}`")

        if st.session_state.get("training_pid"):
            st.write(f"Training PID: `{st.session_state['training_pid']}`")

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