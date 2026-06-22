import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st


def create_run_dir(workdir):
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(workdir) / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def show_training_status(run_dir):
    status_file = run_dir / "status.json"
    log_file = run_dir / "training.log"

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

    if log_file.exists():
        st.subheader("Training Log")
        st.code(log_file.read_text(encoding="utf-8"))


# Example variables from your app
workdir = "./"

training_config = {
    "model": "Random Forest",
    "task_type": "Regression",
    "data_file": "your_data.csv",
    "smiles_col": "smiles",
    "y_col": "pIC50",
    "test_size": 0.2,
    "random_seed": 42,
}

if "current_run_dir" not in st.session_state:
    st.session_state["current_run_dir"] = None


if st.button(
    "Train Selected Models",
    type="primary",
    use_container_width=True,
    key="ml_train_models",
):
    if not training_config.get("model"):
        st.warning("Please select at least one model.")
        st.stop()

    run_id, run_dir = create_run_dir(workdir)

    config_path = run_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=4, default=str)

    subprocess.Popen(
        [
            sys.executable,
            "src/train.py",
            str(config_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    st.session_state["current_run_dir"] = str(run_dir)

    st.success(f"Training started: {run_id}")


if st.session_state["current_run_dir"]:
    run_dir = Path(st.session_state["current_run_dir"])

    if st.button("Refresh Training Status", use_container_width=True):
        st.rerun()

    show_training_status(run_dir)