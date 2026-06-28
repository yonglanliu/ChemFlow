# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import toml

from src.utils.style import load_css as inject_css
from src.streamlit.utils.select_dir import directory_picker
from src.streamlit.utils.select_file import file_picker

from src.chemflow.machine_learning.configs.training import LLMTrainingConfig
from src.chemflow.machine_learning.configs.tokenizer import TokenizerConfig
from src.chemflow.machine_learning.configs.models import LSTMConfig
from src.chemflow.machine_learning.configs.generation import GenerationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def divider() -> None:
    st.markdown(
        """
        <div style="
            height:3px;
            width:100%;
            background:linear-gradient(90deg,#3b82f6,#06b6d4,#10b981);
            border-radius:10px;
            margin:20px 0;
        "></div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataclass_to_dict(obj: Any) -> dict:
    return dict(obj.__dict__)


def update_training_paths(work_dir: str | Path) -> None:
    work_dir = Path(work_dir)

    st.session_state["training_config"]["workdir"] = str(work_dir)
    st.session_state["training_config"]["cache_dir"] = str(work_dir / "cache")
    st.session_state["training_config"]["checkpoint_dir"] = str(work_dir / "checkpoints")
    st.session_state["training_config"]["tensorboard_dir"] = str(work_dir / "tensorboard")
    st.session_state["training_config"]["cached_data_path"] = str(
        work_dir / "cache" / "cached_data.pt"
    )
    st.session_state["training_config"]["pretrained_ckpt_path"] = str(
        work_dir / "checkpoints" / "best_model.pt"
    )


def initialize_session_state() -> None:
    if "training_config" not in st.session_state:
        st.session_state["training_config"] = dataclass_to_dict(LLMTrainingConfig())

    if "tokenizer_config" not in st.session_state:
        st.session_state["tokenizer_config"] = dataclass_to_dict(TokenizerConfig())

    if "model_config" not in st.session_state:
        st.session_state["model_config"] = {}

    if "generation_config" not in st.session_state:
        st.session_state["generation_config"] = dataclass_to_dict(GenerationConfig())

    if "generative_model_type" not in st.session_state:
        st.session_state["generative_model_type"] = "LSTM"

    if "config" not in st.session_state:
        st.session_state["config"] = {}


def build_final_config() -> dict:
    return {
        "generative_model_type": st.session_state["generative_model_type"],
        "training_config": st.session_state["training_config"],
        "tokenizer_config": st.session_state["tokenizer_config"],
        "model_config": st.session_state["model_config"],
        "generation_config": st.session_state["generation_config"],
    }


inject_css()
initialize_session_state()


st.markdown(
    """
    <div class="page-title">
        Setup For Generative Model Training
    </div>
    """,
    unsafe_allow_html=True,
)

divider()


# ============================================================
# Working Directory
# ============================================================

section_title("Select Working Directory")

with st.expander("Working Directory", expanded=True):
    workdir = directory_picker(
        label="",
        start_dir=PROJECT_ROOT,
        key_prefix="generative_model_workdir",
    )

    if workdir:
        update_training_paths(workdir)
    else:
        current_work_dir = st.session_state["training_config"].get(
            "workdir",
            str(PROJECT_ROOT),
        )
        update_training_paths(current_work_dir)

    st.write("Current working directory:")
    st.code(st.session_state["training_config"]["workdir"])

divider()


# ============================================================
# Generative Model Type
# ============================================================

section_title("Which Generative Model To Train?")

model_options = ["LSTM", "GPT", "GPT2", "VAE", "Diffusion"]
current_model_type = st.session_state.get("generative_model_type", "LSTM")

model_type = st.selectbox(
    label="",
    options=model_options,
    index=model_options.index(current_model_type)
    if current_model_type in model_options
    else 0,
    key="generative_model_type_selectbox",
    help="Select the type of generative model you want to train.",
    label_visibility="collapsed",
)

st.session_state["generative_model_type"] = model_type

divider()


_, mid, _ = st.columns(3)
with mid:
    section_title("Configuration Parameters")


# ============================================================
# Dataset Configuration
# ============================================================

with st.expander("Select Dataset File", expanded=False):
    dataset_path = file_picker(
        start_dir=PROJECT_ROOT,
        allowed_extensions=[".csv", ".parquet"],
        key_prefix="generative_model_dataset",
    )

    split_method = st.selectbox(
        label="Split Method",
        options=["random", "scaffold", "stratified"],
        key="split_method",
        help="Select the method to split the dataset.",
    )

    val_train_split = st.number_input(
        label="Validation Split Ratio",
        value=float(st.session_state["training_config"].get("val_train_split", 0.1)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="val_train_split",
        help="Fraction of data to use for validation.",
    )

    if dataset_path:
        st.session_state["training_config"]["dataset_path"] = str(dataset_path)

    st.session_state["training_config"]["split_method"] = split_method
    st.session_state["training_config"]["val_train_split"] = float(val_train_split)

    st.json(
        {
            "dataset_path": st.session_state["training_config"].get("dataset_path"),
            "split_method": split_method,
            "val_train_split": val_train_split,
        }
    )


# ============================================================
# Tokenizer Configuration
# ============================================================

with st.expander("Select Tokenizer Model", expanded=False):
    tokenizer_options = [
        "seyonec/SmilesTokenizer_ChemBERTa_zinc250k_40k",
        "seyonec/ChemBERTa_zinc250k_v2_40k",
        "seyonec/ChemBERTa-zinc250k-v1",
        "seyonec/ChemBERTa-zinc-base-v1",
    ]

    current_tokenizer = st.session_state["tokenizer_config"].get(
        "tokenizer_name",
        tokenizer_options[0],
    )

    tokenizer_name = st.selectbox(
        label="Tokenizer Model",
        options=tokenizer_options,
        index=tokenizer_options.index(current_tokenizer)
        if current_tokenizer in tokenizer_options
        else 0,
        key="tokenizer_name",
        help="Select the tokenizer for the generative model.",
    )

    tokenizer_max_length = st.number_input(
        label="Tokenizer Max Length",
        min_value=1,
        value=int(st.session_state["tokenizer_config"].get("max_length", 128)),
        step=1,
        key="tokenizer_max_length",
        help="Set the maximum sequence length for tokenization.",
    )

    condition_tokens_str = st.text_input(
        label="Condition Tokens",
        value="",
        key="condition_tokens",
        help="Comma-separated condition tokens, for example: <PI3K_ALPHA>, <PI3K_BETA>",
    )

    condition_tokens = (
        [token.strip() for token in condition_tokens_str.split(",") if token.strip()]
        if condition_tokens_str.strip()
        else None
    )

    st.session_state["tokenizer_config"].update(
        {
            "tokenizer_name": tokenizer_name,
            "max_length": int(tokenizer_max_length),
            "condition_tokens": condition_tokens,
        }
    )

    st.json(st.session_state["tokenizer_config"])


# ============================================================
# Model Configuration
# ============================================================

with st.expander("Model Parameters", expanded=False):
    if st.session_state["generative_model_type"] == "LSTM":
        if not st.session_state["model_config"]:
            st.session_state["model_config"] = dataclass_to_dict(LSTMConfig())

        model_config = st.session_state["model_config"]

        embedding_dim = st.number_input(
            label="Embedding Dimension",
            min_value=1,
            value=int(model_config.get("embedding_dim", 256)),
            step=1,
            key="embedding_dim",
        )

        hidden_dim = st.number_input(
            label="Hidden Dimension",
            min_value=1,
            value=int(model_config.get("hidden_dim", 512)),
            step=1,
            key="hidden_dim",
        )

        num_layers = st.number_input(
            label="Number of Layers",
            min_value=1,
            value=int(model_config.get("num_layers", 2)),
            step=1,
            key="num_layers",
        )

        dropout = st.number_input(
            label="Dropout Rate",
            min_value=0.0,
            max_value=1.0,
            value=float(model_config.get("dropout", 0.2)),
            step=0.01,
            key="dropout",
        )

        st.session_state["model_config"].update(
            {
                "embedding_dim": int(embedding_dim),
                "hidden_dim": int(hidden_dim),
                "num_layers": int(num_layers),
                "dropout": float(dropout),
            }
        )

    else:
        st.session_state["model_config"] = {}
        st.info(
            f"{st.session_state['generative_model_type']} model parameters "
            "will be configured in future updates."
        )

    st.json(st.session_state["model_config"])


# ============================================================
# Training Configuration
# ============================================================

with st.expander("Training Parameters", expanded=False):
    training_config = st.session_state["training_config"]

    num_workers = st.number_input(
        label="Number of Workers",
        min_value=0,
        value=int(training_config.get("num_workers", 4)),
        step=1,
        key="num_workers",
    )

    batch_size = st.number_input(
        label="Batch Size",
        min_value=1,
        value=int(training_config.get("batch_size", 64)),
        step=1,
        key="batch_size",
    )

    learning_rate = st.number_input(
        label="Learning Rate",
        min_value=1e-6,
        max_value=1.0,
        value=float(training_config.get("learning_rate", 1e-3)),
        step=1e-5,
        format="%.6f",
        key="learning_rate",
    )

    reward_learning_rate = st.number_input(
        label="Reward Learning Rate",
        min_value=1e-7,
        max_value=1.0,
        value=float(training_config.get("reward_learning_rate", 1e-4)),
        step=1e-5,
        format="%.7f",
        key="reward_learning_rate",
    )

    epochs_no_reward = st.number_input(
        label="Epochs Without Reward",
        min_value=1,
        value=int(training_config.get("epochs_no_reward", 100)),
        step=1,
        key="epochs_no_reward",
    )

    epochs_with_reward = st.number_input(
        label="Epochs With Reward",
        min_value=0,
        value=int(training_config.get("epochs_with_reward", 50)),
        step=1,
        key="epochs_with_reward",
    )

    weight_decay = st.number_input(
        label="Weight Decay",
        min_value=0.0,
        max_value=1.0,
        value=float(training_config.get("weight_decay", 1e-5)),
        step=1e-5,
        format="%.6f",
        key="weight_decay",
    )

    gradient_clip = st.number_input(
        label="Gradient Clip",
        min_value=0.0,
        max_value=100.0,
        value=float(training_config.get("gradient_clip", 1.0)),
        step=0.1,
        key="gradient_clip",
    )

    scheduler_options = ["linear", "cosine", "exponential", "plateau"]
    current_scheduler = training_config.get("scheduler", "linear")

    scheduler_type = st.selectbox(
        label="Scheduler Type",
        options=scheduler_options,
        index=scheduler_options.index(current_scheduler)
        if current_scheduler in scheduler_options
        else 0,
        key="scheduler_type",
    )

    early_stop_patience = st.number_input(
        label="Early Stop Patience",
        min_value=1,
        value=int(training_config.get("early_stop_patience", 20)),
        step=1,
        key="early_stop_patience",
    )

    fine_tune = st.checkbox(
        label="Fine-tune From Pretrained Checkpoint",
        value=bool(training_config.get("fine_tune", False)),
        key="fine_tune",
    )

    pretrained_ckpt_path = st.text_input(
        label="Pretrained Checkpoint Path",
        value=str(training_config.get("pretrained_ckpt_path", "")),
        key="pretrained_ckpt_path",
    )

    st.session_state["training_config"].update(
        {
            "num_workers": int(num_workers),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "reward_learning_rate": float(reward_learning_rate),
            "epochs_no_reward": int(epochs_no_reward),
            "epochs_with_reward": int(epochs_with_reward),
            "weight_decay": float(weight_decay),
            "gradient_clip": float(gradient_clip),
            "scheduler": scheduler_type,
            "early_stop_patience": int(early_stop_patience),
            "fine_tune": bool(fine_tune),
            "pretrained_ckpt_path": pretrained_ckpt_path,
        }
    )

    st.json(st.session_state["training_config"])


# ============================================================
# Generation Configuration
# ============================================================

with st.expander("Generation Parameters", expanded=False):
    generation_config = st.session_state["generation_config"]

    generation_max_length = st.number_input(
        label="Generation Max Length",
        min_value=1,
        value=int(generation_config.get("max_length", 128)),
        step=1,
        key="generation_config_max_length",
    )

    temperature = st.number_input(
        label="Temperature",
        min_value=0.0,
        max_value=5.0,
        value=float(generation_config.get("temperature", 1.0)),
        step=0.01,
        key="generation_temperature",
    )

    top_k = st.number_input(
        label="Top-K Sampling",
        min_value=1,
        value=int(generation_config.get("top_k", 50)),
        step=1,
        key="generation_top_k",
    )

    num_samples = st.number_input(
        label="Number of Samples",
        min_value=1,
        value=int(generation_config.get("num_samples", 100)),
        step=1,
        key="generation_num_samples",
    )

    max_generation_length = st.number_input(
        label="Max Generated SMILES Length",
        min_value=1,
        value=int(generation_config.get("max_generation_length", 128)),
        step=1,
        key="generation_max_generation_length",
    )

    st.session_state["generation_config"].update(
        {
            "max_length": int(generation_max_length),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "num_samples": int(num_samples),
            "max_generation_length": int(max_generation_length),
        }
    )

    st.json(st.session_state["generation_config"])


# ============================================================
# Final Config Preview
# ============================================================

st.session_state["config"] = build_final_config()

section_title("Current Configuration")
st.json(st.session_state["config"])


# ============================================================
# Save Configuration
# ============================================================

_, center, _ = st.columns(3)

with center:
    if st.button("Save Configuration", key="save_config"):
        work_dir = Path(st.session_state["training_config"]["workdir"])
        config_path = work_dir / "config.toml"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            toml.dump(st.session_state["config"], f)

        st.success(f"Configuration saved to {config_path}")