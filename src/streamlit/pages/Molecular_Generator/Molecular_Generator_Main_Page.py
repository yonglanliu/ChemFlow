# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer

from src.utils.style import load_css as inject_css
from src.chemflow.machine_learning.llm.rnn import SmilesLSTMGenerator
from src.chemflow.machine_learning.generator.lstm_generator import generate


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


@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        "seyonec/ChemBERTa_zinc250k_v2_40k",
        use_fast=False,
    )


def build_model_config(cfg: dict) -> dict:
    """
    Keep only architecture parameters accepted by SmilesLSTMGenerator.
    Training-only parameters such as workdir, epochs, batch_size, lr, etc.
    must not be passed into the model constructor.
    """

    allowed_keys = {
        "vocab_size",
        "embedding_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
        "pad_token_id",
    }

    model_cfg = {k: v for k, v in cfg.items() if k in allowed_keys}

    if "pad_token_id" not in model_cfg:
        model_cfg["pad_token_id"] = 0

    return model_cfg


@st.cache_resource
def load_generator_model(checkpoint_path: str):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'config'.")

    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model'.")

    training_config = checkpoint["config"]

    tokenizer_path = Path(training_config["workdir"]).expanduser().resolve() / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    lstm_config = training_config.get("model_config", None)

    # If model_config was not saved inside training_config,
    # use the same architecture you trained with.
    model = SmilesLSTMGenerator(
        vocab_size=len(tokenizer),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    return model, tokenizer, device


# ============================================================
# Page Setup
# ============================================================

inject_css()

st.markdown(
    """
    <div class="page-title">
        Molecular Generator
    </div>
    """,
    unsafe_allow_html=True,
)

divider()

section_title("Select Generator Model")

selected_model = st.text_input(
    "Model Path",
    value="./checkpoints/generator_model.pt",
    placeholder="Enter the path to the generator model checkpoint.",
    help="Enter the path to a trained LSTM generator checkpoint.",
)

num_molecules = st.number_input(
    "Number of molecules",
    min_value=1,
    max_value=100,
    value=20,
    step=1,
)

max_length = st.slider(
    "Maximum SMILES length",
    min_value=32,
    max_value=256,
    value=128,
    step=16,
)

temperature = st.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=0.8,
    step=0.1,
)

top_k = st.slider(
    "Top-k sampling",
    min_value=1,
    max_value=100,
    value=20,
    step=1,
)

if st.button("Generate Molecules"):
    checkpoint_path = Path(selected_model).expanduser().resolve()

    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()

    try:
        tokenizer = load_tokenizer()
        model, tokenizer, device = load_generator_model(str(checkpoint_path))

    except Exception as e:
        st.error("Failed to load generator model.")
        st.exception(e)
        st.stop()

    st.markdown("### Generated Molecules")

    generated_smiles = []

    with st.spinner("Generating molecules..."):
        with torch.no_grad():
            for _ in range(num_molecules):
                smiles = generate(
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    device=device,
                )
                generated_smiles.append(smiles)

    for i, smiles in enumerate(generated_smiles, start=1):
        st.write(f"{i:02d}: {smiles}")

    st.download_button(
        label="Download SMILES",
        data="\n".join(generated_smiles),
        file_name="generated_smiles.smi",
        mime="text/plain",
    )