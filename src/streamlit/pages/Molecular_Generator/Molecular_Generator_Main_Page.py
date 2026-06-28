# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer

from src.config import PROJECT_ROOT
from src.utils.style import load_css as inject_css
from src.chemflow.machine_learning.llm.rnn import SmilesLSTMGenerator
from src.chemflow.machine_learning.generator.lstm_generator import generate


# ============================================================
# Helpers
# ============================================================

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


@st.cache_resource
def load_generator_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    cfg = checkpoint["config"]

    model = SmilesLSTMGenerator(**cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


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

models = {
    "Unconditional LSTM Generator": PROJECT_ROOT
    / "checkpoints"
    / "smiles_lstm_best.pt"
}

selected_model_name = st.selectbox(
    "Generator model",
    options=list(models.keys()),
    label_visibility="collapsed",
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
    checkpoint_path = models[selected_model_name]

    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()

    tokenizer = load_tokenizer()
    model, device = load_generator_model(str(checkpoint_path))

    st.markdown("### Generated Molecules")

    generated_smiles = []

    with st.spinner("Generating molecules..."):
        with torch.no_grad():
            for i in range(num_molecules):
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