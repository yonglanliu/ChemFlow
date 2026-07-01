# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import torch
from transformers import AutoTokenizer

from src.config import PROJECT_ROOT
from src.utils.style import load_css as inject_css
from src.gpt.model import GPT
from src.gpt.generator import generate


# ============================================================
# UI Helpers
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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def remove_condition_tokens(
    text: str,
    condition_tokens: list[str] | None,
) -> str:
    if not condition_tokens:
        return text

    for token in condition_tokens:
        text = text.replace(token, "")

    return text.strip()


def build_input_ids(
    tokenizer,
    prompt: str | None,
    include_bos: bool,
    device: torch.device,
) -> torch.Tensor:
    prompt = "" if prompt is None else str(prompt).strip()

    ids_list: list[int] = []

    if include_bos and tokenizer.bos_token_id is not None:
        ids_list.append(tokenizer.bos_token_id)

    if prompt:
        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        ids_list.extend(prompt_ids)

    if not ids_list:
        if tokenizer.bos_token_id is None:
            raise ValueError(
                "Empty prompt and tokenizer has no BOS token. "
                "Please enter a prompt or define a BOS token."
            )
        ids_list = [tokenizer.bos_token_id]

    return torch.tensor(
        [ids_list],
        dtype=torch.long,
        device=device,
    )


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_gpt_generator_model(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    device = get_device()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'config'.")

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    config = checkpoint["config"]

    if "GPTConfig" not in config:
        raise KeyError("Checkpoint config does not contain 'GPTConfig'.")

    gpt_config = SimpleNamespace(**config["GPTConfig"])

    resolved_config = config.get("ResolvedConfig", {})
    tokenizer_dir = resolved_config.get("tokenizer_dir", None)

    if tokenizer_dir is not None:
        tokenizer_path = Path(tokenizer_dir).expanduser().resolve()
    else:
        workdir = Path(config["GPTTrainingConfig"]["workdir"]).expanduser().resolve()
        tokenizer_path = workdir / "tokenizer"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=False,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    vocab_size = len(tokenizer)

    if vocab_size <= 0:
        raise ValueError(
            f"Tokenizer vocabulary size is {vocab_size}. "
            f"Please check tokenizer path: {tokenizer_path}"
        )

    gpt_config.vocab_size = vocab_size
    gpt_config.pad_token_id = tokenizer.pad_token_id
    gpt_config.bos_token_id = tokenizer.bos_token_id
    gpt_config.eos_token_id = tokenizer.eos_token_id

    model = GPT(
        vocab_size=gpt_config.vocab_size,
        max_len=gpt_config.max_len,
        d_model=gpt_config.d_model,
        n_heads=gpt_config.n_heads,
        n_layers=gpt_config.n_layers,
        d_ff=gpt_config.d_ff,
        dropout=gpt_config.dropout,
        pad_token_id=gpt_config.pad_token_id,
        bos_token_id=getattr(gpt_config, "bos_token_id", None),
        eos_token_id=getattr(gpt_config, "eos_token_id", None),
        use_quant_noise=getattr(gpt_config, "use_quant_noise", False),
        quant_noise_p=getattr(gpt_config, "quant_noise_p", 0.0),
        quant_noise_block_size=getattr(
            gpt_config,
            "quant_noise_block_size",
            8,
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    condition_tokens = (
        config.get("TokenizerConfig", {})
        .get("condition_tokens", [])
    )

    return model, tokenizer, device, config, condition_tokens, tokenizer_path


# ============================================================
# Page Setup
# ============================================================

inject_css()

st.markdown(
    """
    <div class="page-title">
        GPT Molecular Generator
    </div>
    """,
    unsafe_allow_html=True,
)

divider()

section_title("Select GPT Generator Model")

selected_model = st.text_input(
    "Model checkpoint path",
    value=str(PROJECT_ROOT / "outputs" / "gpt" / "checkpoints" / "best_model.pt"),
    placeholder="Enter the path to the GPT generator checkpoint.",
    help="Path to best_model.pt or last_model.pt.",
)

num_molecules = st.number_input(
    "Number of molecules",
    min_value=1,
    max_value=100,
    value=20,
    step=1,
)

max_new_tokens = st.slider(
    "Maximum new tokens",
    min_value=16,
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
    max_value=200,
    value=20,
    step=1,
)

divider()

section_title("Condition Prompt")

condition_prompt = st.text_input(
    "Prompt / condition token",
    value="",
    placeholder="Optional, e.g. <PI3K_ALPHA>",
    help="Leave empty for unconditional generation.",
)

include_bos = st.checkbox(
    "Prepend BOS token",
    value=True,
)

remove_prompt_from_output = st.checkbox(
    "Remove condition token from displayed SMILES",
    value=True,
)

skip_special_tokens = st.checkbox(
    "Skip special tokens during decoding",
    value=True,
)

divider()

if st.button("Generate Molecules"):
    checkpoint_path = Path(selected_model).expanduser().resolve()

    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()

    try:
        (
            model,
            tokenizer,
            device,
            config,
            condition_tokens,
            tokenizer_path,
        ) = load_gpt_generator_model(str(checkpoint_path))

    except Exception as e:
        st.error("Failed to load GPT generator model.")
        st.exception(e)
        st.stop()

    st.success(f"Loaded model on device: {device}")

    with st.expander("Model config", expanded=False):
        st.json(config.get("GPTConfig", {}))

    with st.expander("Tokenizer info", expanded=False):
        st.write("Tokenizer path:", tokenizer_path)
        st.write("Vocabulary size:", tokenizer.vocab_size)
        st.write("Length with added tokens:", len(tokenizer))
        st.write("PAD:", tokenizer.pad_token, tokenizer.pad_token_id)
        st.write("BOS:", tokenizer.bos_token, tokenizer.bos_token_id)
        st.write("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
        st.write("Condition tokens:", condition_tokens)

    generated_smiles = []

    progress = st.progress(0)

    with st.spinner("Generating molecules..."):
        with torch.no_grad():
            for i in range(num_molecules):
                input_ids = build_input_ids(
                    tokenizer=tokenizer,
                    prompt=condition_prompt,
                    include_bos=include_bos,
                    device=device,
                )

                generated_ids = generate(
                    model=model,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=tokenizer.eos_token_id,
                )

                text = tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=skip_special_tokens,
                )

                if remove_prompt_from_output:
                    text = remove_condition_tokens(text, condition_tokens)

                    for special_token in [
                        tokenizer.bos_token,
                        tokenizer.eos_token,
                        tokenizer.pad_token,
                        tokenizer.unk_token,
                    ]:
                        if special_token:
                            text = text.replace(special_token, "")

                text = text.strip()
                generated_smiles.append(text)

                progress.progress((i + 1) / num_molecules)

    st.markdown("### Generated Molecules")

    for i, smiles in enumerate(generated_smiles, start=1):
        st.code(f"{i:02d}: {smiles}", language="text")

    st.download_button(
        label="Download SMILES",
        data="\n".join(generated_smiles),
        file_name="generated_smiles.smi",
        mime="text/plain",
    )