# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from src.gpt.model import GPT


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 20,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -model.max_len:]

        logits = model(input_cond)

        if isinstance(logits, tuple):
            logits = logits[0]

        logits = logits[:, -1, :]
        logits = logits / temperature

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            min_values = values[:, -1].unsqueeze(-1)

            logits = torch.where(
                logits < min_values,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

        if eos_token_id is not None:
            if torch.all(next_token.squeeze(-1) == eos_token_id):
                break

    return input_ids


def load_tokenizer(tokenizer_path: str | Path):
    tokenizer_path = Path(tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: str | torch.device | None = None,
):
    checkpoint_path = Path(checkpoint_path)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    device = torch.device(device)

    tokenizer = load_tokenizer(tokenizer_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = checkpoint["config"]

    if "GPTConfig" in config:
        gpt_config = SimpleNamespace(**config["GPTConfig"])
    else:
        gpt_config = SimpleNamespace(**config)

    vocab_size = len(tokenizer)

    if vocab_size <= 0:
        raise ValueError(
            f"Tokenizer vocabulary size is {vocab_size}. "
            f"Please check tokenizer_path: {tokenizer_path}"
        )

    gpt_config.vocab_size = vocab_size
    gpt_config.pad_token_id = tokenizer.pad_token_id
    gpt_config.bos_token_id = tokenizer.bos_token_id
    gpt_config.eos_token_id = tokenizer.eos_token_id

    print("Tokenizer check")
    print("---------------")
    print("tokenizer_path:", tokenizer_path)
    print("len(tokenizer):", len(tokenizer))
    print("tokenizer.vocab_size:", tokenizer.vocab_size)
    print("pad_token:", tokenizer.pad_token)
    print("pad_token_id:", tokenizer.pad_token_id)
    print("bos_token:", tokenizer.bos_token)
    print("bos_token_id:", tokenizer.bos_token_id)
    print("eos_token:", tokenizer.eos_token)
    print("eos_token_id:", tokenizer.eos_token_id)
    print()

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

    return model, tokenizer, gpt_config, checkpoint, device


def build_input_ids(
    tokenizer,
    prompt: str | None = None,
) -> torch.Tensor:
    if prompt is None or str(prompt).strip() == "":
        if tokenizer.bos_token_id is None:
            raise ValueError(
                "prompt is None and tokenizer.bos_token_id is also None. "
                "Please provide a prompt or define a BOS token."
            )

        return torch.tensor(
            [[tokenizer.bos_token_id]],
            dtype=torch.long,
        )

    prompt = str(prompt).strip()

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    if input_ids.numel() == 0:
        raise ValueError(
            f"Prompt produced empty token IDs: {prompt!r}. "
            "This usually means the condition token is not in the tokenizer vocabulary."
        )

    return input_ids


def generate_smiles(
    model: torch.nn.Module,
    tokenizer,
    prompt: str | None = None,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 20,
) -> str:
    input_ids = build_input_ids(
        tokenizer=tokenizer,
        prompt=prompt,
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
        skip_special_tokens=True,
    )

    return text.strip()


if __name__ == "__main__":
    checkpoint_path = "/Users/yonglanliu/Desktop/ChemFlow/gpt_training/checkpoints/best_model.pt"
    tokenizer_path = "/Users/yonglanliu/Desktop/ChemFlow/gpt_training/tokenizer"

    model, tokenizer, config, checkpoint, device = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
    )

    # None means unconditional generation.
    # You can also use something like:
    # prompt = "<PI3K_ALPHA>"
    prompt = None
    print("Device:", device)
    print("\nGenerated SMILES:")

    num_samples = 20

    for i in range(num_samples):
        smiles = generate_smiles(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=20,
        )

        print(f"{i + 1}: {smiles}")