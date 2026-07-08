# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class GPTConfig:
    # Vocabulary
    vocab_size: int

    # Sequence
    max_len: int = 128
    
    # Model
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024

    # Regularization
    dropout: float = 0.1

    # Special tokens
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    # Quantization Noise
    use_quant_noise: bool = False
    quant_noise_p: float = 0.0
    quant_noise_block_size: int = 8

@dataclass
class GPTTrainingConfig:
    # Training
    workdir: str | Path | None = None
    smiles_column: str | None = None
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    seed: int = 42
    early_stopping: bool = True
    early_stopping_patience: int = 5
    scheduler: str | None = None
    gradient_clip_value: float = 1.0
    num_workers: int = 4
    plot_training_history: bool = True

@dataclass
class DatasetConfig:
    dataset_path: str | Path | None = None
    smiles_column: str | None = "SMILES"
    training_X: str = "train_smiles"  # Options: "train", "val", "test"
    validation_X: str = "val_smiles"  # Options: "train", "val", "test"
    training_y: str = "train_labels"  # Options: "train", "val", "test"
    validation_y: str = "val_labels"  # Options: "train", "val", "test"
    val_fraction: float = 0.1  # Fraction of data to use for validation
    max_length: int = 128
    preprocess_batch_size: int = 100_000
    seed: int = 42


@dataclass
class TokenizerConfig:

    tokenizer_name: str = "seyonec/ChemBERTa_zinc250k_v2_40k"

    max_length: int = 128

    condition_tokens: List[str] | None = None #["<PI3K_ALPHA>", "<PI3K_BETA>","<HIGH_PIC50>"]

@dataclass
class GPTGenerationConfig:
    # Generation
    max_gen_len: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
