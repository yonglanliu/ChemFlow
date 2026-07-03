
from dataclasses import dataclass
from typing import List


@dataclass
class GenerationConfig:

    temperature: float = 0.8
    top_k: int = 50
    max_length: int = 128
    num_samples: int = 100
    max_generation_length: int = 100
    


@dataclass
class TokenizerConfig:

    tokenizer_name: str = "seyonec/ChemBERTa_zinc250k_v2_40k"

    max_length: int = 128

    condition_tokens: List[str] = None


@dataclass
class LSTMConfig:
    vocab_size: int = 10000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2



@dataclass
class TrainingConfig:
    workdir: str = "./"
    dataset_path: str = "data/dataset.parquet"
    cache_dir: str = "cache"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "tensorboard"
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    gradient_clip_value: float = 1.0
    device: str = "cuda"  # Options: "cuda", "cpu"
    scheduler: str = "linear"  # Options: "linear", "cosine", "exponential"
    plateau_patience: int = 5
    val_train_split: float = 0.1  # Fraction of data to use for validation
    training_X: str = "train_smiles"  # Options: "train", "val", "test"
    validation_X: str = "val_smiles"  # Options: "train", "val", "test"
    training_y: str = "train_labels"  # Options: "train", "val", "test"
    validation_y: str = "val_labels"  # Options: "train", "val", "test"
    test_X: str = "test_smiles"  # Options: "train", "val", "test"
    test_y: str = "test_labels"  # Options: "train", "val", "test"
    fine_tune: bool = False  # Whether to fine-tune the model or train from scratch
    num_workers: int = 4  # Number of workers for data loading
    pretrained_ckpt_path: str = "checkpoints/best_model.pt"  # Path to the pre-trained model for fine-tuning
    early_stop_patience: int = 20  # Number of epochs with no improvement after which training will be stopped