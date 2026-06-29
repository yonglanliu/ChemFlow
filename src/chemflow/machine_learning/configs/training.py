from dataclasses import dataclass
from pathlib import Path

@dataclass
class LLMTrainingConfig:
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
    cached_data_path: str = "cache/cached_data.pt"
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