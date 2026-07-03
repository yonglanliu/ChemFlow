from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CommonConfig:
    seed: int = 42
    max_nodes: int = 64
    multi_hop_max_dist: int = 5
    spatial_pos_max: int = 1024
    hidden_dim: int = 768


@dataclass
class DatasetConfig:
    dataset_path: str | Path | None = None
    smiles_column: str | None = "SMILES"

    training_X: str = "train_smiles"
    validation_X: str = "val_smiles"
    training_y: str = "train_labels"
    validation_y: str = "val_labels"

    val_fraction: float = 0.1
    preprocess_batch_size: int = 100_000


@dataclass
class FeaturizerConfig:
    remove_hs: bool = True
    reorder_atoms: bool = False


@dataclass
class GraphormerEncoderConfig:
    num_atoms: int = 16 * 512
    num_edges: int = 16 * 512
    num_in_degree: int = 512
    num_out_degree: int = 512
    num_edge_dis: int = 128

    edge_type: str = "multi_hop"

    num_encoder_layers: int = 12
    ffn_embedding_dim: int = 768
    num_attention_heads: int = 32

    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    layerdrop: float = 0.0

    encoder_normalize_before: bool = False
    pre_layernorm: bool = False
    apply_graphormer_init: bool = True
    activation_fn: str = "gelu"

    embed_scale: Optional[float] = None
    freeze_layer_indices: Optional[list[int]] = None
    traceable: bool = False
    last_state_only: bool = True

    use_quant_noise: bool = False
    q_noise: float = 0.0
    qn_block_size: int = 8


@dataclass
class GraphormerDenoiserConfig:
    num_atom_types: int = 120
    num_bond_types: int = 6
    dropout: float = 0.1
    bond_pair_mode: str = "sum"


@dataclass
class GraphormerDiffusionConfig:
    num_timesteps: int = 50

    atom_mask_token: int = 120
    bond_mask_token: int = 6
    atom_pad_token: int = 0
    bond_pad_token: int = 0

    atom_loss_weight: float = 1.0
    bond_loss_weight: float = 1.0


@dataclass
class GraphormerDiffusionTrainingConfig:
    workdir: str | Path = "./workdir"

    batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 50

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"  # "none", "linear", "cosine", "exponential", "plateau"

    grad_clip_norm: float = 1.0
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    num_workers: int = 0

    resume: bool = False
    resume_checkpoint: str | Path | None = None

    plot_training_history: bool = True


@dataclass
class FullConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    featurizer: FeaturizerConfig = field(default_factory=FeaturizerConfig)
    encoder: GraphormerEncoderConfig = field(default_factory=GraphormerEncoderConfig)
    denoiser: GraphormerDenoiserConfig = field(default_factory=GraphormerDenoiserConfig)
    diffusion: GraphormerDiffusionConfig = field(default_factory=GraphormerDiffusionConfig)
    training: GraphormerDiffusionTrainingConfig = field(default_factory=GraphormerDiffusionTrainingConfig)