from typing import Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import II

CURRENT_PATH = Path(__file__).resolve()

@dataclass
class GraphormerPretrainedConfig:
    """
    Configuration for pretrained Graphormer models.
    """
    pretrained_path: str | Path = field(
        default=CURRENT_PATH.parent / "pretrained" / "graphormer-base-pcqm4mv1.pt",
        metadata={"help": "Path to the pretrained model file."},
        )
    
    max_nodes: int = field(
        default=128,
        metadata={"help": "Maximum number of nodes in the graph."},
    )

    spatial_pos_max: int = field(
        default=1024,
        metadata={"help": "Maximum spatial position for the model."},
    )
    
    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "Maximum distance for multi-hop edges."},
    )
    #-----------------------------------
    # GraphormerGraphEncoder parameters
    #-----------------------------------
    num_atoms: int = field(
        default=512 * 9,
        metadata={"help": "Number of atom types in the graph."},
    )
    num_in_degree: int = field(
        default=512,
        metadata={"help": "Number of in-degree types in the graph."},
    )
    num_out_degree: int = field(
        default=512,
        metadata={"help": "Number of out-degree types in the graph."},
    )
    num_edges: int = field(
        default=512 * 3,
        metadata={"help": "Number of edge types in the graph."},
    )
    num_spatial: int = field(
        default=512,
        metadata={"help": "Number of spatial types in the graph."},
    )
    num_edge_dis: int = field(
        default=128,
        metadata={"help": "Number of edge distance types in the graph."},
    )
    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "Edge type for the graph."},
    )
    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "Maximum distance for multi-hop edges."},
    )
    num_encoder_layers: int = field(
        default=12,
        metadata={"help": "Number of encoder layers in the model."},
    )
    encoder_embed_dim: int = field(
        default=768,
        metadata={"help": "Embedding dimension for the encoder."},
    )
    ffn_embedding_dim: int = field(
        default=768,
        metadata={"help": "Feedforward network embedding dimension for the encoder."},
    )
    encoder_attention_heads: int = field(
        default=32,
        metadata={"help": "Number of attention heads in the encoder."},
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for the model."},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for attention weights."},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability for activation functions."},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "LayerDrop probability for the encoder."},
    )
    encoder_normalize_before: bool = field(
        default=True,
        metadata={"help": "Whether to apply layer normalization before the encoder."},
    )
    pre_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether to apply pre-layer normalization."},
    )
    apply_graphormer_init: bool = field(
        default=False,
        metadata={"help": "Whether to apply Graphormer-specific parameter initialization."},
    )
    activation_fn: str = field(
        default="gelu",
        metadata={"help": "Activation function for the model."},
    )
    embed_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Scale factor for the embeddings."},
    )
    freeze_layer_indices: Optional[list[int]] = field(
        default=None,
        metadata={"help": "Indices of layers to freeze during training."},
    )
    traceable: bool = field(
        default=False,
        metadata={"help": "Whether to make the model traceable for TorchScript."},
    )
    last_state_only: bool = field(
        default=False,
        metadata={"help": "Whether to return only the last state from the encoder."},
    )
    use_quant_noise: bool = field(
        default=False,
        metadata={"help": "Whether to use quantization noise for the model."},
    )
    q_noise: float = field(
        default=0.0,
        metadata={"help": "Quantization noise probability."},
    )
    qn_block_size: int = field(
        default=8,
        metadata={"help": "Block size for quantization noise."},
    )

    # ------------------------------------------

@dataclass
class LoraConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation) in Graphormer models.
    """
    lora_target: str = field(
        default="attention",
        metadata={"help": "Target for LoRA application (e.g., 'attention', 'ffn', or 'all')."},
    )
    lora_r: int = field(
        default=4,
        metadata={"help": "Rank for LoRA."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Alpha for LoRA."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA."},
    )
    apply_lora_to_k_proj: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA to the key projection."},
    )
    lora_ffn_r: int = field(
        default=4,
        metadata={"help": "Rank for LoRA in FFN."},
    )
    lora_ffn_alpha: int = field(
        default=16,
        metadata={"help": "Alpha for LoRA in FFN."},
    )
    lora_use_fc2: bool = field(
        default=False,
        metadata={"help": "Whether to use the second fully connected layer in LoRA for FFN."},
    )

@dataclass
class GraphormerFinetuneRegressionConfig(GraphormerPretrainedConfig, LoraConfig):
    """
    Configuration for fine-tuning Graphormer models.
    """
    head_hidden_dim: int = field(
        default=768,
        metadata={"help": "Hidden size for the regression head."},
    )

    head_intermediate_dim: int = field(
        default=256,
        metadata={"help": "Intermediate size for the regression head."},
    )

    head_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for the regression head."},
    )
    freeze_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the encoder during fine-tuning."},
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for fine-tuning."},
    )

@dataclass
class GraphormerFinetuneClassificationConfig(GraphormerPretrainedConfig, LoraConfig):
    """
    Configuration for fine-tuning Graphormer models for classification tasks.
    """
    num_classes: int = field(
        default=2,
        metadata={"help": "Number of classes for classification."},
    )
    head_hidden_dim: int = field(
        default=768,
        metadata={"help": "Hidden size for the classification head."},
    )

    head_intermediate_dim: int = field(
        default=256,
        metadata={"help": "Intermediate size for the classification head."},
    )

    head_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for the classification head."},
    )
    freeze_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the encoder during fine-tuning."},
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for fine-tuning."},
    )
    loss_type: str = field(
        default="cross_entropy",
        metadata={"help": "Loss function: cross_entropy or bce."},
    )
    class_weights: Optional[list[float]] = field(
        default=None,
        metadata={"help": "Class weights for imbalanced classification."},
    )
    positive_weight: Optional[float] = field(
        default=None,
        metadata={"help": "Positive class weight for BCE."},
    )