from typing import Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import II

CURRENT_PATH = Path(__file__).resolve()

def calculate_task_weights(
    train_df: pd.DataFrame,
    endpoints: list[str] | None = None,
    method: str = "sqr_inverse",
    custom_task_weights: list[float] | None = None,
) -> dict[str, float]:
    """Calculate task weights from label availability in a training dataframe.

    Supported methods:
      - "inverse": w_i = 1 / availability_i
      - "sqrt_inverse" / "sqr_inverse": w_i = 1 / sqrt(availability_i)
      - "customed" / "customized": use a user-provided custom_task_weights list
    """
    endpoint_list = list(endpoints or [])

    missing = [name for name in endpoint_list if name not in train_df.columns]
    if missing:
        raise ValueError(
            "Missing task columns for weight calculation: "
            f"{missing}"
        )

    availability = train_df[endpoint_list].notna().mean()
    method_name = str(method).strip().lower()
    method_name = "sqrt_inverse" if method_name == "sqr_inverse" else method_name
    method_name = "customed" if method_name in {"customed", "customized"} else method_name

    if method_name == "inverse":
        weights = 1.0 / availability
    elif method_name == "sqrt_inverse":
        weights = 1.0 / np.sqrt(availability)
    elif method_name == "customed":
        if custom_task_weights is None:
            raise ValueError(
                "custom_task_weights must be provided when method is 'customed'."
            )
        if len(custom_task_weights) != len(endpoint_list):
            raise ValueError(
                "custom_task_weights length does not match the number of endpoints: "
                f"expected {len(endpoint_list)}, got {len(custom_task_weights)}."
            )
        weights = pd.Series(custom_task_weights, index=endpoint_list)
    else:
        raise ValueError(
            "method must be one of: 'inverse', 'sqrt_inverse', 'customed', or 'customized'."
        )

    weights = weights / weights.mean()
    print(f'Weights calculated using method "{method_name}": {weights.to_dict()}')
    return {key: float(value) for key, value in weights.to_dict().items()}

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

@dataclass
class GraphormerFinetuneMultitaskConfig(GraphormerPretrainedConfig, LoraConfig):
    """
    Configuration for fine-tuning Graphormer models for multi-task learning.
    """
    num_tasks: int = field(
        default=2,
        metadata={"help": "Number of tasks for multi-task learning."},
    )
    adaptor_bottleneck_dim: int = field(
        default=768,
        metadata={"help": "Hidden size for the multi-task head."},
    )

    head_intermediate_dim: int = field(
        default=256,
        metadata={"help": "Intermediate size for the multi-task head."},
    )

    adaptor_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for the multi-task head."},
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
        default="mse",
        metadata={"help": "Loss function for multi-task learning: mse or mae or huber."},
    )
    task_weights: Optional[list[float]] = field(
        default=None,
        metadata={"help": "Explicit weights for each task in multi-task learning."},
    )
    task_weight_method: str = field(
        default="sqrt_inverse",
        metadata={"help": "Task-weight generation method: 'inverse', 'sqrt_inverse', or 'customed'."},
    )
    custom_task_weights: Optional[list[float]] = field(
        default=None,
        metadata={"help": "Custom task weights used when task_weight_method is 'customed'."},
    )
    num_adapters: Optional[int] = field(
        default=None,
        metadata={"help": "Number of shared adaptors. If omitted, it defaults to one adaptor per task."},
    )
    task_groups: Optional[list[list[int]]] = field(
        default=None,
        metadata={"help": "Explicit task group assignments, where each inner list contains task indices sharing one adaptor."},
    )
