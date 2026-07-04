# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.deep_learning.graphormer.multihead_attention import MultiheadAttention
from src.deep_learning.utils.quant_noise import quant_noise
from src.deep_learning.graphormer.graphormer_layers import (
    GraphormerGraphEncoderLayer,
    GraphNodeFeature,
    GraphAttnBias,
)


def init_graphormer_params(module: nn.Module) -> None:
    """Initialize weights following Graphormer/Fairseq-style initialization."""

    def normal_(data: torch.Tensor) -> None:
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()

    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_atoms: int = 512 * 9,
        num_in_degree: int = 512,
        num_out_degree: int = 512,
        num_edges: int = 512 * 3,
        num_spatial: int = 512,
        num_edge_dis: int = 128,
        edge_type: str = "multi_hop",
        multi_hop_max_dist: int = 5,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = True,
        pre_layernorm: bool = False,
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: Optional[float] = None,
        freeze_layer_indices: Optional[list[int]] = None,
        traceable: bool = False,
        last_state_only: bool = False,
        use_quant_noise: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:
        super().__init__()

        self.dropout_module = nn.Dropout(dropout)
        self.dropout_p = float(dropout)
        self.layerdrop = float(layerdrop)
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable
        self.last_state_only = last_state_only
        self.embed_scale = embed_scale
        self.pre_layernorm = pre_layernorm

        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        if use_quant_noise and q_noise > 0.0:
            self.quant_noise = quant_noise(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.emb_layer_norm = (
            nn.LayerNorm(self.embedding_dim)
            if encoder_normalize_before
            else None
        )

        self.final_layer_norm = (
            nn.LayerNorm(self.embedding_dim)
            if pre_layernorm
            else None
        )

        self.layers = nn.ModuleList(
            [
                self.build_graphormer_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        self.freeze_layers(freeze_layer_indices)

    def freeze_layers(
        self,
        freeze_layer_indices: Optional[list[int]] = None,
    ) -> None:
        if freeze_layer_indices is None:
            return

        num_layers = len(self.layers)

        for layer_idx in freeze_layer_indices:
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(
                    f"Invalid layer index {layer_idx}. "
                    f"Model has {num_layers} layers, valid range is 0 to {num_layers - 1}."
                )

            for parameter in self.layers[layer_idx].parameters():
                parameter.requires_grad = False

    def build_graphormer_graph_encoder_layer(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        activation_fn: str,
        q_noise: float,
        qn_block_size: int,
        pre_layernorm: bool,
    ) -> GraphormerGraphEncoderLayer:
        return GraphormerGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            pre_layernorm=pre_layernorm,
        )

    def forward(
        self,
        batched_data,
        perturb: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[list[torch.Tensor] | torch.Tensor, torch.Tensor]:
        """
        Forward pass of Graphormer encoder.

        Returns:
            inner_states:
                List of hidden states.
                Each tensor has shape:
                    (batch_size, seq_len, embedding_dim)

            graph_rep:
                Graph-level representation from the CLS token.
                Shape:
                    (batch_size, embedding_dim)
        """

        # ============================================================
        # Padding mask
        # ============================================================

        data_x = batched_data["x"]                     # (B, T, atom_feature_dim)
        n_graph, _ = data_x.size()[:2]

        padding_mask = data_x[:, :, 0].eq(0)           # (B, T)

        padding_mask_cls = torch.zeros(
            n_graph,
            1,
            device=padding_mask.device,
            dtype=padding_mask.dtype,
        )                                              # (B, 1)

        padding_mask = torch.cat(
            (padding_mask_cls, padding_mask),
            dim=1,
        )                                              # (B, T+1)

        # ============================================================
        # Node embedding
        # ============================================================

        if token_embeddings is not None:
            x = token_embeddings                       # (B, T+1, C)
        else:
            x = self.graph_node_feature(batched_data)  # (B, T+1, C)

        # ============================================================
        # Perturbation
        # ============================================================

        if perturb is not None:
            # perturb: (B, T, C)
            x[:, 1:, :] = x[:, 1:, :] + perturb

        # ============================================================
        # Graph attention bias
        # ============================================================

        attn_bias = self.graph_attn_bias(
            batched_data
        )                                              # (B, H, T+1, T+1)

        # ============================================================
        # Embedding scaling
        # ============================================================

        if self.embed_scale is not None:
            x = x * self.embed_scale                   # (B, T+1, C)

        # ============================================================
        # Quant noise
        # ============================================================

        if self.quant_noise is not None:
            x = self.quant_noise(x)                    # (B, T+1, C)

        # ============================================================
        # Embedding LayerNorm
        # ============================================================

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)                 # (B, T+1, C)

        # ============================================================
        # Embedding dropout
        # ============================================================

        x = self.dropout_module(x)                     # (B, T+1, C)

        # ============================================================
        # Encoder layers
        # ============================================================

        inner_states = []

        if not self.last_state_only:
            inner_states.append(x)                     # (B, T+1, C)

        for layer in self.layers:

            if self.training and self.layerdrop > 0.0:
                if torch.rand(1, device=x.device).item() < self.layerdrop:
                    continue

            x = layer(
                x,                                    # (B, T+1, C)
                self_attn_padding_mask=padding_mask,   # (B, T+1)
                self_attn_mask=attn_mask,              # (T+1, T+1) or None
                self_attn_bias=attn_bias,              # (B, H, T+1, T+1)
            )                                         # (B, T+1, C)

            if not self.last_state_only:
                inner_states.append(x)

        # ============================================================
        # Final LayerNorm (Pre-LN Graphormer)
        # ============================================================

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)               # (B, T+1, C)

            if not self.last_state_only:
                inner_states[-1] = x

        # ============================================================
        # Graph representation (CLS token)
        # ============================================================

        graph_rep = x[:, 0, :]                         # (B, C)

        if self.last_state_only:
            inner_states = [x]

        # ============================================================
        # Return
        # ============================================================

        if self.traceable:
            # (num_layers+1, B, T+1, C)
            return torch.stack(inner_states), graph_rep

        return inner_states, graph_rep

    def get_config(self) -> dict:
        return {
            "num_atoms": self.graph_node_feature.num_atoms,
            "num_in_degree": self.graph_node_feature.num_in_degree,
            "num_out_degree": self.graph_node_feature.num_out_degree,
            "num_edges": self.graph_attn_bias.num_edges,
            "num_spatial": self.graph_attn_bias.num_spatial,
            "num_edge_dis": self.graph_attn_bias.num_edge_dis,
            "edge_type": self.graph_attn_bias.edge_type,
            "multi_hop_max_dist": self.graph_attn_bias.multi_hop_max_dist,
            "num_encoder_layers": len(self.layers),
            "embedding_dim": self.embedding_dim,
            "ffn_embedding_dim": (
                self.layers[0].embedding_dim
                if len(self.layers) > 0 and hasattr(self.layers[0], "embedding_dim")
                else None
            ),
            "num_attention_heads": (
                self.layers[0].num_heads
                if len(self.layers) > 0
                else None
            ),
            "dropout": self.dropout_p,
            "attention_dropout": (
                self.layers[0].self_attn.dropout_module.p
                if len(self.layers) > 0
                else None
            ),
            "activation_dropout": (
                self.layers[0].activation_dropout_module.p
                if len(self.layers) > 0
                else None
            ),
            "layerdrop": self.layerdrop,
            "encoder_normalize_before": self.emb_layer_norm is not None,
            "pre_layernorm": self.final_layer_norm is not None,
            "apply_graphormer_init": self.apply_graphormer_init,
            "activation_fn": (
                self.layers[0].activation_fn.__class__.__name__
                if len(self.layers) > 0
                else None
            ),
            "embed_scale": self.embed_scale,
            "freeze_layer_indices": [],
            "traceable": self.traceable,
            "last_state_only": self.last_state_only,
            "use_quant_noise": self.quant_noise is not None,
            "q_noise": (
                self.quant_noise.q_noise
                if self.quant_noise is not None and hasattr(self.quant_noise, "q_noise")
                else 0.0
            ),
            "qn_block_size": (
                self.quant_noise.qn_block_size
                if self.quant_noise is not None and hasattr(self.quant_noise, "qn_block_size")
                else 0
            ),
        }