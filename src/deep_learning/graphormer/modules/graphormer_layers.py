# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#

from typing import Callable, Optional

import torch
import torch.nn as nn
from src.deep_learning.graphormer.modules.multihead_attention import MultiheadAttention
from src.deep_learning.utils.quant_noise import quant_noise
import math

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers)) # Prevent variance explosion in deep residual stacks by using sqrt(n_layers)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if getattr(module, "padding_idx", None) is not None and module.padding_idx is not None:
            with torch.no_grad():
                module.weight.data[module.padding_idx].zero_()

class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree

        # +1 for padding/unknown (and graph token handled separately)
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x = batched_data["x"]
        # print("x unique:", torch.unique(batched_data["x"])[:100])
        # print("x min/max:", batched_data["x"].min(), batched_data["x"].max())
        # print("node_mask sum:", batched_data["node_mask"].sum())
        # print("embedding weight mean:", self.atom_encoder.weight.abs().mean())
        in_degree = batched_data["in_degree"]
        out_degree = batched_data["out_degree"]

        n_graph, n_node = x.size()[:2]

        # x is typically [B, N, F] where F is feature fields; sum over fields
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [B, N, C]
        # print("h after embedding:", node_feature.dtype, node_feature.shape, node_feature.abs().mean(), node_feature.min(), node_feature.max())
        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)  # [B, N+1, C]

        return graph_node_feature


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.

    Expects batched_data keys:
      - attn_bias: [B, N+1, N+1]
      - spatial_pos: [B, N, N]
      - x: [B, N, F] (used only for sizes)
      - attn_edge_type: [B, N, N, E] (for non-multi-hop), OR
      - edge_input: [B, N, N, max_dist, E] (for multi-hop)
    """

    def __init__(
        self,
        num_heads: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        hidden_dim: int,
        edge_type: str,
        multi_hop_max_dist: int,
        n_layers: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.edge_type = edge_type
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.hidden_dim = hidden_dim
        self.multi_hop_max_dist = multi_hop_max_dist
        self.num_edge_dis = num_edge_dis

        # Edge-type embedding (per-head bias)
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)

        # Spatial position embedding (per-head bias)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        # Graph token virtual distance bias
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        # Distance encoder used in multi-hop mode.
        #
        # IMPORTANT: to match checkpoints that store:
        #   graph_attn_bias.edge_dis_encoder.weight
        # we keep this module name exactly `edge_dis_encoder`.
        #
        # In many Graphormer implementations, this weight is shaped:
        #   [num_edge_dis * num_heads * num_heads, 1]
        # and then reshaped to [num_edge_dis, num_heads, num_heads].
        #
        # We'll follow that convention *when edge_type == "multi_hop"*.
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        else:
            # If not multi-hop, we still define it so loading won't fail if present,
            # but it won't be used in forward.
            # (If you prefer, you can delete this and accept skipping the key.)
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias = batched_data["attn_bias"]          # [B, N+1, N+1]
        spatial_pos = batched_data["spatial_pos"]      # [B, N, N]
        x = batched_data["x"]                          # [B, N, F]

        n_graph, n_node = x.size()[:2]

        # base bias -> [B, H, N+1, N+1]
        graph_attn_bias = attn_bias.clone().unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # ---- spatial positional bias ----
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)  # [B,H,N,N]
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # ---- graph token virtual distance ----
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)  # [1,H,1]
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # ---- edge feature bias ----
        if self.edge_type == "multi_hop":
            edge_input = batched_data["edge_input"]  # [B,N,N,max_dist,E] (E often 1 or small)
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # pad -> 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)  # >1 => -1

            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]

            # edge_input embed -> [B,N,N,max_dist,H]
            edge_input = self.edge_encoder(edge_input).mean(-2)

            max_dist = edge_input.size(-2)

            # flatten -> [max_dist, B*N*N, H]
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )

            # distance transform weights -> [num_edge_dis, H, H]
            # edge_dis_encoder.weight is [num_edge_dis*H*H, 1]
            w = self.edge_dis_encoder.weight.reshape(self.num_edge_dis, self.num_heads, self.num_heads)

            # only use first max_dist slices
            w = w[:max_dist, :, :]  # [max_dist, H, H]

            # batched matmul: [max_dist, B*N*N, H] x [max_dist, H, H]
            edge_input_flat = torch.bmm(edge_input_flat, w)

            # reshape back -> [B,N,N,max_dist,H]
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)

            # sum over distance and normalize by spatial_pos_
            edge_input = (edge_input.sum(-2) / spatial_pos_.float().unsqueeze(-1)).permute(0, 3, 1, 2)  # [B,H,N,N]
        else:
            attn_edge_type = batched_data["attn_edge_type"]  # [B,N,N,E]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)  # [B,H,N,N]

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input

        # reset (some implementations add original attn_bias again)
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)

        return graph_attn_bias

def get_activation_fn(name: str):
    name = (name or "").lower()
    if name in ("relu",):
        return torch.relu
    if name in ("gelu",):
        return torch.nn.functional.gelu
    if name in ("silu", "swish"):
        return torch.nn.functional.silu
    if name in ("tanh",):
        return torch.tanh
    if name in ("sigmoid",):
        return torch.sigmoid
    raise ValueError(f"Unsupported activation_fn: {name}")


class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        pre_layernorm: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.pre_layernorm = pre_layernorm

        self.dropout_module = nn.Dropout(dropout)
        self.activation_dropout_module = nn.Dropout(activation_dropout)

        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            self.num_heads,
            dropout=attention_dropout,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # Layer norms
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        # FFN
        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        q_noise,
        qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_attention_heads,
            dropout=dropout,
            self_attention=True,
            batch_first=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.

        x: T x B x C
        """
        residual = x
        # print("x before self-attn:", x.dtype, x.shape, x.abs().mean(), x.min(), x.max())
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        #print("x before self-attn:", x.dtype, x.shape, x.abs().mean(), x.min(), x.max())
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if not self.pre_layernorm:
            x = self.final_layer_norm(x)

        return x