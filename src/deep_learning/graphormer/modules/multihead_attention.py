# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.deep_learning.utils.quant_noise import quant_noise


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        self_attention: bool = True,
        batch_first: bool = True,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.self_attention = self_attention
        self.batch_first = batch_first
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        self.kdim = embed_dim
        self.vdim = embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if not self.self_attention:
            raise ValueError("Only self-attention is supported.")
        if not self.qkv_same_dim:
            raise ValueError("Only qkv_same_dim=True is supported.")

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.scaling = self.head_dim ** -0.5
        self.dropout_module = nn.Dropout(dropout, inplace=False)

        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            q_noise,
            qn_block_size,
        )
        self.k_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            q_noise,
            qn_block_size,
        )
        self.v_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            q_noise,
            qn_block_size,
        )
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            q_noise,
            qn_block_size,
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        gain = 1.0 / math.sqrt(2.0)

        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _check_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("query, key, and value must all be 3D tensors")

        if self.batch_first:
            batch_size, tgt_len, embed_dim = query.size()
            key_batch, src_len, key_dim = key.size()
            value_batch, value_src_len, value_dim = value.size()
        else:
            tgt_len, batch_size, embed_dim = query.size()
            src_len, key_batch, key_dim = key.size()
            value_src_len, value_batch, value_dim = value.size()

        if embed_dim != self.embed_dim:
            raise ValueError(f"query embed_dim={embed_dim}, expected {self.embed_dim}")
        if key_batch != batch_size:
            raise ValueError("key batch size mismatch")
        if value_batch != batch_size:
            raise ValueError("value batch size mismatch")
        if key_dim != self.kdim:
            raise ValueError(f"key embed_dim={key_dim}, expected {self.kdim}")
        if value_dim != self.vdim:
            raise ValueError(f"value embed_dim={value_dim}, expected {self.vdim}")
        if value_src_len != src_len:
            raise ValueError("key/value seq_len mismatch")

        return tgt_len, src_len, batch_size, embed_dim

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)
        return q, k, v

    def _shape_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tgt_len: int,
        src_len: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.batch_first:
            q = (
                q.view(batch_size, tgt_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .reshape(batch_size * self.num_heads, tgt_len, self.head_dim)
            )

            k = (
                k.view(batch_size, src_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .reshape(batch_size * self.num_heads, src_len, self.head_dim)
            )

            v = (
                v.view(batch_size, src_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .reshape(batch_size * self.num_heads, src_len, self.head_dim)
            )

        else:
            q = (
                q.view(tgt_len, batch_size, self.num_heads, self.head_dim)
                .permute(1, 2, 0, 3)
                .reshape(batch_size * self.num_heads, tgt_len, self.head_dim)
            )

            k = (
                k.view(src_len, batch_size, self.num_heads, self.head_dim)
                .permute(1, 2, 0, 3)
                .reshape(batch_size * self.num_heads, src_len, self.head_dim)
            )

            v = (
                v.view(src_len, batch_size, self.num_heads, self.head_dim)
                .permute(1, 2, 0, 3)
                .reshape(batch_size * self.num_heads, src_len, self.head_dim)
            )

        return q, k, v

    def _compute_attention_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        batch_size: int,
        tgt_len: int,
        src_len: int,
    ) -> torch.Tensor:
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        expected_shape = [batch_size * self.num_heads, tgt_len, src_len]
        if list(attn_weights.size()) != expected_shape:
            raise RuntimeError(
                f"attn_weights shape={attn_weights.size()}, expected={expected_shape}"
            )

        return attn_weights

    def _apply_attention_bias(
        self,
        attn_weights: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        batch_size: int,
        tgt_len: int,
        src_len: int,
    ) -> torch.Tensor:
        if attn_bias is None:
            return attn_weights

        expected_shape = [batch_size, self.num_heads, tgt_len, src_len]
        if list(attn_bias.size()) != expected_shape:
            raise ValueError(
                f"attn_bias shape={attn_bias.size()}, expected={expected_shape}"
            )

        attn_bias = attn_bias.reshape(
            batch_size * self.num_heads,
            tgt_len,
            src_len,
        )

        return attn_weights + attn_bias

    def _apply_attention_mask(
        self,
        attn_weights: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        tgt_len: int,
        src_len: int,
    ) -> torch.Tensor:
        if attn_mask is None:
            return attn_weights

        expected_shape = [tgt_len, src_len]
        if list(attn_mask.size()) != expected_shape:
            raise ValueError(
                f"attn_mask shape={attn_mask.size()}, expected={expected_shape}"
            )

        return attn_weights + attn_mask.unsqueeze(0)

    def _apply_key_padding_mask(
        self,
        attn_weights: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        tgt_len: int,
        src_len: int,
    ) -> torch.Tensor:
        if key_padding_mask is None:
            return attn_weights

        if key_padding_mask.dim() == 0:
            return attn_weights

        expected_shape = [batch_size, src_len]
        if list(key_padding_mask.size()) != expected_shape:
            raise ValueError(
                f"key_padding_mask shape={key_padding_mask.size()}, "
                f"expected={expected_shape}"
            )

        key_padding_mask = key_padding_mask.to(torch.bool)

        attn_weights = attn_weights.view(
            batch_size,
            self.num_heads,
            tgt_len,
            src_len,
        )

        attn_weights = attn_weights.masked_fill(
            key_padding_mask[:, None, None, :],
            float("-inf"),
        )

        attn_weights = attn_weights.view(
            batch_size * self.num_heads,
            tgt_len,
            src_len,
        )

        return attn_weights

    def _softmax_attention(
        self,
        attn_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights_float = F.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        )

        attn_probs = self.dropout_module(
            attn_weights_float.type_as(attn_weights)
        )

        return attn_weights_float, attn_probs

    def _compute_attention_output(
        self,
        attn_probs: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        tgt_len: int,
        embed_dim: int,
    ) -> torch.Tensor:
        attn = torch.bmm(attn_probs, v)

        expected_shape = [
            batch_size * self.num_heads,
            tgt_len,
            self.head_dim,
        ]

        if list(attn.size()) != expected_shape:
            raise RuntimeError(
                f"attention output shape={attn.size()}, expected={expected_shape}"
            )

        if self.batch_first:
            attn = (
                attn.view(batch_size, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(batch_size, tgt_len, embed_dim)
            )
        else:
            attn = (
                attn.view(batch_size, self.num_heads, tgt_len, self.head_dim)
                .permute(2, 0, 1, 3)
                .reshape(tgt_len, batch_size, embed_dim)
            )

        return self.out_proj(attn)

    def format_attention_weights(
        self,
        attn_weights: torch.Tensor,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        need_head_weights: bool = False,
    ) -> torch.Tensor:
        attn_weights = attn_weights.view(
            batch_size,
            self.num_heads,
            tgt_len,
            src_len,
        )

        if need_head_weights:
            return attn_weights.transpose(0, 1)

        return attn_weights.mean(dim=1)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if key is None:
            key = query

        if value is None:
            value = query

        tgt_len, src_len, batch_size, embed_dim = self._check_inputs(
            query=query,
            key=key,
            value=value,
        )

        q, k, v = self._project_qkv(
            query=query,
            key=key,
            value=value,
        )

        q, k, v = self._shape_qkv(
            q=q,
            k=k,
            v=v,
            tgt_len=tgt_len,
            src_len=src_len,
            batch_size=batch_size,
        )

        attn_weights = self._compute_attention_scores(
            q=q,
            k=k,
            batch_size=batch_size,
            tgt_len=tgt_len,
            src_len=src_len,
        )

        attn_weights = self._apply_attention_bias(
            attn_weights=attn_weights,
            attn_bias=attn_bias,
            batch_size=batch_size,
            tgt_len=tgt_len,
            src_len=src_len,
        )

        attn_weights = self._apply_attention_mask(
            attn_weights=attn_weights,
            attn_mask=attn_mask,
            tgt_len=tgt_len,
            src_len=src_len,
        )

        attn_weights = self._apply_key_padding_mask(
            attn_weights=attn_weights,
            key_padding_mask=padding_mask,
            batch_size=batch_size,
            tgt_len=tgt_len,
            src_len=src_len,
        )

        attn_weights_float, attn_probs = self._softmax_attention(attn_weights)

        attn = self._compute_attention_output(
            attn_probs=attn_probs,
            v=v,
            batch_size=batch_size,
            tgt_len=tgt_len,
            embed_dim=embed_dim,
        )

        if need_weights:
            formatted_weights = self.format_attention_weights(
                attn_weights=attn_weights_float,
                batch_size=batch_size,
                tgt_len=tgt_len,
                src_len=src_len,
                need_head_weights=need_head_weights,
            )
            return attn, formatted_weights

        return attn