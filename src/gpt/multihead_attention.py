# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.gpt.quant_noise import quant_noise



class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        self_attention: bool = True,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.self_attention = self_attention
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        self.kdim = embed_dim
        self.vdim = embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        assert self.self_attention, "Only self-attention is supported."
        assert self.qkv_same_dim, "Only qkv_same_dim=True is supported."

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

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
        """
        Expected shape:
            query/key/value: (seq_len, batch_size, embed_dim)
        """
        tgt_len, batch_size, embed_dim = query.size()
        src_len = key.size(0)

        assert embed_dim == self.embed_dim, (
            f"query embed_dim={embed_dim}, expected {self.embed_dim}"
        )
        assert key.size(1) == batch_size, "key batch size mismatch"
        assert value.size(1) == batch_size, "value batch size mismatch"
        assert key.size(2) == self.kdim, "key embed_dim mismatch"
        assert value.size(2) == self.vdim, "value embed_dim mismatch"
        assert value.size(0) == src_len, "key/value seq_len mismatch"

        return tgt_len, src_len, batch_size, embed_dim

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project input into Q, K, V.

        Input:
            (seq_len, batch_size, embed_dim)

        Output:
            (seq_len, batch_size, embed_dim)
        """
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
        """
        Convert Q/K/V into multi-head format.

        Before:
            q: (tgt_len, batch_size, embed_dim)
            k: (src_len, batch_size, embed_dim)
            v: (src_len, batch_size, embed_dim)

        After:
            q: (batch_size * num_heads, tgt_len, head_dim)
            k: (batch_size * num_heads, src_len, head_dim)
            v: (batch_size * num_heads, src_len, head_dim)
        """
        q = (
            q.contiguous()
            .view(tgt_len, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        k = (
            k.contiguous()
            .view(src_len, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        v = (
            v.contiguous()
            .view(src_len, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
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
        """
        Compute raw attention scores QK^T.

        Output:
            (batch_size * num_heads, tgt_len, src_len)
        """
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        expected_shape = [batch_size * self.num_heads, tgt_len, src_len]
        assert list(attn_weights.size()) == expected_shape, (
            f"attn_weights shape={attn_weights.size()}, "
            f"expected={expected_shape}"
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
        """
        attn_bias shape:
            (batch_size, num_heads, tgt_len, src_len)
        """
        if attn_bias is None:
            return attn_weights

        assert list(attn_bias.size()) == [
            batch_size,
            self.num_heads,
            tgt_len,
            src_len,
        ], f"attn_bias shape={attn_bias.size()} is not expected"

        attn_weights = attn_weights + attn_bias.view(
            batch_size * self.num_heads,
            tgt_len,
            src_len,
        )

        return attn_weights

    def _apply_attention_mask(
        self,
        attn_weights: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        tgt_len: int,
        src_len: int,
    ) -> torch.Tensor:
        """
        attn_mask shape:
            (tgt_len, src_len)

        For GPT causal mask:
            allowed positions = 0
            future positions = -inf
        """
        if attn_mask is None:
            return attn_weights

        assert list(attn_mask.size()) == [tgt_len, src_len], (
            f"attn_mask shape={attn_mask.size()}, "
            f"expected={[tgt_len, src_len]}"
        )

        attn_weights = attn_weights + attn_mask.unsqueeze(0)

        return attn_weights

    def _apply_key_padding_mask(
        self,
        attn_weights: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        tgt_len: int,
        src_len: int,
    ) -> torch.Tensor:
        """
        key_padding_mask shape:
            (batch_size, src_len)

        True means padding token.
        """
        if key_padding_mask is None:
            return attn_weights

        if key_padding_mask.dim() == 0:
            return attn_weights

        assert key_padding_mask.size(0) == batch_size, (
            f"key_padding_mask batch={key_padding_mask.size(0)}, "
            f"expected={batch_size}"
        )
        assert key_padding_mask.size(1) == src_len, (
            f"key_padding_mask seq_len={key_padding_mask.size(1)}, "
            f"expected={src_len}"
        )

        attn_weights = attn_weights.view(
            batch_size,
            self.num_heads,
            tgt_len,
            src_len,
        )

        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
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
        """
        Softmax over source positions.

        Returns:
            attn_weights_float:
                attention probabilities in float32

            attn_probs:
                dropout-applied attention probabilities
        """
        attn_weights_float = F.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        )

        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        return attn_weights_float, attn_probs

    def _compute_attention_output(
        self,
        attn_probs: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        tgt_len: int,
        embed_dim: int,
    ) -> torch.Tensor:
        """
        attn_probs:
            (batch_size * num_heads, tgt_len, src_len)

        v:
            (batch_size * num_heads, src_len, head_dim)

        Output:
            (tgt_len, batch_size, embed_dim)
        """
        attn = torch.bmm(attn_probs, v)

        expected_shape = [
            batch_size * self.num_heads,
            tgt_len,
            self.head_dim,
        ]

        assert list(attn.size()) == expected_shape, (
            f"attention output shape={attn.size()}, "
            f"expected={expected_shape}"
        )

        attn = (
            attn.transpose(0, 1)
            .contiguous()
            .view(tgt_len, batch_size, embed_dim)
        )

        attn = self.out_proj(attn)

        return attn

    def get_attention_weights(
        self,
        attn_weights_float: torch.Tensor,
        batch_size: int,
        tgt_len: int,
        src_len: int,
    ) -> torch.Tensor:
        """
        If need_head_weights=True:
            return shape:
                (num_heads, batch_size, tgt_len, src_len)

        If need_head_weights=False:
            average over heads and return:
                (batch_size, tgt_len, src_len)
        """
        attn_weights = attn_weights_float.view(
            batch_size,
            self.num_heads,
            tgt_len,
            src_len,
        ).transpose(1, 0)

        return attn_weights

    def get_head_attention_weights(
        self,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        return attn_weights.mean(dim=0)


    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Multi-head self-attention forward pass.

        Args:
            query:
                Tensor of shape (seq_len, batch_size, embed_dim)

            key:
                Tensor of shape (seq_len, batch_size, embed_dim).
                If None, key=query.

            value:
                Tensor of shape (seq_len, batch_size, embed_dim).
                If None, value=query.

            attn_bias:
                Optional tensor of shape:
                (batch_size, num_heads, seq_len, seq_len)

            seq_padding_mask:
                Optional tensor of shape:
                (batch_size, seq_len)
                True means padding token.

            attn_mask:
                Optional tensor of shape:
                (seq_len, seq_len)
                For GPT, this is usually the causal mask.

            need_weights:
                Whether to return averaged attention weights.

        Returns:
            attn:
                Tensor of shape (seq_len, batch_size, embed_dim)

            attn_weights:
                None, or:
                (batch_size, seq_len, seq_len), if averaged
                (num_heads, batch_size, seq_len, seq_len), if per-head
        """

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

        # dim: (tgt_len, batch_size, embed_dim)
        attn = self._compute_attention_output(
            attn_probs=attn_probs,
            v=v,
            batch_size=batch_size,
            tgt_len=tgt_len,
            embed_dim=embed_dim,
        )

        attn_weights_out = None

        if need_weights:
            attn_weights_out = self.get_attention_weights(
                attn_weights_float=attn_weights_float,
                batch_size=batch_size,
                tgt_len=tgt_len,
                src_len=src_len,
            ) # dim: (batch_size, tgt_len, src_len) or (num_heads, batch_size, tgt_len, src_len)

        return attn, attn_weights_out


