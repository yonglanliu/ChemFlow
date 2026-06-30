# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import torch
from torch import nn

from src.gpt.position_embedding import PositionalEmbedding
from src.gpt.quant_noise import quant_noise
from src.gpt.multihead_attention import MultiheadAttention


class GPTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 1024,
        bias: bool = False,
        dropout: float = 0.1,
        use_quant_noise: bool = False,
        quant_noise_p: float = 0.0,
        quant_noise_block_size: int = 8,
    ):
        super().__init__()

        if not use_quant_noise:
            quant_noise_p = 0.0

        self.ln1 = nn.LayerNorm(d_model)

        self.attention_layer = MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            bias=bias,
            self_attention=True,
            q_noise=quant_noise_p,
            qn_block_size=quant_noise_block_size,
        )

        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            quant_noise(
                nn.Linear(d_model, d_ff),
                quant_noise_p,
                quant_noise_block_size,
            ),
            nn.GELU(),
            quant_noise(
                nn.Linear(d_ff, d_model),
                quant_noise_p,
                quant_noise_block_size,
            ),
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """
        x:
            (batch_size, seq_len, d_model)

        attn_mask:
            (seq_len, seq_len)

        padding_mask:
            (batch_size, seq_len)
            True = pad token
            False = real token
        """

        x_t = x.transpose(0, 1)  # (B, T, C) -> (T, B, C)

        h = self.ln1(x_t)

        attn_out, attn_weights = self.attention_layer(
            query=h,
            key=h,
            value=h,
            attn_bias=attn_bias,
            attn_mask=attn_mask,
            padding_mask=padding_mask,
            need_weights=False,
        )

        x_t = x_t + self.dropout1(attn_out)

        ffn_out = self.ffn(self.ln2(x_t))
        x_t = x_t + self.dropout2(ffn_out)

        x = x_t.transpose(0, 1)  # (T, B, C) -> (B, T, C)

        return x, attn_weights


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        use_quant_noise: bool = False,
        quant_noise_p: float = 0.0,
        quant_noise_block_size: int = 8,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.vocab_size = vocab_size
        self.max_len = max_len

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout

        self.use_quant_noise = use_quant_noise
        self.quant_noise_p = quant_noise_p if use_quant_noise else 0.0
        self.quant_noise_block_size = quant_noise_block_size

        self.token_emb = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )

        self.pos_emb = PositionalEmbedding(
            d_model=d_model,
            max_len=max_len,
        )

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_quant_noise=use_quant_noise,
                    quant_noise_p=self.quant_noise_p,
                    quant_noise_block_size=quant_noise_block_size,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        self.head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # Weight tying
        self.head.weight = self.token_emb.weight

    def build_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=device,
        )

        mask = torch.triu(mask, diagonal=1)

        return mask

    def forward(
        self,
        batch_data: torch.Tensor | dict,
    ) -> torch.Tensor:
        """
        batch_data can be:

        1. Tensor:
            input_ids with shape (batch_size, seq_len)

        2. Dict:
            {
                "input_ids": tensor,
                "padding_mask": optional bool tensor
            }

        Returns
        -------
        logits:
            (batch_size, seq_len, vocab_size)
        """

        if isinstance(batch_data, torch.Tensor):
            input_ids = batch_data
            padding_mask = None
        else:
            input_ids = batch_data["input_ids"]
            padding_mask = batch_data.get("padding_mask", None)

        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.max_len}"
            )

        x = self.token_emb(input_ids)
        x = self.pos_emb(x)
        x = self.drop(x)

        attn_mask = self.build_causal_mask(
            seq_len=seq_len,
            device=input_ids.device,
        )

        if padding_mask is None and self.pad_token_id is not None:
            padding_mask = input_ids.eq(self.pad_token_id)

        for block in self.blocks:
            x, _ = block(
                x=x,
                attn_bias=None,
                attn_mask=attn_mask,
                padding_mask=padding_mask,
            )

        x = self.ln_f(x)

        logits = self.head(x)

        return logits

    def get_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "use_quant_noise": self.use_quant_noise,
            "quant_noise_p": self.quant_noise_p,
            "quant_noise_block_size": self.quant_noise_block_size,
        }