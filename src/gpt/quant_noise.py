# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

def _fake_quantize_8bit_symmetric(
    w: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-tensor symmetric fake quantization to int8 range [-127, 127] with STE.
    This simulates int8 quantization but keeps dtype as float for training.
    """
    max_abs = w.detach().abs().amax()
    scale = torch.clamp(max_abs / 127.0, min=eps)

    q = torch.clamp((w / scale).round(), -127, 127) * scale

    # STE: forward uses q, backward treats it like w
    return w + (q - w).detach()


class QuantNoiseLinear(nn.Module):
    def __init__(self, base: nn.Linear, p: float, block: int):
        super().__init__()
        self.base = base
        self.p = float(p)
        self.block = int(block)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.base.weight
        b = self.base.bias

        # Only apply quantization noise during training
        if (not self.training) or self.p <= 0:
            return F.linear(x, w, b)

        out_dim, in_dim = w.shape
        block = max(1, self.block)

        # If dimensions are not divisible by block size,
        # fall back to element-wise blocks
        if (out_dim % block) != 0 or (in_dim % block) != 0:
            block = 1

        bo = out_dim // block
        bi = in_dim // block

        # block_mask: 1 means this block will be fake-quantized
        block_mask = (
            torch.rand(bo, bi, device=w.device) < self.p
        ).to(w.dtype)

        # Expand block mask back to full weight matrix shape
        mask = block_mask.repeat_interleave(
            block, dim=0
        ).repeat_interleave(
            block, dim=1
        )

        # Fake-quantized weight
        w_q = _fake_quantize_8bit_symmetric(w)

        # Mix original weight and fake-quantized weight
        w_noisy = w * (1.0 - mask) + w_q * mask

        return F.linear(x, w_noisy, b)


def quant_noise(
    module: nn.Module,
    q_noise: float,
    qn_block_size: int,
) -> nn.Module:
    """
    Training-time regularizer:
    randomly fake-quantize blocks of a Linear layer's weight matrix.
    """
    if q_noise is None or q_noise <= 0:
        return module

    if not isinstance(module, nn.Linear):
        raise TypeError("quant_noise currently supports nn.Linear only.")

    return QuantNoiseLinear(module, q_noise, qn_block_size)