import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable

class ResidualAdaptor(nn.Module):
    """
    Residual adaptor module that learns a small delta and adds it to the input,
    followed by LayerNorm. Intended for use as a lightweight task-specific adapter.
    """
    def __init__(
        self,
        dim: int,
        bottleneck: int = 32,
        dropout: float = 0.1,
        activation: Callable = F.relu,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)
        self.activation = activation

        # weight initialization (helpful for small adapters)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: input tensor of shape (B, ..., dim)

        Returns:
            Tensor of same shape as z after applying adaptor and layer norm.
        """
        out = self.down(z)
        out = self.activation(out)
        out = self.dropout(out)
        delta = self.up(out)
        return self.ln(z + delta)


class GateResidualAdaptor(nn.Module):
    """
    Gated residual adaptor. Learns a delta like ResidualAdaptor but scales it
    with a learned scalar gate (constrained to (0,1) with sigmoid).
    """
    def __init__(
        self,
        dim: int,
        bottleneck: int = 32,
        dropout: float = 0.1,
        activation: Callable = F.relu,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        # a single scalar gate parameter; sigmoid(self.alpha) produces gate in (0,1)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.down(z)
        out = self.activation(out)
        out = self.dropout(out)
        delta = self.up(out)
        gate = torch.tanh(self.alpha)  # Keep identity at the beginning with a parameter
        return self.ln(z + gate * delta)
