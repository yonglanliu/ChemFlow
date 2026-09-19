import torch
from torch import nn

class RegressionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, 1),
        )

    def forward(self, graph_rep):
        return self.net(graph_rep).squeeze(-1)




class ClassificationHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, num_classes),
        )

    def forward(self, graph_rep):
        return self.net(graph_rep)