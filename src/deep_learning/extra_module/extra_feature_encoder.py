
from torch import nn


class ExtraFeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)