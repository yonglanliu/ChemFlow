# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import torch
import torch.nn as nn
from src.deep_learning.graphormer.graphormer_encoder import GraphormerGraphEncoder

class GraphormerDenoiser(nn.Module):
    def __init__(
        self,
        encoder: GraphormerGraphEncoder,
        hidden_dim: int,
        num_atom_types: int,
        num_bond_types: int,
        dropout: float = 0.1,
        bond_pair_mode: str = "sum",
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.num_bond_classes = num_bond_types
        self.bond_pair_mode = bond_pair_mode

        self.atom_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_atom_types),
        )

        if bond_pair_mode == "cat":
            bond_in_dim = hidden_dim * 2

        elif bond_pair_mode == "sum":
            self.node_pair_proj = nn.Linear(hidden_dim, hidden_dim)
            bond_in_dim = hidden_dim

        elif bond_pair_mode == "bilinear":
            self.node_pair_left = nn.Linear(hidden_dim, hidden_dim)
            self.node_pair_right = nn.Linear(hidden_dim, hidden_dim)
            bond_in_dim = hidden_dim

        else:
            raise ValueError(f"Unknown bond_pair_mode: {bond_pair_mode}")

        self.bond_head = nn.Sequential(
            nn.LayerNorm(bond_in_dim),
            nn.Dropout(dropout),
            nn.Linear(bond_in_dim, self.num_bond_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:

        inner_states, _ = self.encoder(
            batch,
            perturb=None,
            attn_mask=None,
        )

        h = inner_states[-1]      # [B, N+1, H]

        assert torch.isfinite(h).all(), "Graphormer encoder output has NaN/Inf"

        node_h = h[:, 1:, :]      # [B, N, H]  no CLS token

        atom_logits = self.atom_head(node_h)

        B, N, H = node_h.shape

        if self.bond_pair_mode == "cat":
            h_i = node_h.unsqueeze(2).expand(B, N, N, H)
            h_j = node_h.unsqueeze(1).expand(B, N, N, H)
            pair_h = torch.cat([h_i, h_j], dim=-1)

        elif self.bond_pair_mode == "sum":
            z = self.node_pair_proj(node_h)
            pair_h = z.unsqueeze(2) + z.unsqueeze(1)

        elif self.bond_pair_mode == "bilinear":
            z_i = self.node_pair_left(node_h)
            z_j = self.node_pair_right(node_h)
            pair_h = z_i.unsqueeze(2) * z_j.unsqueeze(1)

        bond_logits = self.bond_head(pair_h)

        assert torch.isfinite(atom_logits).all(), "atom_logits has NaN/Inf"
        assert torch.isfinite(bond_logits).all(), "bond_logits has NaN/Inf"

        return atom_logits, bond_logits


    def get_config(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_atom_types": self.num_atom_types,
            "num_bond_types": self.num_bond_types,
            "num_bond_classes": self.num_bond_classes,
            "bond_pair_mode": self.bond_pair_mode,
            "dropout": self.atom_head[1].p,
        }