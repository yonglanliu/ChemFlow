# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.deep_learning.diffusion.modules.noise.mask_noise import mask_diffusion_batch


class GraphormerDiffuser(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        num_timesteps: int,
        atom_mask_token: int,
        bond_mask_token: int,
        atom_pad_token: int = 0,
        bond_pad_token: int = 0,
        atom_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.denoiser = denoiser
        self.num_timesteps = num_timesteps

        self.atom_mask_token = atom_mask_token
        self.bond_mask_token = bond_mask_token
        self.atom_pad_token = atom_pad_token
        self.bond_pad_token = bond_pad_token

        self.atom_loss_weight = atom_loss_weight
        self.bond_loss_weight = bond_loss_weight

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        # Mask the input batch to create a noisy version for diffusion training
        noisy_batch = mask_diffusion_batch(
            batch=batch,
            num_timesteps=self.num_timesteps,
            atom_mask_token=self.atom_mask_token,
            bond_mask_token=self.bond_mask_token,
        )

        atom_logits, bond_logits = self.denoiser(noisy_batch)

        clean_atom_types = batch["atom_types"].long()
        clean_bond_types = batch["bond_types"].long()

        node_mask = batch["node_mask"].bool()

        atom_mask = noisy_batch["atom_mask"].bool() & node_mask

        valid_edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        bond_mask = noisy_batch["bond_mask"].bool() & valid_edge_mask


        loss_dict = self.compute_loss(
            atom_logits=atom_logits,
            bond_logits=bond_logits,
            clean_atom_types=clean_atom_types,
            clean_bond_types=clean_bond_types,
            atom_mask=atom_mask,
            bond_mask=bond_mask,
        )

        return {
            **loss_dict,
            "atom_logits": atom_logits,
            "bond_logits": bond_logits,
            "noisy_batch": noisy_batch,
        }

    def compute_loss(
        self,
        atom_logits: torch.Tensor,
        bond_logits: torch.Tensor,
        clean_atom_types: torch.Tensor,
        clean_bond_types: torch.Tensor,
        atom_mask: torch.Tensor,
        bond_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if atom_mask.any():
            atom_loss = F.cross_entropy(
                atom_logits[atom_mask],
                clean_atom_types[atom_mask],
            )
        else:
            atom_loss = atom_logits.sum() * 0.0

        if bond_mask.any():
            bond_loss = F.cross_entropy(
                bond_logits[bond_mask],
                clean_bond_types[bond_mask],
            )
        else:
            bond_loss = bond_logits.sum() * 0.0

        loss = (
            self.atom_loss_weight * atom_loss
            + self.bond_loss_weight * bond_loss
        )

        assert torch.isfinite(loss), "loss has NaN/Inf"
        assert torch.isfinite(atom_loss), "atom_loss has NaN/Inf"
        assert torch.isfinite(bond_loss), "bond_loss has NaN/Inf"

        return {
            "loss": loss,
            "atom_loss": atom_loss,
            "bond_loss": bond_loss,
        }


    def _check_input(self,batch):
        for k, v in batch.items():
            if torch.is_tensor(v):
                if not torch.isfinite(v.float()).all():
                    raise RuntimeError(f"{k} has NaN/Inf")

        x = batch["x"]
        print("x:", x.shape, x.min().item(), x.max().item())

        if "attn_bias" in batch:
            a = batch["attn_bias"]
            print("attn_bias:", a.shape, a.min().item(), a.max().item())

        if "spatial_pos" in batch:
            s = batch["spatial_pos"]
            print("spatial_pos:", s.shape, s.min().item(), s.max().item())

        if "attn_edge_type" in batch:
            e = batch["attn_edge_type"]
            print("attn_edge_type:", e.shape, e.min().item(), e.max().item())

        if "edge_input" in batch:
            ei = batch["edge_input"]
            print("edge_input:", ei.shape, ei.min().item(), ei.max().item())

        if "in_degree" in batch:
            d = batch["in_degree"]
            print("in_degree:", d.shape, d.min().item(), d.max().item())

    def get_config(self) -> dict:
        return {
            "num_timesteps": self.num_timesteps,
            "atom_mask_token": self.atom_mask_token,
            "bond_mask_token": self.bond_mask_token,
            "atom_pad_token": self.atom_pad_token,
            "bond_pad_token": self.bond_pad_token,
            "atom_loss_weight": self.atom_loss_weight,
            "bond_loss_weight": self.bond_loss_weight,
        }