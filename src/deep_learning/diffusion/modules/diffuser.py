# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.deep_learning.diffusion.modules.noise.mask_noise import (
    mask_diffusion_batch,
)


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
        negative_sampling_ratio: float = 1.0,
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
        self.negative_sampling_ratio = negative_sampling_ratio

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

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
            bond_targets = clean_bond_types[bond_mask]  
            masked_bond_logits = bond_logits[bond_mask]

            positive_mask = bond_targets > 0
            negative_mask = bond_targets == 0

            # Extract the indices of positive and negative samples
            positive_index = positive_mask.nonzero(as_tuple=True)[0]
            negative_index = negative_mask.nonzero(as_tuple=True)[0]

            # Count the number of positive and negative samples
            num_positive = positive_index.numel()
            num_negative = negative_index.numel()

            if num_positive > 0 and num_negative > 0:
                num_negative_sample = int(
                    min(
                        num_negative,
                        max(1, round(float(num_positive) * self.negative_sampling_ratio)),
                    )
                )

                perm = torch.randperm(
                    num_negative,
                    device=negative_index.device,
                )

                sampled_negative_index = negative_index[
                    perm[:num_negative_sample]
                ]

                selected_index = torch.cat(
                    [
                        positive_index,
                        sampled_negative_index,
                    ],
                    dim=0,
                )

                bond_loss = F.cross_entropy(
                    masked_bond_logits[selected_index],
                    bond_targets[selected_index],
                )

            else:
                bond_loss = F.cross_entropy(
                    masked_bond_logits,
                    bond_targets,
                )

            # Index of the predicted bond types
            bond_pred = masked_bond_logits.argmax(dim=-1)

            # Compute bond accuracy
            bond_acc = (bond_pred == bond_targets).float().mean()

            real_bond_mask = bond_targets > 0

            if real_bond_mask.any():
                real_bond_acc = (bond_pred[real_bond_mask]== bond_targets[real_bond_mask]).float().mean()
            else:
                real_bond_acc = bond_logits.sum() * 0.0

            no_bond_ratio = (bond_targets == 0).float().mean()

            pred_no_bond_ratio = (bond_pred == 0).float().mean()

        else:
            bond_loss = bond_logits.sum() * 0.0
            bond_acc = bond_logits.sum() * 0.0
            real_bond_acc = bond_logits.sum() * 0.0
            no_bond_ratio = bond_logits.sum() * 0.0
            pred_no_bond_ratio = bond_logits.sum() * 0.0

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
            "bond_acc": bond_acc,
            "real_bond_acc": real_bond_acc,
            "no_bond_ratio": no_bond_ratio,
            "pred_no_bond_ratio": pred_no_bond_ratio,
        }

    def _check_input(self, batch):
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
            "negative_sampling_ratio": self.negative_sampling_ratio,
        }