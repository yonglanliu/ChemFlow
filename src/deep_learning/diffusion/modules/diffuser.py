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
        bond_binary_loss_weight: float = 1.0,
        bond_type_loss_weight: float = 1.0,
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
        self.bond_binary_loss_weight = bond_binary_loss_weight
        self.bond_type_loss_weight = bond_type_loss_weight

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        returns a dictionary containing the following keys:
        noisy_batch: dict[str, torch.Tensor]

        {"x": x,    # shape: (B, N, node_feat_dim)
        "node_feat": x,   # shape: (B, N, node_feat_dim)
        "attn_bias": attn_bias,  # shape: (B, N+1, N+1), including the virtual node
        "attn_edge_type": attn_edge_type,
        "spatial_pos": spatial_pos,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "edge_input": edge_input,
        "atom_types": atom_types,
        "bond_types": bond_types,
        "node_mask": node_mask,
        "atom_mask": atom_mask,
        "bond_mask": bond_mask,
        "noisy_atom_types": noisy_atom_types,
        "noisy_bond_types": noisy_bond_types,
        "timestep": t,
        "t": t
        
        }

        node_feat_dim = 9
        """

        noisy_batch = mask_diffusion_batch(
            batch=batch,
            num_timesteps=self.num_timesteps,
            atom_mask_token=self.atom_mask_token,
            bond_mask_token=self.bond_mask_token,
        )

        atom_logits, bond_exist_logits, bond_type_logits = self.denoiser(noisy_batch)

        clean_atom_types = batch["atom_types"].long()
        clean_bond_types = batch["bond_types"].long()
        node_mask = batch["node_mask"].bool()

        atom_mask = noisy_batch["atom_mask"].bool() & node_mask

        valid_edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        bond_mask = noisy_batch["bond_mask"].bool() & valid_edge_mask

        loss_dict = self.compute_loss(
            atom_logits=atom_logits,
            bond_exist_logits=bond_exist_logits,
            bond_type_logits=bond_type_logits,
            clean_atom_types=clean_atom_types,
            clean_bond_types=clean_bond_types,
            atom_mask=atom_mask,
            bond_mask=bond_mask,
        )

        return {
            **loss_dict,
            "atom_logits": atom_logits,
            "bond_exist_logits": bond_exist_logits,
            "bond_type_logits": bond_type_logits,
            "noisy_batch": noisy_batch,
        }


    def compute_loss(
        self,
        atom_logits: torch.Tensor,
        bond_exist_logits: torch.Tensor,
        bond_type_logits: torch.Tensor,
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

            masked_exist_logits = bond_exist_logits[bond_mask]
            masked_type_logits = bond_type_logits[bond_mask]

            exist_targets = (bond_targets > 0).long()
            binary_class_weight = torch.tensor(
                [1, 1],  # [no_bond, real_bond]
                device= masked_exist_logits.device,
                dtype=masked_exist_logits.dtype
            )

            binary_loss = F.cross_entropy(
                masked_exist_logits.reshape(-1, 2),
                exist_targets.reshape(-1),
                weight=binary_class_weight,
            )
            # binary_loss = F.cross_entropy(
            #     masked_exist_logits,
            #     exist_targets,
            # )

            real_bond_mask = bond_targets > 0

            if real_bond_mask.any():
                type_loss = F.cross_entropy(
                    masked_type_logits[real_bond_mask],
                    bond_targets[real_bond_mask] - 1,
                )
            else:
                type_loss = masked_type_logits.sum() * 0.0

            bond_loss = (
                self.bond_binary_loss_weight * binary_loss
                + self.bond_type_loss_weight * type_loss
            )

            exist_pred = masked_exist_logits.argmax(dim=-1)
            type_pred = masked_type_logits.argmax(dim=-1) + 1

            bond_pred = torch.zeros_like(bond_targets)
            bond_pred[exist_pred == 1] = type_pred[exist_pred == 1]

            bond_acc = (bond_pred == bond_targets).float().mean()

            if real_bond_mask.any():
                real_bond_acc = (
                    bond_pred[real_bond_mask]
                    == bond_targets[real_bond_mask]
                ).float().mean()

                real_bond_recall = (
                    bond_pred[real_bond_mask] > 0
                ).float().mean()

                pred_real_on_real_mask = real_bond_mask & (bond_pred > 0)

                if pred_real_on_real_mask.any():
                    real_bond_type_acc_when_pred_real = (
                        bond_pred[pred_real_on_real_mask]
                        == bond_targets[pred_real_on_real_mask]
                    ).float().mean()
                else:
                    real_bond_type_acc_when_pred_real = bond_exist_logits.sum() * 0.0
            else:
                real_bond_acc = bond_exist_logits.sum() * 0.0
                real_bond_recall = bond_exist_logits.sum() * 0.0
                real_bond_type_acc_when_pred_real = bond_exist_logits.sum() * 0.0

            no_bond_mask = bond_targets == 0

            if no_bond_mask.any():
                no_bond_acc = (
                    bond_pred[no_bond_mask] == 0
                ).float().mean()
            else:
                no_bond_acc = bond_exist_logits.sum() * 0.0

            no_bond_ratio = (bond_targets == 0).float().mean()
            pred_no_bond_ratio = (bond_pred == 0).float().mean()

        else:
            bond_loss = bond_exist_logits.sum() * 0.0
            binary_loss = bond_exist_logits.sum() * 0.0
            type_loss = bond_type_logits.sum() * 0.0
            bond_acc = bond_exist_logits.sum() * 0.0
            real_bond_acc = bond_exist_logits.sum() * 0.0
            real_bond_recall = bond_exist_logits.sum() * 0.0
            real_bond_type_acc_when_pred_real = bond_exist_logits.sum() * 0.0
            no_bond_acc = bond_exist_logits.sum() * 0.0
            no_bond_ratio = bond_exist_logits.sum() * 0.0
            pred_no_bond_ratio = bond_exist_logits.sum() * 0.0

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
            "bond_binary_loss": binary_loss,
            "bond_type_loss": type_loss,
            "bond_acc": bond_acc,
            "real_bond_acc": real_bond_acc,
            "real_bond_recall": real_bond_recall,
            "real_bond_type_acc_when_pred_real": real_bond_type_acc_when_pred_real,
            "no_bond_acc": no_bond_acc,
            "no_bond_ratio": no_bond_ratio,
            "pred_no_bond_ratio": pred_no_bond_ratio,
        }

    def get_config(self) -> dict:
        return {
            "num_timesteps": self.num_timesteps,
            "atom_mask_token": self.atom_mask_token,
            "bond_mask_token": self.bond_mask_token,
            "atom_pad_token": self.atom_pad_token,
            "bond_pad_token": self.bond_pad_token,
            "atom_loss_weight": self.atom_loss_weight,
            "bond_loss_weight": self.bond_loss_weight,
            "bond_binary_loss_weight": self.bond_binary_loss_weight,
            "bond_type_loss_weight": self.bond_type_loss_weight,
        }