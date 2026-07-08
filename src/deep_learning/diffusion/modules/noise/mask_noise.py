# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import torch


def convert_to_single_emb(x: torch.Tensor, offset: int = 512) -> torch.Tensor:
    feature_num = x.size(-1)

    feature_offset = 1 + torch.arange(
        0,
        feature_num * offset,
        offset,
        dtype=torch.long,
        device=x.device,
    )

    return x + feature_offset


def mask_diffusion_batch(
    batch: dict[str, torch.Tensor],
    num_timesteps: int,
    atom_mask_token: int,
    bond_mask_token: int,
    atom_pad_token: int = 0,
    bond_no_bond_token: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Create a noisy molecular graph and rebuild Graphormer inputs from it.

    Clean targets:
        atom_types: [B, max_nodes]
            0 = PAD
            1~118 = atomic numbers
            119 = misc

        bond_types: [B, max_nodes, max_nodes]
            0 = NO_BOND
            1 = SINGLE
            2 = DOUBLE
            3 = TRIPLE
            4 = AROMATIC
            5 = misc

    Noisy inputs:
        masked atoms use atom_mask_token
        masked bonds use bond_mask_token
    """

    clean_atom_types = batch["atom_types"].long()
    clean_bond_types = batch["bond_types"].long()

    device = clean_atom_types.device
    B, N = clean_atom_types.shape

    if "node_mask" in batch:
        node_mask = batch["node_mask"].bool()
    else:
        node_mask = clean_atom_types.ne(atom_pad_token)  # shape: (B, N)
    # print("node_mask:", node_mask.shape, node_mask.sum())

    # Sample a random timestep for each graph in the batch.
    t = torch.randint(
        low=1,
        high=num_timesteps + 1,
        size=(B,),
        device=device,
        dtype=torch.long,
    )

    # Compute the noise probability for each graph based on its timestep.
    # Larger timesteps correspond to more noise, so we scale linearly.
    noise_prob = t.float() / float(num_timesteps) # shape: (B,)

    # Randomly choose atoms and bonds to corrupt based on the noise probability.
    atom_random = torch.rand(B, N, device=device)  # shape: (B, N)
    atom_mask = atom_random < noise_prob[:, None]  # shape: (B, N)
    atom_mask = atom_mask & node_mask  # Only valid nodes can be corrupted. Paddings should not be corrupted.

    # For bonds, we need to ensure that we only corrupt valid edges between valid nodes.
    valid_edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

    eye = torch.eye(N, dtype=torch.bool, device=device)
    valid_edge_mask = valid_edge_mask & (~eye.unsqueeze(0))

    # bond_random = torch.rand(B, N, N, device=device)
    # bond_mask = bond_random < noise_prob[:, None, None]
    # bond_mask = bond_mask & valid_edge_mask

    # # Force symmetric bond mask.
    # bond_mask = bond_mask | bond_mask.transpose(1, 2)
    # =================================================================
    """
    Extract valid edges and sample real bonds and no-bonds separately 
    to ensure a balanced representation of positive and negative samples 
    in the noisy graph. This approach helps maintain the structural integrity 
    of the molecular graph while introducing noise for training purposes.
    """
    # valid edge only, no diagonal
    valid_edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

    eye = torch.eye(N, dtype=torch.bool, device=device)
    valid_edge_mask = valid_edge_mask & (~eye.unsqueeze(0))

    # only sample upper triangle first
    upper = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=device),
        diagonal=1,
    ).unsqueeze(0)

    valid_edge_mask = valid_edge_mask & upper

    # positive = real bonds
    real_bond_edge_mask = clean_bond_types.gt(bond_no_bond_token) & valid_edge_mask

    # negative = no bonds
    no_bond_edge_mask = clean_bond_types.eq(bond_no_bond_token) & valid_edge_mask

    # sample real bonds according to diffusion noise probability
    real_random = torch.rand(B, N, N, device=device)
    real_bond_mask = (
        real_random < noise_prob[:, None, None]
    ) & real_bond_edge_mask

    negative_ratio = 1.0
    real_count = real_bond_mask.sum(dim=(1, 2))

    no_bond_mask = torch.zeros_like(real_bond_mask)

    for b in range(B):
        num_real = int(real_count[b].item())

        if num_real == 0:
            continue

        candidates = no_bond_edge_mask[b].nonzero(as_tuple=False)

        if candidates.numel() == 0:
            continue

        num_negative = min(
            candidates.size(0),
            max(1, round(num_real * negative_ratio)),
        )

        perm = torch.randperm(candidates.size(0), device=device)
        selected = candidates[perm[:num_negative]]

        no_bond_mask[b, selected[:, 0], selected[:, 1]] = True

    # combine positive and negative masks
    bond_mask_upper = real_bond_mask | no_bond_mask

    # make symmetric. bond_mask is selected bond and no-bond edges for training
    bond_mask = bond_mask_upper | bond_mask_upper.transpose(1, 2)
    # ===============================================================

    noisy_atom_types = clean_atom_types.clone()
    noisy_bond_types = clean_bond_types.clone()

    noisy_atom_types[atom_mask] = atom_mask_token
    noisy_bond_types[bond_mask] = bond_mask_token

    # No self bonds.
    noisy_bond_types[:, eye] = bond_no_bond_token

    noisy_batch = dict(batch)
    noisy_batch["noisy_atom_types"] = noisy_atom_types
    noisy_batch["noisy_bond_types"] = noisy_bond_types
    noisy_batch["atom_mask"] = atom_mask
    noisy_batch["bond_mask"] = bond_mask
    noisy_batch["t"] = t
    noisy_batch["timestep"] = t

    graphormer_inputs = rebuild_graphormer_inputs_from_types(
        atom_types=noisy_atom_types,
        bond_types=noisy_bond_types,
        node_mask=node_mask,
        atom_mask_token=atom_mask_token,
        bond_mask_token=bond_mask_token,
        bond_no_bond_token=bond_no_bond_token,
        spatial_pos_max=20,
        multi_hop_max_dist=_infer_multi_hop_max_dist(batch),
    )

    noisy_batch.update(graphormer_inputs)

    return noisy_batch


def rebuild_graphormer_inputs_from_types(
    atom_types: torch.Tensor,
    bond_types: torch.Tensor,
    node_mask: torch.Tensor,
    atom_mask_token: int,
    bond_mask_token: int,
    bond_no_bond_token: int = 0,
    spatial_pos_max: int = 20,
    multi_hop_max_dist: int = 5,
) -> dict[str, torch.Tensor]:
    """
    Rebuild Graphormer-style inputs from atom_types and bond_types.
    This is used by both training noise and sampling.
    """

    device = atom_types.device
    B, N = atom_types.shape

    x = build_noisy_atom_features(
        atom_types=atom_types,
        node_mask=node_mask,
        atom_mask_token=atom_mask_token,
    )

    attn_bias = torch.zeros(
        B,
        N + 1,
        N + 1,
        dtype=torch.float32,
        device=device,
    )   # shape: (B, N+1, N+1), including the virtual node

    known_bond_mask = (
        bond_types.ne(bond_no_bond_token)
        & bond_types.ne(bond_mask_token)
        & node_mask.unsqueeze(1)
        & node_mask.unsqueeze(2)
    )

    eye = torch.eye(N, dtype=torch.bool, device=device)
    known_bond_mask[:, eye] = False

    degree = known_bond_mask.long().sum(dim=-1)
    degree = degree.clamp(min=0, max=511)

    spatial_pos = build_spatial_pos_from_bonds(
        known_bond_mask=known_bond_mask,
        node_mask=node_mask,
        spatial_pos_max=spatial_pos_max,
    )

    attn_edge_type = build_attn_edge_type(
        bond_types=bond_types,
        bond_mask_token=bond_mask_token,
        bond_no_bond_token=bond_no_bond_token,
    )

    edge_input = build_edge_input(
        bond_types=bond_types,
        bond_mask_token=bond_mask_token,
        bond_no_bond_token=bond_no_bond_token,
        multi_hop_max_dist=multi_hop_max_dist,
    )

    return {
        "x": x,
        "node_feat": x,
        "attn_bias": attn_bias,
        "attn_edge_type": attn_edge_type,
        "spatial_pos": spatial_pos,
        "in_degree": degree,
        "out_degree": degree,
        "edge_input": edge_input,
        "atom_types": atom_types,
        "bond_types": bond_types,
        "node_mask": node_mask,
    }


def build_noisy_atom_features(
    atom_types: torch.Tensor,
    node_mask: torch.Tensor,
    atom_mask_token: int,
) -> torch.Tensor:
    
    """
    atom_types: [B, N], including PAD and MISC
    node_mask: [B, N], True for valid nodes, False for padding
    atom_mask_token: int = 119, the token used for masked atoms

    Build Graphormer x: [B, N, 9].  9 is the number of atom features used in Graphormer.
    The features are:
    0: atomic number index (1~118 for atomic numbers 1~118, 119 for misc, 120 for mask)
    1: chirality
    2: degree
    3: formal charge
    4: numH
    5: radical electrons
    6: hybridization
    7: aromatic
    8: ring

    The returned x is already converted with convert_to_single_emb(),
    matching Graphormer preprocessing.
    atom_types: [B, N], including PAD and MISC
    node_mask: [B, N], True for valid nodes, False for padding
    atom_mask_token: int = 119, the token used for masked atoms
    """

    device = atom_types.device
    B, N = atom_types.shape 

    raw_x = torch.zeros(B, N, 9, dtype=torch.long, device=device)

    # Feature 0: atomic number index.
    # clean atom:
    #   atomic number 1 -> raw index 0
    #   atomic number 6 -> raw index 5
    #   atomic number 118 -> raw index 117
    # misc -> 118
    # mask -> 119
    atomic_feature = torch.full_like(atom_types, 118)  # shape: (B, N), default to misc

    real_atom_mask = atom_types.ge(1) & atom_types.le(118)  # Valid atomic numbers
    atomic_feature[real_atom_mask] = atom_types[real_atom_mask] - 1

    atomic_feature[atom_types.eq(atom_mask_token)] = 119

    raw_x[:, :, 0] = atomic_feature

    # Feature 1: chirality, default CHI_UNSPECIFIED = 0
    raw_x[:, :, 1] = 0

    # Feature 2: degree, approximate later from graph if needed.
    raw_x[:, :, 2] = 0

    # Feature 3: formal charge, index of 0 charge in [-5..5, misc] = 5
    raw_x[:, :, 3] = 5

    # Feature 4: numH, default 0
    raw_x[:, :, 4] = 0

    # Feature 5: radical electrons, default 0
    raw_x[:, :, 5] = 0

    # Feature 6: hybridization, default SP3 index = 2
    raw_x[:, :, 6] = 2

    # Feature 7: aromatic false = 0
    raw_x[:, :, 7] = 0

    # Feature 8: ring false = 0
    raw_x[:, :, 8] = 0

    x = convert_to_single_emb(raw_x)

    # Padding nodes should stay 0.
    x = x.masked_fill(~node_mask.unsqueeze(-1), 0)

    return x.long()


def build_attn_edge_type(
    bond_types: torch.Tensor,
    bond_mask_token: int,
    bond_no_bond_token: int = 0,
) -> torch.Tensor:
    """
    Build attn_edge_type: [B, N, N, 3].

    For real bonds:
        bond_type 1~5 -> raw 0~4

    For masked bond:
        bond_mask_token -> raw 5

    For no bond:
        all zeros
    """

    device = bond_types.device
    B, N, _ = bond_types.shape

    raw_edge = torch.zeros(B, N, N, 3, dtype=torch.long, device=device)

    real_bond_mask = (
        bond_types.ne(bond_no_bond_token)
        & bond_types.ne(bond_mask_token)
    )

    raw_edge[..., 0][real_bond_mask] = bond_types[real_bond_mask] - 1

    mask_bond_mask = bond_types.eq(bond_mask_token)
    raw_edge[..., 0][mask_bond_mask] = 5

    attn_edge_type = torch.zeros_like(raw_edge)

    nonzero_edge_mask = real_bond_mask | mask_bond_mask

    if nonzero_edge_mask.any():
        converted = convert_to_single_emb(raw_edge) + 1
        attn_edge_type[nonzero_edge_mask] = converted[nonzero_edge_mask]

    return attn_edge_type.long()


def build_edge_input(
    bond_types: torch.Tensor,
    bond_mask_token: int,
    bond_no_bond_token: int = 0,
    multi_hop_max_dist: int = 5,
) -> torch.Tensor:
    """
    Build edge_input: [B, N, N, multi_hop_max_dist, 3].

    Your current Graphormer implementation uses edge_input directly
    without convert_to_single_emb, so we keep raw edge features here.
    """

    device = bond_types.device
    B, N, _ = bond_types.shape

    edge_input = torch.zeros(
        B,
        N,
        N,
        multi_hop_max_dist,
        3,
        dtype=torch.long,
        device=device,
    )

    real_bond_mask = (
        bond_types.ne(bond_no_bond_token)
        & bond_types.ne(bond_mask_token)
    )

    edge_input[..., 0, 0][real_bond_mask] = bond_types[real_bond_mask] - 1

    mask_bond_mask = bond_types.eq(bond_mask_token)
    edge_input[..., 0, 0][mask_bond_mask] = 5

    return edge_input.long()


def build_spatial_pos_from_bonds(
    known_bond_mask: torch.Tensor,
    node_mask: torch.Tensor,
    spatial_pos_max: int = 20,
) -> torch.Tensor:
    """
    Compute shortest-path distance from currently known bonds.

    Disconnected pairs are assigned spatial_pos_max.
    """

    device = known_bond_mask.device
    B, N, _ = known_bond_mask.shape

    spatial = torch.full(
        (B, N, N),
        spatial_pos_max,
        dtype=torch.long,
        device=device,
    )

    eye = torch.eye(N, dtype=torch.bool, device=device)

    for b in range(B):
        valid_nodes = node_mask[b]

        dist = torch.full(
            (N, N),
            spatial_pos_max,
            dtype=torch.long,
            device=device,
        )

        dist[eye] = 0
        dist[known_bond_mask[b]] = 1

        # Floyd-Warshall for small molecular graphs.
        for k in range(N):
            dist = torch.minimum(
                dist,
                dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0),
            )

        valid_pair_mask = valid_nodes.unsqueeze(0) & valid_nodes.unsqueeze(1)
        dist = dist.clamp(min=0, max=spatial_pos_max)
        dist = torch.where(valid_pair_mask, dist, torch.zeros_like(dist))

        spatial[b] = dist

    return spatial.long()


def _infer_multi_hop_max_dist(batch: dict[str, torch.Tensor]) -> int:
    if "edge_input" in batch and torch.is_tensor(batch["edge_input"]):
        if batch["edge_input"].dim() == 5:
            return int(batch["edge_input"].size(3))

    return 5