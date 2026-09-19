import torch


def get_item(item, key):
    if isinstance(item, dict):
        return item[key]
    return getattr(item, key)


def has_item(item, key):
    if isinstance(item, dict):
        return key in item
    return hasattr(item, key)


# ============================================================
# Graphormer input padding
# These use +1 because pad id = 0.
# ============================================================

def pad_1d_input(x, padlen):
    x = x + 1
    xlen = x.size(0)

    if xlen < padlen:
        out = x.new_zeros((padlen,), dtype=x.dtype)
        out[:xlen] = x
        x = out

    return x


def pad_2d_input(x, padlen):
    x = x + 1
    xlen, xdim = x.size()

    if xlen < padlen:
        out = x.new_zeros((padlen, xdim), dtype=x.dtype)
        out[:xlen, :] = x
        x = out

    return x


def pad_spatial_pos_input(x, padlen):
    x = x + 1
    xlen = x.size(0)

    if xlen < padlen:
        out = x.new_zeros((padlen, padlen), dtype=x.dtype)
        out[:xlen, :xlen] = x
        x = out

    return x


def pad_3d_input(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()

    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        out = x.new_zeros(
            (padlen1, padlen2, padlen3, xlen4),
            dtype=x.dtype,
        )
        out[:xlen1, :xlen2, :xlen3, :] = x
        x = out

    return x


def pad_attn_bias(x, padlen):
    xlen = x.size(0)

    if xlen < padlen:
        out = x.new_full(
            (padlen, padlen),
            float("-inf"),
            dtype=x.dtype,
        )
        out[:xlen, :xlen] = x
        out[xlen:, :xlen] = 0
        x = out

    return x


def pad_attn_edge_type(x, padlen):
    xlen = x.size(0)

    if xlen < padlen:
        out = x.new_zeros(
            (padlen, padlen, x.size(-1)),
            dtype=x.dtype,
        )
        out[:xlen, :xlen, :] = x
        x = out

    return x


# ============================================================
# Diffusion target padding
# These do NOT use +1.
# 0 already means PAD / NO_BOND.
# ============================================================

def pad_atom_types_target(x, padlen):
    xlen = x.size(0)

    if xlen < padlen:
        out = x.new_zeros((padlen,), dtype=x.dtype)
        out[:xlen] = x
        x = out

    return x


def pad_bond_types_target(x, padlen):
    xlen = x.size(0)

    if xlen < padlen:
        out = x.new_zeros((padlen, padlen), dtype=x.dtype)
        out[:xlen, :xlen] = x
        x = out

    return x


# ============================================================
# Collator
# ============================================================

def graphormer_collate_fn(
    batch,
    max_nodes=512,
    multi_hop_max_dist=20,
    spatial_pos_max=20,
):
    batch = [
        item
        for item in batch
        if item is not None and get_item(item, "x").size(0) <= max_nodes
    ]

    if len(batch) == 0:
        return None

    for item in batch:
        item.edge_input = get_item(item, "edge_input")[:, :, :multi_hop_max_dist, :]

        item.attn_bias[1:, 1:][
            get_item(item, "spatial_pos") >= spatial_pos_max
        ] = float("-inf")

    max_node_num = max(get_item(item, "x").size(0) for item in batch)

    max_dist = min(
        multi_hop_max_dist,
        max(get_item(item, "edge_input").size(-2) for item in batch),
    )

    collated = {}

    collated["x"] = torch.stack([
        pad_2d_input(get_item(item, "x"), max_node_num)
        for item in batch
    ])

    collated["edge_input"] = torch.stack([
        pad_3d_input(
            get_item(item, "edge_input"),
            max_node_num,
            max_node_num,
            max_dist,
        )
        for item in batch
    ])

    collated["attn_bias"] = torch.stack([
        pad_attn_bias(
            get_item(item, "attn_bias"),
            max_node_num + 1,
        )
        for item in batch
    ])

    collated["attn_edge_type"] = torch.stack([
        pad_attn_edge_type(
            get_item(item, "attn_edge_type"),
            max_node_num,
        )
        for item in batch
    ])

    collated["spatial_pos"] = torch.stack([
        pad_spatial_pos_input(
            get_item(item, "spatial_pos"),
            max_node_num,
        )
        for item in batch
    ])

    collated["in_degree"] = torch.stack([
        pad_1d_input(
            get_item(item, "in_degree"),
            max_node_num,
        )
        for item in batch
    ])

    collated["out_degree"] = torch.stack([
        pad_1d_input(
            get_item(item, "out_degree"),
            max_node_num,
        )
        for item in batch
    ])

    collated["atom_types"] = torch.stack([
        pad_atom_types_target(
            get_item(item, "atom_types").long(),
            max_node_num,
        )
        for item in batch
    ])

    collated["bond_types"] = torch.stack([
        pad_bond_types_target(
            get_item(item, "bond_types").long(),
            max_node_num,
        )
        for item in batch
    ])

    node_mask = torch.zeros(
        len(batch),
        max_node_num,
        dtype=torch.bool,
    )

    for i, item in enumerate(batch):
        n = get_item(item, "x").size(0)
        node_mask[i, :n] = True

    collated["node_mask"] = node_mask

    if has_item(batch[0], "smiles"):
        collated["smiles"] = [
            get_item(item, "smiles")
            for item in batch
        ]

    if has_item(batch[0], "y"):
        ys = [get_item(item, "y") for item in batch]

        if all(y is not None for y in ys):
            collated["y"] = torch.stack([
                y if torch.is_tensor(y) else torch.tensor(y)
                for y in ys
            ])
        else:
            collated["y"] = None

    return collated