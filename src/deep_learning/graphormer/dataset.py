from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.deep_learning.gpt.dataset import _iter_smiles_batches
from src.deep_learning.graphormer import GraphormerFeaturizer


def safe_torch_load(path: str | Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def featurize_smiles(
    smiles: str,
    featurizer: GraphormerFeaturizer,
) -> Data | None:
    try:
        data = featurizer(smiles)

        if isinstance(data, dict):
            data = Data(**data)

        if not isinstance(data, Data):
            raise TypeError(f"Unexpected featurizer output: {type(data)}")

        return data

    except Exception:
        return None


def featurize_smiles_list(
    smiles_list: list[str],
    featurizer: GraphormerFeaturizer,
    max_nodes: int | None = None,
) -> list[Data]:
    data_list: list[Data] = []

    for smi in smiles_list:
        data = featurize_smiles(smi, featurizer)

        if data is None:
            continue

        num_nodes = int(data.num_nodes)

        if max_nodes is not None and num_nodes > max_nodes:
            continue

        data_list.append(data)

    return data_list


def update_dataset_stats(
    data_list: list[Data],
    real_max_num_nodes: int,
    real_max_spatial_dist: int,
) -> tuple[int, int]:
    for data in data_list:
        real_max_num_nodes = max(real_max_num_nodes, int(data.num_nodes))

        if hasattr(data, "spatial_pos"):
            real_max_spatial_dist = max(
                real_max_spatial_dist,
                int(data.spatial_pos.max().item()),
            )

    return real_max_num_nodes, real_max_spatial_dist


def featurize_and_cache_dataset(
    dataset_config,
    featurizer: GraphormerFeaturizer,
    cache_dir,
):
    dataset_path = Path(dataset_config.dataset_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()

    manifest_path = cache_dir / "graphormer_manifest.pt"
    train_cache_dir = cache_dir / "train"
    val_cache_dir = cache_dir / "val"

    if manifest_path.exists():
        print(f"Loading Graphormer cache manifest from {manifest_path}")
        return safe_torch_load(manifest_path, map_location="cpu")

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    train_shards: list[str] = []
    val_shards: list[str] = []

    smiles_column = dataset_config.smiles_column
    val_fraction = getattr(dataset_config, "val_fraction", 0.1)
    seed = getattr(dataset_config, "seed", 42)
    batch_size = getattr(dataset_config, "preprocess_batch_size", 100_000)

    max_node = getattr(dataset_config, "max_node", None)
    multi_hop_max_dist = getattr(dataset_config, "multi_hop_max_dist", None)
    spatial_pos_max = getattr(dataset_config, "spatial_pos_max", None)
    remove_hs = getattr(dataset_config, "remove_hs", True)
    reorder_atoms = getattr(dataset_config, "reorder_atoms", False)

    rng = np.random.default_rng(seed)

    real_max_spatial_dist = 0
    real_max_num_nodes = 0

    skipped_too_large = 0
    train_idx = 0
    val_idx = 0
    total_count = 0
    valid_count = 0

    print(f"Building Graphormer cache from {dataset_path}")

    for batch_idx, smiles_batch in enumerate(
        _iter_smiles_batches(
            dataset_path=dataset_path,
            smiles_column=smiles_column,
            batch_size=batch_size,
        )
    ):
        smiles_batch = [
            str(s).strip()
            for s in smiles_batch
            if s and str(s).strip()
        ]

        if not smiles_batch:
            continue

        rng.shuffle(smiles_batch)
        is_val = rng.random(len(smiles_batch)) < val_fraction

        train_smiles = [
            smi for smi, flag in zip(smiles_batch, is_val)
            if not flag
        ]

        val_smiles = [
            smi for smi, flag in zip(smiles_batch, is_val)
            if flag
        ]

        if train_smiles:
            raw_count = len(train_smiles)

            train_data_list = featurize_smiles_list(
                train_smiles,
                featurizer=featurizer,
                max_nodes=max_node,
            )

            skipped_too_large += raw_count - len(train_data_list)

            if train_data_list:
                real_max_num_nodes, real_max_spatial_dist = update_dataset_stats(
                    train_data_list,
                    real_max_num_nodes,
                    real_max_spatial_dist,
                )

                train_path = train_cache_dir / f"train_{train_idx:06d}.pt"
                torch.save(train_data_list, train_path)

                train_shards.append(str(train_path))
                train_idx += 1
                valid_count += len(train_data_list)

        if val_smiles:
            raw_count = len(val_smiles)

            val_data_list = featurize_smiles_list(
                val_smiles,
                featurizer=featurizer,
                max_nodes=max_node,
            )

            skipped_too_large += raw_count - len(val_data_list)

            if val_data_list:
                real_max_num_nodes, real_max_spatial_dist = update_dataset_stats(
                    val_data_list,
                    real_max_num_nodes,
                    real_max_spatial_dist,
                )

                val_path = val_cache_dir / f"val_{val_idx:06d}.pt"
                torch.save(val_data_list, val_path)

                val_shards.append(str(val_path))
                val_idx += 1
                valid_count += len(val_data_list)

        total_count += len(smiles_batch)

        print(
            f"Processed batch {batch_idx + 1} | "
            f"total={total_count:,} | "
            f"valid={valid_count:,} | "
            f"skipped={skipped_too_large:,} | "
            f"train_shards={len(train_shards)} | "
            f"val_shards={len(val_shards)}"
        )

    manifest = {
        "train": train_shards,
        "val": val_shards,
        "dataset_path": str(dataset_path),
        "smiles_column": smiles_column,
        "val_fraction": val_fraction,
        "total_count": total_count,
        "valid_count": valid_count,
        "skipped_too_large": skipped_too_large,
        "max_node": max_node,
        "multi_hop_max_dist": multi_hop_max_dist,
        "spatial_pos_max": spatial_pos_max,
        "real_max_num_nodes": real_max_num_nodes,
        "real_max_spatial_dist": real_max_spatial_dist,
        "remove_hs": remove_hs,
        "reorder_atoms": reorder_atoms,
    }

    torch.save(manifest, manifest_path)
    print(f"Saved Graphormer cache manifest to {manifest_path}")

    return manifest


class GraphormerMoleculeDataset(Dataset):
    def __init__(self, shard_paths: list[str | Path]):
        self.shard_paths = [
            Path(p).expanduser().resolve()
            for p in shard_paths
        ]
        self.data_list: list[Data] = []

        for path in self.shard_paths:
            shard = safe_torch_load(path, map_location="cpu")

            if not isinstance(shard, list):
                raise TypeError(
                    f"Expected shard to be list[Data], got {type(shard)}"
                )

            self.data_list.extend(shard)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]