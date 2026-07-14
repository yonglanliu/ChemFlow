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
    target_list: list[Any] | None = None,
) -> list[Data]:
    if target_list is not None and len(smiles_list) != len(target_list):
        raise ValueError(
            "smiles_list and target_list must have the same length: "
            f"{len(smiles_list)} vs {len(target_list)}."
        )

    data_list: list[Data] = []
    for index, smiles in enumerate(smiles_list):
        target = target_list[index] if target_list is not None else None
        data = featurize_smiles(smiles, featurizer)
        if data is None:
            continue
        if target is not None:
            data.y = torch.as_tensor(target)
        data_list.append(data)
    return data_list


def _filter_smiles_and_targets(
    raw_smiles_batch: list[Any],
    raw_target_batch: list[Any] | None,
) -> tuple[list[str], list[Any] | None]:
    if raw_target_batch is None:
        smiles_batch = [
            str(smiles).strip()
            for smiles in raw_smiles_batch
            if smiles is not None and str(smiles).strip()
        ]
        return smiles_batch, None

    if len(raw_smiles_batch) != len(raw_target_batch):
        raise ValueError(
            "Raw SMILES and target batches have different lengths: "
            f"{len(raw_smiles_batch)} vs {len(raw_target_batch)}."
        )

    filtered_pairs = [
        (str(smiles).strip(), target)
        for smiles, target in zip(raw_smiles_batch, raw_target_batch)
        if smiles is not None and str(smiles).strip()
    ]
    return (
        [smiles for smiles, _ in filtered_pairs],
        [target for _, target in filtered_pairs],
    )


def _shuffle_aligned(
    smiles_batch: list[str],
    target_batch: list[Any] | None,
    rng: np.random.Generator,
) -> tuple[list[str], list[Any] | None]:
    indices = rng.permutation(len(smiles_batch))
    shuffled_smiles = [smiles_batch[i] for i in indices]
    if target_batch is None:
        return shuffled_smiles, None
    return shuffled_smiles, [target_batch[i] for i in indices]


def _split_batch(
    smiles_batch: list[str],
    target_batch: list[Any] | None,
    val_fraction: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> dict[str, dict[str, list[Any] | None]]:
    random_values = rng.random(len(smiles_batch))
    is_test = random_values < test_fraction
    is_val = (
        (random_values >= test_fraction)
        & (random_values < test_fraction + val_fraction)
    )

    train_smiles: list[str] = []
    val_smiles: list[str] = []
    test_smiles: list[str] = []
    train_targets = [] if target_batch is not None else None
    val_targets = [] if target_batch is not None else None
    test_targets = [] if target_batch is not None else None

    for index, smiles in enumerate(smiles_batch):
        target = target_batch[index] if target_batch is not None else None
        if is_test[index]:
            test_smiles.append(smiles)
            if test_targets is not None:
                test_targets.append(target)
        elif is_val[index]:
            val_smiles.append(smiles)
            if val_targets is not None:
                val_targets.append(target)
        else:
            train_smiles.append(smiles)
            if train_targets is not None:
                train_targets.append(target)

    if train_targets is not None and len(train_smiles) != len(train_targets):
        raise RuntimeError("Train SMILES and targets became misaligned.")
    if val_targets is not None and len(val_smiles) != len(val_targets):
        raise RuntimeError("Validation SMILES and targets became misaligned.")
    if test_targets is not None and len(test_smiles) != len(test_targets):
        raise RuntimeError("Test SMILES and targets became misaligned.")
    if len(train_smiles) + len(val_smiles) + len(test_smiles) != len(smiles_batch):
        raise RuntimeError("Split sizes do not sum to the original batch size.")

    return {
        "train": {"smiles": train_smiles, "targets": train_targets},
        "val": {"smiles": val_smiles, "targets": val_targets},
        "test": {"smiles": test_smiles, "targets": test_targets},
    }


def _save_shard(
    split_name: str,
    smiles_list: list[str],
    target_list: list[Any] | None,
    featurizer: GraphormerFeaturizer,
    split_cache_dir: Path,
    shard_index: int,
) -> tuple[str | None, int]:
    if not smiles_list:
        return None, 0

    data_list = featurize_smiles_list(
        smiles_list=smiles_list,
        featurizer=featurizer,
        target_list=target_list,
    )
    if not data_list:
        return None, 0

    shard_path = split_cache_dir / f"{split_name}_{shard_index:06d}.pt"
    torch.save(data_list, shard_path)
    return str(shard_path), len(data_list)


def featurize_and_cache_dataset(
    dataset_config: Any,
    featurizer: GraphormerFeaturizer,
    cache_dir: str | Path,
) -> dict[str, Any]:
    dataset_path = Path(dataset_config.dataset_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()

    manifest_path = cache_dir / "graphormer_manifest.pt"
    train_cache_dir = cache_dir / "train"
    val_cache_dir = cache_dir / "val"
    test_cache_dir = cache_dir / "test"

    if manifest_path.exists():
        print(f"Loading Graphormer cache manifest from {manifest_path}")
        return safe_torch_load(manifest_path, map_location="cpu")

    smiles_column = dataset_config.smiles_column
    target_column = getattr(dataset_config, "target_column", None)
    val_fraction = float(getattr(dataset_config, "val_fraction", 0.1))
    test_fraction = float(getattr(dataset_config, "test_fraction", 0.0))

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)
    if test_fraction > 0.0:
        test_cache_dir.mkdir(parents=True, exist_ok=True)

    seed = int(getattr(dataset_config, "seed", 42))
    batch_size = int(getattr(dataset_config, "preprocess_batch_size", 100_000))
    multi_hop_max_dist = getattr(dataset_config, "multi_hop_max_dist", None)
    spatial_pos_max = getattr(dataset_config, "spatial_pos_max", None)
    max_nodes = getattr(dataset_config, "max_nodes", None)
    remove_hs = getattr(dataset_config, "remove_hs", True)
    reorder_atoms = getattr(dataset_config, "reorder_atoms", False)

    rng = np.random.default_rng(seed)

    train_shards: list[str] = []
    val_shards: list[str] = []
    test_shards: list[str] = []
    train_idx = val_idx = test_idx = 0
    total_count = valid_count = 0
    split_raw_counts = {"train": 0, "val": 0, "test": 0}
    split_valid_counts = {"train": 0, "val": 0, "test": 0}

    print(f"Building Graphormer cache from {dataset_path}")

    for batch_idx, batch in enumerate(
        _iter_smiles_batches(
            dataset_path=dataset_path,
            smiles_column=smiles_column,
            batch_size=batch_size,
            target_column=target_column,
        )
    ):
        raw_smiles_batch = list(batch[0])
        raw_target_batch = list(batch[1]) if target_column is not None else None

        smiles_batch, target_batch = _filter_smiles_and_targets(
            raw_smiles_batch,
            raw_target_batch,
        )
        if not smiles_batch:
            continue

        smiles_batch, target_batch = _shuffle_aligned(
            smiles_batch,
            target_batch,
            rng,
        )
        split_data = _split_batch(
            smiles_batch,
            target_batch,
            val_fraction,
            test_fraction,
            rng,
        )

        total_count += len(smiles_batch)

        for split_name in ("train", "val", "test"):
            split_raw_counts[split_name] += len(split_data[split_name]["smiles"])

        train_path, train_valid = _save_shard(
            "train",
            split_data["train"]["smiles"],
            split_data["train"]["targets"],
            featurizer,
            train_cache_dir,
            train_idx,
        )
        if train_path is not None:
            train_shards.append(train_path)
            train_idx += 1
            valid_count += train_valid
            split_valid_counts["train"] += train_valid

        val_path, val_valid = _save_shard(
            "val",
            split_data["val"]["smiles"],
            split_data["val"]["targets"],
            featurizer,
            val_cache_dir,
            val_idx,
        )
        if val_path is not None:
            val_shards.append(val_path)
            val_idx += 1
            valid_count += val_valid
            split_valid_counts["val"] += val_valid

        if test_fraction > 0.0:
            test_path, test_valid = _save_shard(
                "test",
                split_data["test"]["smiles"],
                split_data["test"]["targets"],
                featurizer,
                test_cache_dir,
                test_idx,
            )
            if test_path is not None:
                test_shards.append(test_path)
                test_idx += 1
                valid_count += test_valid
                split_valid_counts["test"] += test_valid

        print(
            f"Processed batch {batch_idx + 1} | "
            f"total={total_count:,} | valid={valid_count:,} | "
            f"train_shards={len(train_shards)} | "
            f"val_shards={len(val_shards)} | "
            f"test_shards={len(test_shards)}"
        )

    manifest: dict[str, Any] = {
        "train": train_shards,
        "val": val_shards,
        "test": test_shards,
        "dataset_path": str(dataset_path),
        "smiles_column": smiles_column,
        "target_column": target_column,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "total_count": total_count,
        "valid_count": valid_count,
        "split_raw_counts": split_raw_counts,
        "split_valid_counts": split_valid_counts,
        "max_nodes": max_nodes,
        "multi_hop_max_dist": multi_hop_max_dist,
        "spatial_pos_max": spatial_pos_max,
        "remove_hs": remove_hs,
        "reorder_atoms": reorder_atoms,
        "seed": seed,
        "preprocess_batch_size": batch_size,
    }

    torch.save(manifest, manifest_path)
    print(f"Saved Graphormer cache manifest to {manifest_path}")
    return manifest


class GraphormerMoleculeDataset(Dataset):
    def __init__(
        self,
        shard_paths: list[str | Path],
        max_nodes: int = 128,
        multi_hop_max_dist: int = 20,
        spatial_pos_max: int = 20,
    ) -> None:
        self.shard_paths = [Path(p).expanduser().resolve() for p in shard_paths]
        self.data_list: list[Data] = []
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

        for path in self.shard_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Shard does not exist: {path}")
            shard = safe_torch_load(path, map_location="cpu")
            if not isinstance(shard, list):
                raise TypeError(f"Expected shard to be list[Data], got {type(shard)}")
            for item in shard:
                if not isinstance(item, Data):
                    raise TypeError(
                        "Expected every shard item to be torch_geometric.data.Data, "
                        f"got {type(item)}"
                    )
            self.data_list.extend(shard)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]
