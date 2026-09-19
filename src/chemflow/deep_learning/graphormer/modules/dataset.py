from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from chemprop import data as chemprop_data
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from chemflow.deep_learning.gpt.dataset import _iter_smiles_batches
from chemflow.deep_learning.graphormer import GraphormerFeaturizer


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
    """Featurize a list of SMILES strings for Graphormer."""

    if target_list is not None:
        if len(smiles_list) != len(target_list):
            raise ValueError(
                "smiles_list and target_list must have the same length: "
                f"{len(smiles_list)} vs {len(target_list)}."
            )

    data_list: list[Data] = []

    for index, smiles in enumerate(smiles_list):
        data = featurize_smiles(
            smiles,
            featurizer,
        )

        # Skip invalid Graphormer molecules.
        if data is None:
            continue

        # -----------------------------------------------------
        # Targets
        # -----------------------------------------------------
        if target_list is not None:
            target = target_list[index]
            data.y = torch.as_tensor(
                target,
                dtype=torch.float32,
            )

        data_list.append(data)

    return data_list

def _filter_smiles_and_targets(
    raw_smiles_batch: list[Any],
    raw_target_batch: list[Any] | None,
) -> tuple[list[str], list[Any] | None]:
    if raw_target_batch is None:
        smiles_batch = [str(smiles).strip() for smiles in raw_smiles_batch if smiles is not None and str(smiles).strip()]
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
    is_val = ((random_values >= test_fraction) & (random_values < test_fraction + val_fraction))

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

def _normalize_split_name(value: Any) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    return aliases.get(normalized, None)


def _count_task_samples(
    data_list: list[Data],
    task_names: list[str],
) -> dict[str, int]:
    counts = {task_name: 0 for task_name in task_names}

    for data in data_list:
        if not hasattr(data, "y") or data.y is None:
            continue

        y = torch.as_tensor(data.y, dtype=torch.float32).reshape(-1)

        if y.numel() != len(task_names):
            raise ValueError(
                f"Expected {len(task_names)} targets, "
                f"but got {y.numel()}."
            )

        valid_mask = ~torch.isnan(y)

        for i, task_name in enumerate(task_names):
            if valid_mask[i]:
                counts[task_name] += 1

    return counts

def _save_shard(
    split_name,
    smiles,
    targets,
    featurizer,
    cache_dir,
    shard_idx,
    task_names=None,
):
    data_list = featurize_smiles_list(
        smiles,
        featurizer,
        targets,
    )

    if not data_list:
        return None, 0, {}

    shard_path = cache_dir / f"{split_name}_{shard_idx:05d}.pt"

    torch.save(data_list, shard_path)

    task_counts = {}

    if task_names is not None:
        task_counts = _count_task_samples(data_list, task_names)

    return (str(shard_path), len(data_list), task_counts)


def _read_dataset_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(
        "Graphormer datasets must be CSV or Parquet files; "
        f"received {path}."
    )


def _dataset_cache_key(dataset_config: Any) -> str:
    paths = [Path(dataset_config.dataset_path).expanduser().resolve()]
    external = getattr(dataset_config, "test_dataset_path", None)
    if external:
        paths.append(Path(external).expanduser().resolve())

    file_state = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {path}")
        stat = path.stat()
        file_state.append(
            {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )

    target_columns = _target_columns(dataset_config)
    config_fields = (
        "smiles_column",
        "target_column",
        "split_column",
        "split_col",
        "split_type",
        "val_fraction",
        "test_fraction",
        "seed",
        "preprocess_batch_size",
        "max_nodes",
        "multi_hop_max_dist",
        "spatial_pos_max",
        "remove_hs",
        "reorder_atoms",
    )
    config_state = {
        key: repr(getattr(dataset_config, key, None)) for key in config_fields
    }
    config_state["task_names"] = repr(
        list(getattr(dataset_config, "task_names", None) or target_columns)
    )
    payload = json.dumps(
        {"schema_version": 2, "files": file_state, "config": config_state},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _target_columns(dataset_config: Any) -> list[str]:
    target_column = getattr(dataset_config, "target_column", None)
    if target_column is None:
        return []
    return [target_column] if isinstance(target_column, str) else list(target_column)


def _validated_records(
    frame: pd.DataFrame,
    dataset_config: Any,
    *,
    source: str,
    split_column: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    smiles_column = str(dataset_config.smiles_column)
    target_columns = _target_columns(dataset_config)
    required = [smiles_column, *target_columns]
    if split_column:
        required.append(split_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Dataset {source!r} is missing required columns: {missing}")

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        raw_smiles = row[smiles_column]
        smiles = "" if pd.isna(raw_smiles) else str(raw_smiles).strip()
        reason = None
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            reason = "invalid_smiles"

        targets: list[float] | None = None
        if target_columns:
            targets = []
            for column in target_columns:
                value = pd.to_numeric(
                    pd.Series([row[column]]), errors="coerce"
                ).iloc[0]
                targets.append(float(value) if np.isfinite(value) else float("nan"))
            if not any(np.isfinite(value) for value in targets):
                reason = reason or "missing_or_non_numeric_target"

        split = None
        if split_column:
            split = _normalize_split_name(row[split_column])
            if split is None:
                reason = reason or "unknown_split"

        if reason:
            rejected.append(
                {
                    "source_dataset": source,
                    "original_index": row_index,
                    "smiles": smiles,
                    "reason": reason,
                }
            )
            continue

        records.append(
            {
                "source_dataset": source,
                "original_index": row_index,
                "smiles": smiles,
                "mol": molecule,
                "targets": targets,
                "split": split,
            }
        )
    return records, rejected


def _partition_records(
    records: list[dict[str, Any]],
    dataset_config: Any,
) -> dict[str, list[dict[str, Any]]]:
    split_column = getattr(dataset_config, "split_column", None)
    split_column = getattr(dataset_config, "split_col", split_column)
    if split_column:
        return {
            name: [record for record in records if record["split"] == name]
            for name in ("train", "val", "test")
        }

    val_fraction = float(getattr(dataset_config, "val_fraction", 0.1))
    test_fraction = float(getattr(dataset_config, "test_fraction", 0.0))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")

    split_type = str(getattr(dataset_config, "split_type", "random")).lower()
    allowed = {
        "random",
        "random_with_repeated_smiles",
        "scaffold_balanced",
        "kennard_stone",
        "kmeans",
    }
    if split_type not in allowed:
        raise ValueError(
            f"Unsupported split_type {split_type!r}; expected one of {sorted(allowed)}."
        )

    sizes = (1.0 - val_fraction - test_fraction, val_fraction, test_fraction)
    train_indices, val_indices, test_indices = chemprop_data.make_split_indices(
        [record["mol"] for record in records],
        split=split_type,
        sizes=sizes,
        seed=int(getattr(dataset_config, "seed", 42)),
    )
    return {
        "train": [records[int(index)] for index in train_indices[0]],
        "val": [records[int(index)] for index in val_indices[0]],
        "test": [records[int(index)] for index in test_indices[0]],
    }


def _write_dataset_artifacts(
    workdir: Path,
    splits: dict[str, list[dict[str, Any]]],
    rejected: list[dict[str, Any]],
    target_columns: list[str],
) -> None:
    split_rows = []
    for split_name in ("train", "val", "test"):
        for record in splits[split_name]:
            row = {
                "source_dataset": record["source_dataset"],
                "original_index": record["original_index"],
                "smiles": record["smiles"],
                "split": split_name,
            }
            if record["targets"] is not None:
                row.update(zip(target_columns, record["targets"]))
            split_rows.append(row)
    pd.DataFrame(split_rows).to_csv(workdir / "data_splits.csv", index=False)
    pd.DataFrame(
        rejected,
        columns=["source_dataset", "original_index", "smiles", "reason"],
    ).to_csv(workdir / "rejected_rows.csv", index=False)


def featurize_and_cache_dataset(
    dataset_config: Any,
    featurizer: GraphormerFeaturizer,
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Validate, globally split, featurize, and cache a Graphormer dataset."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "graphormer_manifest.pt"
    cache_key = _dataset_cache_key(dataset_config)
    if manifest_path.is_file():
        manifest = safe_torch_load(manifest_path, map_location="cpu")
        if manifest.get("cache_key") == cache_key:
            print(f"Loading Graphormer cache manifest from {manifest_path}")
            return manifest
        print("Dataset configuration changed; rebuilding the Graphormer cache.")

    dataset_path = Path(dataset_config.dataset_path).expanduser().resolve()
    split_column = getattr(dataset_config, "split_column", None)
    split_column = getattr(dataset_config, "split_col", split_column)
    split_column = str(split_column) if split_column else None
    primary_frame = _read_dataset_table(dataset_path)
    records, rejected = _validated_records(
        primary_frame,
        dataset_config,
        source="training",
        split_column=split_column,
    )
    if not records:
        raise ValueError("No valid labeled molecules remain in the training dataset.")

    external_path_value = getattr(dataset_config, "test_dataset_path", None)
    external_path = None
    external_frame: pd.DataFrame | None = None
    if external_path_value:
        if float(getattr(dataset_config, "test_fraction", 0.0)) != 0.0:
            raise ValueError(
                "test_fraction must be 0 when test_dataset_path is provided."
            )
        external_path = Path(external_path_value).expanduser().resolve()

    splits = _partition_records(records, dataset_config)
    if external_path is not None:
        if splits["test"]:
            raise ValueError(
                "The training dataset contains test rows while test_dataset_path "
                "is configured. Use only one test source."
            )
        external_frame = _read_dataset_table(external_path)
        external_records, external_rejected = _validated_records(
            external_frame,
            dataset_config,
            source="external_test",
            split_column=None,
        )
        if not external_records:
            raise ValueError("No valid labeled molecules remain in the test dataset.")
        splits["test"] = external_records
        rejected.extend(external_rejected)

    if not splits["train"] or not splits["val"]:
        raise ValueError("Training and validation splits must both be non-empty.")

    target_columns = _target_columns(dataset_config)
    task_names = list(getattr(dataset_config, "task_names", None) or target_columns)
    if len(task_names) != len(target_columns):
        raise ValueError("task_names must match the configured target columns.")

    batch_size = int(getattr(dataset_config, "preprocess_batch_size", 100_000))
    shard_paths = {name: [] for name in ("train", "val", "test")}
    split_valid_counts = {name: 0 for name in shard_paths}
    split_task_counts = {
        name: {task_name: 0 for task_name in task_names}
        for name in shard_paths
    }
    for split_name, split_records in splits.items():
        split_dir = cache_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for shard_index, start in enumerate(range(0, len(split_records), batch_size)):
            chunk = split_records[start : start + batch_size]
            shard_path, valid_count, task_counts = _save_shard(
                split_name,
                [record["smiles"] for record in chunk],
                [record["targets"] for record in chunk] if target_columns else None,
                featurizer,
                split_dir,
                shard_index,
                task_names=task_names,
            )
            if shard_path:
                shard_paths[split_name].append(shard_path)
            split_valid_counts[split_name] += valid_count
            for task_name, count in task_counts.items():
                split_task_counts[split_name][task_name] += count

    if not shard_paths["train"] or not shard_paths["val"]:
        raise ValueError(
            "Graphormer featurization produced an empty train or validation split."
        )

    _write_dataset_artifacts(cache_dir.parent, splits, rejected, target_columns)
    manifest: dict[str, Any] = {
        **shard_paths,
        "schema_version": 2,
        "cache_key": cache_key,
        "dataset_path": str(dataset_path),
        "test_dataset_path": str(external_path) if external_path else None,
        "smiles_column": str(dataset_config.smiles_column),
        "target_column": getattr(dataset_config, "target_column", None),
        "task_names": task_names,
        "split_column": split_column,
        "split_type": (
            "existing"
            if split_column
            else str(getattr(dataset_config, "split_type", "random"))
        ),
        "val_fraction": float(getattr(dataset_config, "val_fraction", 0.1)),
        "test_fraction": float(getattr(dataset_config, "test_fraction", 0.0)),
        "total_count": int(
            len(primary_frame)
            + (len(external_frame) if external_frame is not None else 0)
        ),
        "valid_count": int(sum(split_valid_counts.values())),
        "rejected_count": len(rejected),
        "split_raw_counts": {name: len(values) for name, values in splits.items()},
        "split_valid_counts": split_valid_counts,
        "split_task_counts": split_task_counts,
        "max_nodes": getattr(dataset_config, "max_nodes", None),
        "multi_hop_max_dist": getattr(dataset_config, "multi_hop_max_dist", None),
        "spatial_pos_max": getattr(dataset_config, "spatial_pos_max", None),
        "remove_hs": getattr(dataset_config, "remove_hs", True),
        "reorder_atoms": getattr(dataset_config, "reorder_atoms", False),
        "seed": int(getattr(dataset_config, "seed", 42)),
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
