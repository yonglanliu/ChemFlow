from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.deep_learning.gpt.dataset import _iter_smiles_batches
from src.deep_learning.graphormer import GraphormerFeaturizer

# Shared molecular-feature preprocessing used by regular ML and Graphormer.
# If your module has a different filename, only change this import path.
from src.deep_learning.graphormer.modules.data_pipeline import (
    _normalize_feature_types,
    _featurize_single_smiles,
    make_feature_preprocessor,
    get_processed_feature_dimensions,
    split_processed_features,
)


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
    extra_feature_types: list[str] | str | None = None,
) -> list[Data]:
    """
    Featurize a list of SMILES strings for Graphormer.

    If `extra_feature_types` is provided, raw molecular features are also
    generated and attached to each PyG Data object as:

        data.raw_extra_features

    These features are intentionally stored RAW here.  The train-set
    preprocessor is fitted only after all train shards have been created,
    then train/val/test shards are transformed consistently.
    """

    if target_list is not None:
        if len(smiles_list) != len(target_list):
            raise ValueError(
                "smiles_list and target_list must have the same length: "
                f"{len(smiles_list)} vs {len(target_list)}."
            )

    use_extra_features = extra_feature_types is not None

    if use_extra_features:
        extra_feature_types = _normalize_feature_types(
            extra_feature_types
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

        # -----------------------------------------------------
        # Raw descriptors / fingerprints
        # -----------------------------------------------------
        if use_extra_features:
            raw_extra = _featurize_single_smiles(
                smiles,
                extra_feature_types,
            )

            # Keep Graphormer features and extra features aligned.
            # If the auxiliary featurizer fails, skip this molecule.
            if raw_extra is None:
                continue

            data.raw_extra_features = torch.as_tensor(
                raw_extra,
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
    extra_feature_types=None,
):
    data_list = featurize_smiles_list(
        smiles,
        featurizer,
        targets,
        extra_feature_types=extra_feature_types,
    )

    if not data_list:
        return None, 0, {}

    shard_path = cache_dir / f"{split_name}_{shard_idx:05d}.pt"

    torch.save(data_list, shard_path)

    task_counts = {}

    if task_names is not None:
        task_counts = _count_task_samples(data_list, task_names)

    return (str(shard_path), len(data_list), task_counts)


def _fit_and_apply_extra_feature_preprocessor(
    manifest,
    *,
    feature_types,
    cache_dir,
    fingerprint_variance_threshold=None,
):
    feature_types = _normalize_feature_types(
        feature_types
    )

    print(
        "\nStarting extra-feature preprocessing..."
    )
    print(
        f"Feature types: {feature_types}"
    )

    train_raw_features = []

    # ============================================================
    # Collect training features
    # ============================================================

    print(
        "Collecting raw extra features from training shards..."
    )

    for shard_index, shard_path in enumerate(
        manifest.get("train", []),
        start=1,
    ):
        print(
            f"  Loading train shard "
            f"{shard_index}: {shard_path}"
        )

        shard = safe_torch_load(
            shard_path,
            map_location="cpu",
        )

        for data in shard:
            raw = getattr(
                data,
                "raw_extra_features",
                None,
            )

            if raw is None:
                raise RuntimeError(
                    "Extra features were requested, but a "
                    "training sample does not contain "
                    "raw_extra_features."
                )

            train_raw_features.append(
                raw.detach().cpu().numpy()
            )

    print(
        f"Collected {len(train_raw_features):,} "
        f"training feature vectors."
    )

    # ============================================================
    # Stack training features
    # ============================================================

    print(
        "Stacking training feature matrix..."
    )

    X_train_raw = np.vstack(
        train_raw_features
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        f"Raw training feature matrix: "
        f"{X_train_raw.shape}"
    )

    # ============================================================
    # Build and fit preprocessor
    # ============================================================

    print(
        "Building extra-feature preprocessor..."
    )

    preprocessor = make_feature_preprocessor(
        feature_types=feature_types,
        fingerprint_variance_threshold=(
            fingerprint_variance_threshold
        ),
    )

    print(
        "Fitting extra-feature preprocessor "
        "on TRAINING data only..."
    )

    preprocessor.fit(
        X_train_raw
    )

    print(
        "Extra-feature preprocessor fitted."
    )

    dims = get_processed_feature_dimensions(
        preprocessor=preprocessor,
        X_reference=X_train_raw,
        feature_types=feature_types,
    )

    descriptor_dim = int(
        dims["descriptor_dim"]
    )

    fingerprint_dim = int(
        dims["fingerprint_dim"]
    )

    print(
        f"Processed descriptor dim: "
        f"{descriptor_dim}"
    )

    print(
        f"Processed fingerprint dim: "
        f"{fingerprint_dim}"
    )

    print(
        f"Total processed dim: "
        f"{dims['total_dim']}"
    )

    # We no longer need this large matrix after fitting.
    del X_train_raw
    del train_raw_features

    # ============================================================
    # Transform train / val / test
    # ============================================================

    for split_name in (
        "train",
        "val",
        "test",
    ):
        shard_paths = manifest.get(
            split_name,
            [],
        )

        print(
            f"\nTransforming {split_name} "
            f"extra features..."
        )

        for shard_index, shard_path in enumerate(
            shard_paths,
            start=1,
        ):
            print(
                f"  [{shard_index}/{len(shard_paths)}] "
                f"Loading {shard_path}"
            )

            shard = safe_torch_load(
                shard_path,
                map_location="cpu",
            )

            raw_matrix = np.vstack(
                [
                    data.raw_extra_features
                    .detach()
                    .cpu()
                    .numpy()
                    for data in shard
                ]
            ).astype(
                np.float32,
                copy=False,
            )

            print(
                f"    Raw shape: "
                f"{raw_matrix.shape}"
            )

            print(
                "    Transforming..."
            )

            processed = (
                preprocessor.transform(
                    raw_matrix
                )
            )

            processed = np.asarray(
                processed,
                dtype=np.float32,
            )

            print(
                f"    Processed shape: "
                f"{processed.shape}"
            )

            blocks = split_processed_features(
                processed,
                descriptor_dim=descriptor_dim,
            )

            descriptor_features = blocks[
                "descriptors"
            ]

            fingerprint_features = blocks[
                "fingerprints"
            ]

            print(
                "    Attaching processed features "
                "to Graphormer Data objects..."
            )

            for idx, data in enumerate(shard):

                if descriptor_features is not None:
                    data.descriptor_features = (
                        torch.as_tensor(
                            descriptor_features[idx],
                            dtype=torch.float32,
                        )
                    )

                if fingerprint_features is not None:
                    data.fingerprint_features = (
                        torch.as_tensor(
                            fingerprint_features[idx],
                            dtype=torch.float32,
                        )
                    )

                if hasattr(
                    data,
                    "raw_extra_features",
                ):
                    del data.raw_extra_features

            print(
                "    Saving processed shard..."
            )

            torch.save(
                shard,
                shard_path,
            )

            print(
                "    Done."
            )

    # ============================================================
    # Save preprocessor
    # ============================================================

    preprocessor_path = (
        cache_dir
        / "extra_feature_preprocessor.pt"
    )

    print(
        f"\nSaving fitted preprocessor to "
        f"{preprocessor_path}"
    )

    torch.save(
        preprocessor,
        preprocessor_path,
    )

    manifest["extra_features"] = {
        "enabled": True,
        "feature_types": list(
            feature_types
        ),
        "descriptor_dim": descriptor_dim,
        "fingerprint_dim": fingerprint_dim,
        "total_dim": int(
            dims["total_dim"]
        ),
        "fingerprint_variance_threshold": (
            fingerprint_variance_threshold
        ),
        "preprocessor_path": str(
            preprocessor_path
        ),
    }

    print(
        "\nExtra-feature preprocessing complete."
    )

    return manifest


def featurize_and_cache_dataset(
    dataset_config: Any,
    featurizer: GraphormerFeaturizer,
    cache_dir: str | Path,
) -> dict[str, Any]:

    dataset_path = (Path(dataset_config.dataset_path).expanduser().resolve())

    cache_dir = (Path(cache_dir).expanduser().resolve())

    manifest_path = (cache_dir / "graphormer_manifest.pt")

    train_cache_dir = cache_dir / "train"
    val_cache_dir = cache_dir / "val"
    test_cache_dir = cache_dir / "test"

    if manifest_path.exists():
        print(f"Loading Graphormer cache manifest from {manifest_path}")

        return safe_torch_load(manifest_path, map_location="cpu")

    smiles_column = dataset_config.smiles_column

    target_column = getattr(dataset_config, "target_column", None)

    # ---------------------------------------------------------
    # Resolve task names
    # ---------------------------------------------------------

    if target_column is None:
        target_columns = []
        task_names = []

    elif isinstance(target_column, str):
        target_columns = [target_column]
        task_names = [target_column]

    else:
        target_columns = list(target_column)

        task_names = getattr(
            dataset_config,
            "task_names",
            None,
        )

        if task_names is None:
            task_names = target_columns

        if len(task_names) != len(target_columns):
            raise ValueError(
                f"Number of task names ({len(task_names)}) "
                f"does not match number of target columns "
                f"({len(target_columns)})."
            )

    val_fraction = float(
        getattr(
            dataset_config,
            "val_fraction",
            0.1,
        )
    )

    test_fraction = float(
        getattr(
            dataset_config,
            "test_fraction",
            0.0,
        )
    )

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in [0, 1), "
            f"got {val_fraction}."
        )

    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(
            f"test_fraction must be in [0, 1), "
            f"got {test_fraction}."
        )

    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            "val_fraction + test_fraction "
            "must be less than 1."
        )

    seed = int(
        getattr(
            dataset_config,
            "seed",
            42,
        )
    )

    batch_size = int(
        getattr(
            dataset_config,
            "preprocess_batch_size",
            100_000,
        )
    )

    # ---------------------------------------------------------
    # Optional descriptors / fingerprints for late fusion
    # ---------------------------------------------------------
    extra_feature_types = getattr(
        dataset_config,
        "extra_feature_types",
        None,
    )

    use_extra_features = bool(
        getattr(
            dataset_config,
            "use_extra_features",
            extra_feature_types is not None,
        )
    )

    if not use_extra_features:
        extra_feature_types = None

    fingerprint_variance_threshold = getattr(
        dataset_config,
        "fingerprint_variance_threshold",
        None,
    )

    if fingerprint_variance_threshold is not None:
        fingerprint_variance_threshold = float(
            fingerprint_variance_threshold
        )

    split_column = getattr(dataset_config, "split_column", None)
    split_column = getattr(dataset_config, "split_col", split_column)

    if split_column is not None:
        split_column = str(split_column)

        if dataset_path.suffix.lower() in {".csv"}:
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix.lower() in {".parquet", ".pq"}:
            try:
                import pyarrow.parquet as pq
                df = pq.read_table(dataset_path).to_pandas()
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Failed to read parquet dataset for split column '{split_column}'.") from exc
        else:
            raise ValueError(
                f"split_column is only supported for CSV or Parquet datasets, got {dataset_path.suffix}"
            )

        if split_column not in df.columns:
            raise ValueError(
                f"split_column '{split_column}' was not found in the dataset. "
                f"Available columns: {list(df.columns)[:20]}"
            )

        required_columns = [smiles_column]
        if target_column is not None:
            if isinstance(target_column, str):
                required_columns.append(target_column)
            else:
                required_columns.extend(list(target_column))
        required_columns.append(split_column)

        df = df[required_columns].dropna(subset=[smiles_column])

        split_groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        split_targets: dict[str, list[Any]] = {"train": [], "val": [], "test": []}

        if target_column is None:
            for _, row in df.iterrows():
                split_value = _normalize_split_name(row[split_column])
                if split_value is None:
                    continue
                split_groups[split_value].append(str(row[smiles_column]).strip())
        else:
            if isinstance(target_column, str):
                target_columns = [target_column]
            else:
                target_columns = list(target_column)

            for _, row in df.iterrows():
                split_value = _normalize_split_name(row[split_column])
                if split_value is None:
                    continue

                split_groups[split_value].append(str(row[smiles_column]).strip())
                split_targets[split_value].append([
                    row[col] if col in row and pd.notna(row[col]) else np.nan
                    for col in target_columns
                ])

        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for split_name in ("train", "val", "test"):
            current_dir = cache_dir / split_name
            current_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "train": [],
            "val": [],
            "test": [],
            "dataset_path": str(dataset_path),
            "smiles_column": smiles_column,
            "target_column": target_column,
            "task_names": task_names,
            "split_column": split_column,
            "split_mode": "existing",
            "preprocess_batch_size": batch_size,
            "total_count": int(sum(len(v) for v in split_groups.values())),
            "valid_count": int(sum(len(v) for v in split_groups.values())),
        }

        for split_name in ("train", "val", "test"):
            split_smiles = split_groups.get(split_name, [])
            split_target_values = split_targets.get(split_name, []) if target_column is not None else None
            if not split_smiles:
                continue

            shard_path = cache_dir / split_name / f"{split_name}_00000.pt"
            data_list = featurize_smiles_list(
                split_smiles,
                featurizer,
                split_target_values,
                extra_feature_types=extra_feature_types,
            )
            torch.save(data_list, shard_path)
            manifest[split_name] = [str(shard_path)]

        if use_extra_features:
            manifest = _fit_and_apply_extra_feature_preprocessor(
                manifest,
                feature_types=extra_feature_types,
                cache_dir=cache_dir,
                fingerprint_variance_threshold=(
                    fingerprint_variance_threshold
                ),
            )

        manifest_path = cache_dir / "graphormer_manifest.pt"
        torch.save(manifest, manifest_path)
        print(f"Loaded existing split from column '{split_column}' and saved Graphormer cache manifest to {manifest_path}")
        return manifest

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if test_fraction > 0.0:
        test_cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    multi_hop_max_dist = getattr(
        dataset_config,
        "multi_hop_max_dist",
        None,
    )

    spatial_pos_max = getattr(
        dataset_config,
        "spatial_pos_max",
        None,
    )

    max_nodes = getattr(
        dataset_config,
        "max_nodes",
        None,
    )

    remove_hs = getattr(
        dataset_config,
        "remove_hs",
        True,
    )

    reorder_atoms = getattr(
        dataset_config,
        "reorder_atoms",
        False,
    )

    rng = np.random.default_rng(seed)

    train_shards: list[str] = []
    val_shards: list[str] = []
    test_shards: list[str] = []

    train_idx = 0
    val_idx = 0
    test_idx = 0

    total_count = 0
    valid_count = 0

    split_raw_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    split_valid_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    # ---------------------------------------------------------
    # Per-task sample counts
    # ---------------------------------------------------------

    split_task_counts = {
        split: {
            task_name: 0
            for task_name in task_names
        }
        for split in (
            "train",
            "val",
            "test",
        )
    }

    print(
        f"Building Graphormer cache "
        f"from {dataset_path}"
    )

    for batch_idx, batch in enumerate(
        _iter_smiles_batches(
            dataset_path=dataset_path,
            smiles_column=smiles_column,
            batch_size=batch_size,
            target_column=target_column,
        )
    ):

        raw_smiles_batch = list(
            batch[0]
        )

        raw_target_batch = (
            list(batch[1])
            if target_column is not None
            else None
        )

        smiles_batch, target_batch = (
            _filter_smiles_and_targets(
                raw_smiles_batch,
                raw_target_batch,
            )
        )

        if not smiles_batch:
            continue

        smiles_batch, target_batch = (
            _shuffle_aligned(
                smiles_batch,
                target_batch,
                rng,
            )
        )

        split_data = _split_batch(
            smiles_batch,
            target_batch,
            val_fraction,
            test_fraction,
            rng,
        )

        total_count += len(
            smiles_batch
        )

        for split_name in ("train", "val", "test"):
            split_smiles = split_data[split_name].get("smiles") or []
            split_raw_counts[split_name] += len(split_smiles)

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        (
            train_path,
            train_valid,
            train_task_counts,
        ) = _save_shard(
            "train",
            split_data["train"]["smiles"],
            split_data["train"]["targets"],
            featurizer,
            train_cache_dir,
            train_idx,
            task_names=task_names,
            extra_feature_types=extra_feature_types,
        )

        if train_path is not None:
            train_shards.append(
                train_path
            )

            train_idx += 1

            valid_count += train_valid

            split_valid_counts[
                "train"
            ] += train_valid

            for task_name, count in (
                train_task_counts.items()
            ):
                split_task_counts[
                    "train"
                ][task_name] += count

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        (
            val_path,
            val_valid,
            val_task_counts,
        ) = _save_shard(
            "val",
            split_data["val"]["smiles"],
            split_data["val"]["targets"],
            featurizer,
            val_cache_dir,
            val_idx,
            task_names=task_names,
            extra_feature_types=extra_feature_types,
        )

        if val_path is not None:
            val_shards.append(
                val_path
            )

            val_idx += 1

            valid_count += val_valid

            split_valid_counts[
                "val"
            ] += val_valid

            for task_name, count in (
                val_task_counts.items()
            ):
                split_task_counts[
                    "val"
                ][task_name] += count

        # -----------------------------------------------------
        # Test
        # -----------------------------------------------------

        if test_fraction > 0.0:
            (
                test_path,
                test_valid,
                test_task_counts,
            ) = _save_shard(
                "test",
                split_data["test"]["smiles"],
                split_data["test"]["targets"],
                featurizer,
                test_cache_dir,
                test_idx,
                task_names=task_names,
                extra_feature_types=extra_feature_types,
            )

            if test_path is not None:
                test_shards.append(
                    test_path
                )

                test_idx += 1

                valid_count += test_valid

                split_valid_counts[
                    "test"
                ] += test_valid

                for task_name, count in (
                    test_task_counts.items()
                ):
                    split_task_counts[
                        "test"
                    ][task_name] += count

        print(
            f"Processed batch {batch_idx + 1} | "
            f"total={total_count:,} | "
            f"valid={valid_count:,} | "
            f"train_shards={len(train_shards)} | "
            f"val_shards={len(val_shards)} | "
            f"test_shards={len(test_shards)}"
        )

    # ---------------------------------------------------------
    # Print task counts
    # ---------------------------------------------------------

    if task_names:
        print("\nSamples per task:")

        for task_name in task_names:
            print(
                f"{task_name}: "
                f"train="
                f"{split_task_counts['train'][task_name]:,}, "
                f"val="
                f"{split_task_counts['val'][task_name]:,}, "
                f"test="
                f"{split_task_counts['test'][task_name]:,}"
            )

    # ---------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------

    manifest: dict[str, Any] = {
        "train": train_shards,
        "val": val_shards,
        "test": test_shards,
        "dataset_path": str(dataset_path),
        "smiles_column": smiles_column,
        "target_column": target_column,
        "task_names": task_names,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "total_count": total_count,
        "valid_count": valid_count,
        "split_raw_counts": split_raw_counts,
        "split_valid_counts": split_valid_counts,

        # NEW
        "split_task_counts": split_task_counts,

        "max_nodes": max_nodes,
        "multi_hop_max_dist": multi_hop_max_dist,
        "spatial_pos_max": spatial_pos_max,
        "remove_hs": remove_hs,
        "reorder_atoms": reorder_atoms,
        "seed": seed,
        "preprocess_batch_size": batch_size,
    }

    if use_extra_features:
        manifest = _fit_and_apply_extra_feature_preprocessor(
            manifest,
            feature_types=extra_feature_types,
            cache_dir=cache_dir,
            fingerprint_variance_threshold=(
                fingerprint_variance_threshold
            ),
        )

    torch.save(
        manifest,
        manifest_path,
    )

    print(
        f"Saved Graphormer cache manifest "
        f"to {manifest_path}"
    )

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
