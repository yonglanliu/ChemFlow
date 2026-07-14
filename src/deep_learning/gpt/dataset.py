# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

def _iter_smiles_batches(dataset_path: Path, smiles_column: str, batch_size: int, target_column: str | None = None):
    suffix = dataset_path.suffix.lower()

    if suffix in [".parquet", ".pq"]:
        parquet_file = pq.ParquetFile(dataset_path)

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=[smiles_column] + ([target_column] if target_column is not None else []),
        ):
            df = batch.to_pandas()
            if target_column is not None:
                df = df[[smiles_column, target_column]].dropna(subset=[smiles_column, target_column])
            smiles = (
                df[smiles_column]
                .astype(str)
                .str.strip()
                .tolist()
            )
            if target_column is not None:
                targets = df[target_column].tolist()
                yield smiles, targets
            else:
                yield smiles

    elif suffix == ".csv":
        for df in pd.read_csv(dataset_path, chunksize=batch_size):
            if target_column is not None:
                df = df[[smiles_column, target_column]].dropna(subset=[smiles_column, target_column])
                smiles = (
                    df[smiles_column]
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                targets = df[target_column].tolist()
                yield smiles, targets
            else:
                smiles = (
                    df[smiles_column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                yield smiles

    elif suffix == ".json":
        df = pd.read_json(dataset_path)
        smiles = (
            df[smiles_column]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        yield smiles

    elif suffix == ".smi":
        if target_column is not None:
            raise ValueError("Target column is not supported for .smi files.")
        smiles = []

        with dataset_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # .smi may be: "SMILES molecule_name"
                smiles.append(line.split()[0])

                if len(smiles) >= batch_size:
                    yield smiles
                    smiles = []

        if smiles:
            yield smiles

    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")


def _tokenize_smiles(smiles, tokenizer, max_length: int = 128):
    encoded = tokenizer(
        smiles,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]

    return {"input_ids": input_ids,}


def tokenize_and_cache_dataset(dataset_config, tokenizer, max_length, cache_dir):
    dataset_path = Path(dataset_config.dataset_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()

    cached_manifest_path = cache_dir / "tokenized_manifest.pt"
    train_cache_dir = cache_dir / "train"
    val_cache_dir = cache_dir / "val"

    if cached_manifest_path.exists():
        print(f"Loading tokenized cache manifest from {cached_manifest_path}")
        return torch.load(cached_manifest_path)

    print(f"Building tokenized cache from {dataset_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    smiles_column = getattr(dataset_config, "smiles_column", "SMILES")
    val_fraction = getattr(dataset_config, "val_fraction", 0.1)
    seed = getattr(dataset_config, "seed", 42)
    batch_size = getattr(dataset_config, "preprocess_batch_size", 100_000)

    rng = np.random.default_rng(seed)

    train_shards = []
    val_shards = []

    train_idx = 0
    val_idx = 0
    total_count = 0

    for batch_idx, smiles_batch in enumerate(
        _iter_smiles_batches(
            dataset_path=dataset_path,
            smiles_column=smiles_column,
            batch_size=batch_size,
        )
    ):
        smiles_batch = [s for s in smiles_batch if s]

        if not smiles_batch:
            continue

        # Random split inside each batch.
        # This avoids loading the whole large-scale dataset into memory.
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
            train_data = _tokenize_smiles(
                train_smiles,
                tokenizer=tokenizer,
                max_length=max_length,
            )

            train_path = train_cache_dir / f"train_{train_idx:06d}.pt"
            torch.save(train_data, train_path)

            train_shards.append(str(train_path))
            train_idx += 1

        if val_smiles:
            val_data = _tokenize_smiles(
                val_smiles,
                tokenizer=tokenizer,
                max_length=max_length,
            )

            val_path = val_cache_dir / f"val_{val_idx:06d}.pt"
            torch.save(val_data, val_path)

            val_shards.append(str(val_path))
            val_idx += 1

        total_count += len(smiles_batch)

        print(
            f"Processed batch {batch_idx + 1} | "
            f"total molecules={total_count:,} | "
            f"train={len(train_shards)} | "
            f"val={len(val_shards)}"
        )

    manifest = {
        "train": train_shards,
        "val": val_shards,
        "dataset_path": str(dataset_path),
        "smiles_column": smiles_column,
        "max_length": max_length,
        "val_fraction": val_fraction,
        "total_count": total_count,
    }

    torch.save(manifest, cached_manifest_path)

    print(f"Saved tokenized cache manifest to {cached_manifest_path}")

    return manifest


class TokenizedSmilesCacheDataset(Dataset):
    def __init__(self, shard_paths):
        self.shard_paths = [Path(p) for p in shard_paths]

        self.shards = []
        self.index = []

        for shard_id, path in enumerate(self.shard_paths):
            data = torch.load(path, map_location="cpu")

            # expected:
            # data["input_ids"], data["attention_mask"], data["labels"]
            n = data["input_ids"].size(0)

            self.shards.append(data)

            for i in range(n):
                self.index.append((shard_id, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        shard_id, row_id = self.index[idx]
        shard = self.shards[shard_id]

        return {"input_ids": shard["input_ids"][row_id]}