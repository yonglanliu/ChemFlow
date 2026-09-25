"""Sequential frozen-encoder fusion for CheMeleon and KERMT."""

from __future__ import annotations

import gc
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


DEFAULT_KERMT_EMBEDDINGS = (
    "atom_from_atom",
    "atom_from_bond",
    "bond_from_atom",
    "bond_from_bond",
)


class FusionHead(nn.Module):
    """Small trainable prediction head over concatenated frozen embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Fusion dataset not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _require_checkpoint(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} trained checkpoint does not exist: {path}"
        )
    print(f"{name} trained checkpoint: {path}", flush=True)
    return path


def _device(value: str) -> str:
    requested = str(value or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Fusion requested CUDA, but CUDA is unavailable.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Fusion device must be 'auto', 'cpu', or 'cuda'.")
    return requested


def _release_encoder() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_embeddings(
    frame: pd.DataFrame,
    smiles_column: str,
    chemeleon_checkpoint: Path,
    kermt_checkpoint: Path,
    kermt_embedding_types: tuple[str, ...],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    smiles = frame[smiles_column].astype(str).tolist()

    from chemflow.deep_learning.chemeleon.predictor import CheMeleonPredictor

    print("Extracting CheMeleon embeddings", flush=True)
    chemeleon = CheMeleonPredictor(chemeleon_checkpoint, device=device)
    for parameter in chemeleon.model.parameters():
        parameter.requires_grad_(False)
    chemeleon_output = chemeleon.predict_smiles(
        smiles,
        batch_size=batch_size,
        return_embeddings=True,
    )
    chemeleon_embeddings = np.asarray(chemeleon_output["embedding"], dtype=np.float32)
    del chemeleon, chemeleon_output
    _release_encoder()

    from chemflow.deep_learning.kermt.vendor.task.extract_embeddings import (
        extract_all_embeddings,
        load_encoder_from_checkpoint,
    )

    print("Extracting KERMT embeddings", flush=True)
    kermt_encoder, kermt_readout, kermt_args = load_encoder_from_checkpoint(
        str(kermt_checkpoint), device=device
    )
    for parameter in kermt_encoder.parameters():
        parameter.requires_grad_(False)
    kermt_output, _, kermt_valid = extract_all_embeddings(
        kermt_encoder,
        kermt_readout,
        smiles,
        kermt_args,
        batch_size=batch_size,
        device=device,
    )
    missing_types = sorted(set(kermt_embedding_types) - set(kermt_output))
    if missing_types:
        raise KeyError(f"KERMT embeddings not available: {missing_types}")
    kermt_embeddings = np.concatenate(
        [np.asarray(kermt_output[name], dtype=np.float32) for name in kermt_embedding_types],
        axis=1,
    )
    print(
        "Full frozen embedding dimensions: "
        f"CheMeleon={chemeleon_embeddings.shape[1]}, "
        f"KERMT={kermt_embeddings.shape[1]}, "
        f"combined={chemeleon_embeddings.shape[1] + kermt_embeddings.shape[1]}",
        flush=True,
    )
    del kermt_encoder, kermt_readout, kermt_args, kermt_output
    _release_encoder()

    valid = np.isfinite(chemeleon_embeddings).all(axis=1)
    valid &= np.asarray(kermt_valid, dtype=bool)
    valid &= np.isfinite(kermt_embeddings).all(axis=1)
    return np.concatenate([chemeleon_embeddings, kermt_embeddings], axis=1), valid


def _split_indices(size: int, val_fraction: float, test_fraction: float, seed: int):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if not 0.0 <= test_fraction < 1.0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")
    indices = np.random.default_rng(seed).permutation(size)
    val_count = max(1, int(round(size * val_fraction)))
    test_count = max(0, int(round(size * test_fraction)))
    if val_count + test_count >= size:
        raise ValueError("Fusion split leaves no training rows.")
    return {
        "train": indices[test_count + val_count :],
        "val": indices[test_count : test_count + val_count],
        "test": indices[:test_count],
    }


def _fit_head(
    embeddings: np.ndarray,
    targets: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
) -> tuple[FusionHead, dict[str, Any]]:
    train_indices = splits["train"]
    embedding_mean = embeddings[train_indices].mean(axis=0)
    embedding_scale = embeddings[train_indices].std(axis=0)
    embedding_scale[embedding_scale < 1e-8] = 1.0
    target_mean = targets[train_indices].mean(axis=0)
    target_scale = targets[train_indices].std(axis=0)
    target_scale[target_scale < 1e-8] = 1.0

    normalized_embeddings = (embeddings - embedding_mean) / embedding_scale
    normalized_targets = (targets - target_mean) / target_scale
    model = FusionHead(embeddings.shape[1], hidden_dim, targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = None
    best_validation = float("inf")
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_indices)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            x = torch.as_tensor(normalized_embeddings[batch], dtype=torch.float32, device=device)
            y = torch.as_tensor(normalized_targets[batch], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            validation_x = torch.as_tensor(
                normalized_embeddings[splits["val"]], dtype=torch.float32, device=device
            )
            validation_y = torch.as_tensor(
                normalized_targets[splits["val"]], dtype=torch.float32, device=device
            )
            validation_loss = float(
                nn.functional.mse_loss(model(validation_x), validation_y).cpu()
            )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            print(
                f"Fusion epoch {epoch + 1}/{epochs}: "
                f"train_loss={np.mean(losses):.6f}, val_loss={validation_loss:.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "embedding_mean": embedding_mean.tolist(),
        "embedding_scale": embedding_scale.tolist(),
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "best_validation_loss": best_validation,
    }
    return model.cpu(), metadata


class FusionTrainer:
    """Train a head over sequentially extracted frozen CheMeleon/KERMT embeddings."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        with self.config_path.open("rb") as stream:
            raw = tomllib.load(stream)
        self.base = raw.get("BaseConfig", {})
        self.dataset = raw.get("DatasetConfig", {})
        self.fusion = raw.get("FusionConfig", {})
        self.config_dir = self.config_path.parent

    def train(self) -> Path:
        workdir = _resolve(self.base.get("workdir", "./fusion_run"), self.config_dir)
        workdir.mkdir(parents=True, exist_ok=True)
        frame = _read_table(_resolve(self.dataset["dataset_path"], self.config_dir))
        smiles_column = str(self.dataset.get("smiles_column", "SMILES"))
        target_columns = self.dataset.get("target_column", "target")
        target_columns = [target_columns] if isinstance(target_columns, str) else list(target_columns)
        if smiles_column not in frame or any(column not in frame for column in target_columns):
            raise KeyError("Fusion dataset is missing the configured SMILES or target columns.")
        targets = frame[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid_targets = np.isfinite(targets).all(axis=1)
        device = _device(self.fusion.get("device", "auto"))
        embedding_types = tuple(self.fusion.get("kermt_embedding_types", DEFAULT_KERMT_EMBEDDINGS))
        chemeleon_checkpoint = _require_checkpoint(
            _resolve(self.fusion["chemeleon_checkpoint"], self.config_dir),
            "CheMeleon",
        )
        kermt_checkpoint = _require_checkpoint(
            _resolve(self.fusion["kermt_checkpoint"], self.config_dir),
            "KERMT",
        )
        embeddings, valid_embeddings = _extract_embeddings(
            frame,
            smiles_column,
            chemeleon_checkpoint,
            kermt_checkpoint,
            embedding_types,
            int(self.fusion.get("batch_size", 64)),
            device,
        )
        valid = valid_targets & valid_embeddings
        if valid.sum() < 3:
            raise ValueError("Fusion training requires at least three valid molecules.")
        embeddings = embeddings[valid]
        targets = targets[valid]
        np.savez_compressed(workdir / "embeddings.npz", embeddings=embeddings, targets=targets)
        splits = _split_indices(
            len(embeddings),
            float(self.dataset.get("val_fraction", 0.1)),
            float(self.dataset.get("test_fraction", 0.1)),
            int(self.base.get("seed", 42)),
        )
        model, metadata = _fit_head(
            embeddings,
            targets,
            splits,
            hidden_dim=int(self.fusion.get("hidden_dim", 512)),
            epochs=int(self.fusion.get("epochs", 100)),
            batch_size=int(self.fusion.get("head_batch_size", 128)),
            learning_rate=float(self.fusion.get("learning_rate", 1e-3)),
            weight_decay=float(self.fusion.get("weight_decay", 1e-4)),
            device=device,
        )
        checkpoint = workdir / "fusion_head.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "target_columns": target_columns,
                "kermt_embedding_types": list(embedding_types),
                "input_dim": embeddings.shape[1],
                "hidden_dim": int(self.fusion.get("hidden_dim", 512)),
                "encoders_frozen": True,
                "embedding_projection": None,
                "encoder_device": device,
                "chemeleon_checkpoint": str(chemeleon_checkpoint),
                "kermt_checkpoint": str(kermt_checkpoint),
                **metadata,
            },
            checkpoint,
        )
        print(f"Fusion head saved to {checkpoint}", flush=True)
        return checkpoint
