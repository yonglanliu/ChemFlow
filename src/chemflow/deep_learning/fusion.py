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

from chemflow.deep_learning.task_weighting import resolve_task_loss_weights


DEFAULT_KERMT_EMBEDDINGS = (
    "atom_from_atom",
    "atom_from_bond",
    "bond_from_atom",
    "bond_from_bond",
)


class FusionHead(nn.Module):
    """Small trainable prediction head over concatenated frozen embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    kermt_valid = np.asarray(kermt_valid, dtype=bool)
    kermt_valid_indices = np.flatnonzero(kermt_valid)
    kermt_embeddings_valid = np.concatenate(
        [np.asarray(kermt_output[name], dtype=np.float32) for name in kermt_embedding_types],
        axis=1,
    )
    kermt_embeddings = np.full(
        (len(smiles), kermt_embeddings_valid.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    kermt_embeddings[kermt_valid_indices] = kermt_embeddings_valid[kermt_valid_indices]
    kermt_smiles = [smiles[index] for index in kermt_valid_indices]
    print(
        f"KERMT accepted {len(kermt_smiles):,}/{len(smiles):,} molecules; "
        "CheMeleon will process only this accepted subset.",
        flush=True,
    )
    del kermt_encoder, kermt_readout, kermt_args, kermt_output
    _release_encoder()

    from chemflow.deep_learning.chemeleon.predictor import CheMeleonPredictor

    print("Extracting CheMeleon embeddings", flush=True)
    chemeleon = CheMeleonPredictor(chemeleon_checkpoint, device=device)
    for parameter in chemeleon.model.parameters():
        parameter.requires_grad_(False)
    chemeleon_output = chemeleon.predict_smiles(
        kermt_smiles,
        batch_size=batch_size,
        return_embeddings=True,
    )
    chemeleon_embeddings_valid = np.asarray(
        chemeleon_output["embedding"], dtype=np.float32
    )
    chemeleon_valid = np.isfinite(chemeleon_embeddings_valid).all(axis=1)
    chemeleon_valid_indices = kermt_valid_indices[chemeleon_valid]
    chemeleon_embeddings = np.full(
        (len(smiles), chemeleon_embeddings_valid.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    chemeleon_embeddings[chemeleon_valid_indices] = (
        chemeleon_embeddings_valid[chemeleon_valid]
    )
    del chemeleon, chemeleon_output
    _release_encoder()

    valid = np.zeros(len(smiles), dtype=bool)
    valid[chemeleon_valid_indices] = True
    combined = np.concatenate([chemeleon_embeddings, kermt_embeddings], axis=1)
    print(
        f"Both encoders accepted {int(valid.sum()):,}/{len(smiles):,} molecules. "
        "Full frozen embedding dimensions: "
        f"CheMeleon={chemeleon_embeddings.shape[1]}, "
        f"KERMT={kermt_embeddings.shape[1]}, combined={combined.shape[1]}",
        flush=True,
    )
    return combined, valid


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


def _elementwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    name: str,
    huber_delta: float,
    nll_scale: float,
    gaussian_nll_variance: float,
) -> torch.Tensor:
    if name in {"l2", "mse"}:
        return (prediction - target).square()
    if name == "mae":
        return (prediction - target).abs()
    if name == "huber":
        return nn.functional.smooth_l1_loss(
            prediction, target, beta=huber_delta, reduction="none"
        )
    if name == "nll":
        scale = torch.as_tensor(nll_scale, dtype=prediction.dtype, device=prediction.device)
        return (prediction - target).abs() / scale + torch.log(scale)
    if name == "gaussian_nll":
        variance = torch.as_tensor(
            gaussian_nll_variance,
            dtype=prediction.dtype,
            device=prediction.device,
        )
        return 0.5 * ((prediction - target).square() / variance + torch.log(variance))
    raise ValueError(
        "regression_loss must be one of l2, mse, mae, huber, nll, gaussian_nll."
    )


def _task_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    names: list[str],
    prefix: str,
) -> tuple[dict[str, float], str]:
    metrics: dict[str, float] = {}
    rendered = []
    for index, name in enumerate(names):
        finite = np.isfinite(target[:, index]) & np.isfinite(prediction[:, index])
        if not finite.any():
            metrics.update(
                {
                    f"{prefix}_{name}_mae": float("nan"),
                    f"{prefix}_{name}_rmse": float("nan"),
                    f"{prefix}_{name}_r2": float("nan"),
                }
            )
            rendered.append(f"{name}:no_labels")
            continue
        error = prediction[finite, index] - target[finite, index]
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error ** 2)))
        finite_target = target[finite, index]
        centered = finite_target - finite_target.mean()
        denominator = float(np.sum(centered ** 2))
        r2 = float(1.0 - np.sum(error ** 2) / denominator) if denominator else float("nan")
        metrics.update(
            {
                f"{prefix}_{name}_mae": mae,
                f"{prefix}_{name}_rmse": rmse,
                f"{prefix}_{name}_r2": r2,
            }
        )
        rendered.append(f"{name}:MAE={mae:.4f},RMSE={rmse:.4f},R2={r2:.4f}")
    return metrics, "; ".join(rendered)


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
    dropout: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    resume_checkpoint: Path | None,
    target_names: list[str],
    regression_loss: str,
    huber_delta: float,
    nll_scale: float,
    gaussian_nll_variance: float,
    task_loss_weighting: str,
    task_loss_weights: list[float] | dict[str, float] | None,
    lr_scheduler: str,
    min_learning_rate: float,
    device: str,
) -> tuple[FusionHead, dict[str, Any]]:
    train_indices = splits["train"]
    embedding_mean = embeddings[train_indices].mean(axis=0)
    embedding_scale = embeddings[train_indices].std(axis=0)
    embedding_scale[embedding_scale < 1e-8] = 1.0
    target_mean = np.nanmean(targets[train_indices], axis=0)
    target_scale = np.nanstd(targets[train_indices], axis=0)
    if not np.isfinite(target_mean).all():
        raise ValueError("Every fusion target needs at least one training label.")
    target_scale[target_scale < 1e-8] = 1.0
    regression_loss = str(regression_loss).strip().lower()
    if huber_delta <= 0.0 or nll_scale <= 0.0 or gaussian_nll_variance <= 0.0:
        raise ValueError("Fusion loss scale parameters must be positive.")
    _, resolved_task_weights = resolve_task_loss_weights(
        targets[train_indices],
        target_names,
        strategy=task_loss_weighting,
        configured=task_loss_weights,
    )
    task_weights = torch.as_tensor(resolved_task_weights, dtype=torch.float32, device=device)

    normalized_embeddings = (embeddings - embedding_mean) / embedding_scale
    normalized_target_mask = np.isfinite(targets).astype(np.float32)
    normalized_targets = np.nan_to_num(
        (targets - target_mean) / target_scale,
        nan=0.0,
    )
    if not 0.0 <= dropout < 1.0:
        raise ValueError("Fusion head dropout must be between 0 and 1.")
    model = FusionHead(
        embeddings.shape[1], hidden_dim, targets.shape[1], dropout=dropout
    ).to(device)
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        print(f"Resumed fusion head from {resume_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = str(lr_scheduler).strip().lower()
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=min_learning_rate,
        )
    elif lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif lr_scheduler in {"constant", "none"}:
        scheduler = None
    else:
        raise ValueError(
            "lr_scheduler must be 'constant', 'cosine', or 'exponential'."
        )
    best_state = None
    best_validation = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_indices)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            x = torch.as_tensor(normalized_embeddings[batch], dtype=torch.float32, device=device)
            y = torch.as_tensor(normalized_targets[batch], dtype=torch.float32, device=device)
            mask = torch.as_tensor(
                normalized_target_mask[batch], dtype=torch.float32, device=device
            )
            optimizer.zero_grad()
            elementwise = _elementwise_loss(
                model(x),
                y,
                regression_loss,
                huber_delta,
                nll_scale,
                gaussian_nll_variance,
            )
            counts = mask.sum(dim=0)
            task_losses = (elementwise * mask).sum(dim=0) / counts.clamp_min(1.0)
            active = counts > 0
            loss = (task_losses * task_weights * active).sum() / (
                task_weights * active
            ).sum().clamp_min(1.0)
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
            validation_mask = torch.as_tensor(
                normalized_target_mask[splits["val"]],
                dtype=torch.float32,
                device=device,
            )
            validation_prediction = model(validation_x)
            validation_elementwise = _elementwise_loss(
                validation_prediction,
                validation_y,
                regression_loss,
                huber_delta,
                nll_scale,
                gaussian_nll_variance,
            )
            validation_counts = validation_mask.sum(dim=0)
            validation_task_losses = (
                validation_elementwise * validation_mask
            ).sum(dim=0) / validation_counts.clamp_min(1.0)
            validation_active = validation_counts > 0
            validation_loss = float(
                (
                    validation_task_losses
                    * task_weights
                    * validation_active
                ).sum()
                / (task_weights * validation_active).sum().clamp_min(1.0)
            .cpu()
            )
            train_x = torch.as_tensor(
                normalized_embeddings[train_indices], dtype=torch.float32, device=device
            )
            train_prediction = model(train_x).cpu().numpy()
            validation_prediction = validation_prediction.cpu().numpy()
        target_scale_array = np.asarray(target_scale)
        target_mean_array = np.asarray(target_mean)
        train_metrics, _ = _task_metrics(
            train_prediction * target_scale_array + target_mean_array,
            targets[train_indices],
            target_names,
            "train",
        )
        validation_metrics, validation_text = _task_metrics(
            validation_prediction * target_scale_array + target_mean_array,
            targets[splits["val"]],
            target_names,
            "val",
        )
        if validation_loss < best_validation - early_stopping_min_delta:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(losses)),
                "val_loss": float(validation_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **train_metrics,
                **validation_metrics,
            })
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            print(
                f"Fusion epoch {epoch + 1}/{epochs}: "
                f"train_loss={np.mean(losses):.6f}, val_loss={validation_loss:.6f}; "
                f"validation metrics: {validation_text}",
                flush=True,
            )
        if early_stopping_patience > 0 and bad_epochs >= early_stopping_patience:
            print(
                f"Fusion early stopping at epoch {epoch + 1} "
                f"after {bad_epochs} unimproved epochs.",
                flush=True,
            )
            break
        if scheduler is not None:
            scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "embedding_mean": embedding_mean.tolist(),
        "embedding_scale": embedding_scale.tolist(),
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "best_validation_loss": best_validation,
        "dropout": dropout,
        "regression_loss": regression_loss,
        "task_loss_weighting": task_loss_weighting,
        "task_loss_weights": resolved_task_weights.tolist(),
        "lr_scheduler": lr_scheduler,
        "min_learning_rate": min_learning_rate,
        "history": history,
    }
    return model.cpu(), metadata


def _predict_head(
    model: FusionHead,
    embeddings: np.ndarray,
    metadata: dict[str, Any],
    *,
    mc_samples: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mc_samples == 1 or mc_samples < 0:
        raise ValueError("test_mc_dropout_samples must be 0 or at least 2.")
    normalized = (
        embeddings - np.asarray(metadata["embedding_mean"])
    ) / np.asarray(metadata["embedding_scale"])
    model = model.to(device)
    values = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        direct = model(values).cpu().numpy()
        if mc_samples >= 2:
            draws = []
            model.train()
            for _ in range(mc_samples):
                draws.append(model(values).cpu().numpy())
            model.eval()
            draws_array = np.stack(draws, axis=0)
            mean = draws_array.mean(axis=0)
            standard_deviation = draws_array.std(axis=0, ddof=1)
        else:
            mean = direct.copy()
            standard_deviation = np.zeros_like(direct)
    target_scale = np.asarray(metadata["target_scale"])
    target_mean = np.asarray(metadata["target_mean"])
    return (
        direct * target_scale + target_mean,
        mean * target_scale + target_mean,
        standard_deviation * target_scale,
    )


class FusionTrainer:
    """Train a head over sequentially extracted frozen CheMeleon/KERMT embeddings."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        with self.config_path.open("rb") as stream:
            raw = tomllib.load(stream)
        self.base = raw.get("BaseConfig", {})
        self.dataset = raw.get("DatasetConfig", {})
        self.fusion_config = raw.get("FusionConfig", {})
        self.fusion_training = raw.get(
            "FusionTrainingConfig",
            raw.get("TrainingConfig", {}),
        )
        # Keep the original single-section format working while supporting
        # the same model/training split used by the other deep-learning models.
        self.fusion = {**self.fusion_config, **self.fusion_training}
        self.config_dir = self.config_path.parent

    def train(self) -> Path:
        workdir = _resolve(self.base.get("workdir", "./fusion_run"), self.config_dir)
        workdir.mkdir(parents=True, exist_ok=True)
        frame = _read_table(_resolve(self.dataset["dataset_path"], self.config_dir))
        test_path = self.dataset.get("test_dataset_path")
        test_frame = _read_table(_resolve(test_path, self.config_dir)) if test_path else None
        if test_frame is not None and float(self.dataset.get("test_fraction", 0.0)) != 0.0:
            raise ValueError("test_fraction must be 0 when test_dataset_path is configured.")
        smiles_column = str(self.dataset.get("smiles_column", "SMILES"))
        target_columns = self.dataset.get("target_column", "target")
        target_columns = [target_columns] if isinstance(target_columns, str) else list(target_columns)
        for dataset_name, dataset_frame in (("training", frame), ("test", test_frame)):
            if dataset_frame is not None and (
                smiles_column not in dataset_frame
                or any(column not in dataset_frame for column in target_columns)
            ):
                raise KeyError(
                    f"Fusion {dataset_name} dataset is missing the configured "
                    "SMILES or target columns."
                )
        train_count = len(frame)
        extraction_frame = (
            pd.concat([frame, test_frame], ignore_index=True)
            if test_frame is not None
            else frame
        )
        all_targets = extraction_frame[target_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        train_targets = all_targets[:train_count]
        valid_train_targets = np.isfinite(train_targets).any(axis=1)
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
        all_embeddings, valid_embeddings = _extract_embeddings(
            extraction_frame,
            smiles_column,
            chemeleon_checkpoint,
            kermt_checkpoint,
            embedding_types,
            int(self.fusion.get("batch_size", 64)),
            device,
        )
        valid_train = valid_train_targets & valid_embeddings[:train_count]
        if valid_train.sum() < 3:
            label_counts = np.isfinite(train_targets[valid_train]).sum(axis=0)
            raise ValueError(
                "Fusion training requires at least three encoder-valid molecules. "
                f"Valid rows={int(valid_train.sum())}; label counts by task="
                f"{dict(zip(target_columns, label_counts.astype(int)))}."
            )
        embeddings = all_embeddings[:train_count][valid_train]
        targets = train_targets[valid_train]
        test_embeddings = None
        test_targets = None
        if test_frame is not None:
            test_targets_all = all_targets[train_count:]
            valid_test = np.isfinite(test_targets_all).any(axis=1) & valid_embeddings[train_count:]
            test_embeddings = all_embeddings[train_count:][valid_test]
            test_targets = test_targets_all[valid_test]
            if len(test_embeddings) == 0:
                raise ValueError("External test dataset has no valid molecules.")
        clean_dir = workdir / "prepared_data"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_training_frame = frame.loc[valid_train].reset_index(drop=True)
        np.savez_compressed(
            workdir / "embeddings.npz",
            embeddings=embeddings,
            targets=targets,
            test_embeddings=test_embeddings
            if test_embeddings is not None
            else np.empty((0, embeddings.shape[1]), dtype=np.float32),
            test_targets=test_targets
            if test_targets is not None
            else np.empty((0, targets.shape[1]), dtype=np.float32),
        )
        splits = _split_indices(
            len(embeddings),
            float(self.dataset.get("val_fraction", 0.1)),
            float(self.dataset.get("test_fraction", 0.1)),
            int(self.base.get("seed", 42)),
        )
        clean_training_frame.iloc[splits["train"]].to_csv(
            clean_dir / "train.csv", index=False
        )
        clean_training_frame.iloc[splits["val"]].to_csv(
            clean_dir / "val.csv", index=False
        )
        if test_frame is not None:
            test_frame.loc[valid_test].to_csv(clean_dir / "test.csv", index=False)
        else:
            clean_training_frame.iloc[splits["test"]].to_csv(
                clean_dir / "test.csv", index=False
            )
        print(
            "Prepared filtered fusion datasets: "
            f"train={len(splits['train']):,}, "
            f"val={len(splits['val']):,}, "
            f"test={len(test_embeddings) if test_embeddings is not None else len(splits['test']):,}; "
            f"directory={clean_dir}",
            flush=True,
        )
        resume_checkpoint = None
        if bool(self.fusion.get("resume", False)):
            resume_value = self.fusion.get("resume_checkpoint")
            resume_checkpoint = _resolve(
                resume_value or workdir / "fusion_head.pt",
                self.config_dir,
            )
            if not resume_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Fusion resume checkpoint does not exist: {resume_checkpoint}"
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
            dropout=float(self.fusion.get("dropout", 0.0)),
            early_stopping_patience=int(
                self.fusion.get("early_stopping_patience", 10)
            ),
            early_stopping_min_delta=float(
                self.fusion.get("early_stopping_min_delta", 0.0)
            ),
            resume_checkpoint=resume_checkpoint,
            target_names=target_columns,
            regression_loss=str(self.fusion.get("regression_loss", "mse")),
            huber_delta=float(self.fusion.get("huber_delta", 1.0)),
            nll_scale=float(self.fusion.get("nll_scale", 1.0)),
            gaussian_nll_variance=float(
                self.fusion.get("gaussian_nll_variance", 1.0)
            ),
            task_loss_weighting=str(
                self.fusion.get("task_loss_weighting", "uniform")
            ),
            task_loss_weights=self.fusion.get("task_loss_weights"),
            lr_scheduler=str(self.fusion.get("lr_scheduler", "cosine")),
            min_learning_rate=float(
                self.fusion.get("min_learning_rate", 1e-5)
            ),
            device=device,
        )
        history = pd.DataFrame(metadata.pop("history", []))
        history.to_csv(workdir / "training_history.csv", index=False)
        if bool(self.fusion.get("plot_training_history", True)) and not history.empty:
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot(history["epoch"], history["train_loss"], label="Training")
            axis.plot(history["epoch"], history["val_loss"], label="Validation")
            axis.set(xlabel="Epoch", ylabel="Loss", title="Fusion head training")
            axis.legend()
            figure.tight_layout()
            figure.savefig(workdir / "training_curve.png", dpi=160)
            plt.close(figure)

        evaluation_embeddings = (
            test_embeddings
            if test_embeddings is not None
            else embeddings[splits["test"]]
        )
        evaluation_targets = (
            test_targets
            if test_targets is not None
            else targets[splits["test"]]
        )
        if len(evaluation_embeddings):
            mc_samples = int(self.fusion.get("test_mc_dropout_samples", 30))
            confidence = float(
                self.fusion.get("test_calibration_confidence", 0.90)
            )
            if not 0.0 < confidence < 1.0:
                raise ValueError("test_calibration_confidence must be between 0 and 1.")
            direct, mc_mean, mc_std = _predict_head(
                model,
                evaluation_embeddings,
                metadata,
                mc_samples=mc_samples,
                device=device,
            )
            validation_direct, _, _ = _predict_head(
                model,
                embeddings[splits["val"]],
                metadata,
                mc_samples=0,
                device=device,
            )
            validation_targets = targets[splits["val"]]
            coverage_label = int(round(confidence * 100))
            output = pd.DataFrame()
            if test_frame is not None:
                output[smiles_column] = test_frame.loc[valid_test, smiles_column].to_numpy()
            else:
                valid_training_frame = frame.loc[valid_train].reset_index(drop=True)
                output[smiles_column] = valid_training_frame.loc[splits["test"], smiles_column].to_numpy()
            for index, name in enumerate(target_columns):
                validation_finite = (
                    np.isfinite(validation_targets[:, index])
                    & np.isfinite(validation_direct[:, index])
                )
                if validation_finite.any():
                    validation_error = (
                        validation_targets[validation_finite, index]
                        - validation_direct[validation_finite, index]
                    )
                    bias = float(np.median(validation_error))
                    residuals = np.abs(validation_error - bias)
                    radius = float(
                        np.quantile(
                            residuals,
                            min(
                                1.0,
                                np.ceil((len(residuals) + 1) * confidence)
                                / len(residuals),
                            ),
                            method="higher",
                        )
                    )
                else:
                    bias = 0.0
                    radius = float("nan")
                calibrated = mc_mean[:, index] + bias
                output[f"{name}_truth"] = evaluation_targets[:, index]
                output[f"direct_{name}"] = direct[:, index]
                output[f"mc_mean_{name}"] = mc_mean[:, index]
                output[f"mc_std_{name}"] = mc_std[:, index]
                output[f"{name}_prediction"] = mc_mean[:, index]
                output[f"calibrated_{name}"] = calibrated
                output[f"{name}_lower_{coverage_label}"] = calibrated - radius
                output[f"{name}_upper_{coverage_label}"] = calibrated + radius
                output[f"{name}_error"] = mc_mean[:, index] - evaluation_targets[:, index]
            output.to_csv(workdir / "test_predictions.csv", index=False)
            _, test_metrics_text = _task_metrics(
                mc_mean,
                evaluation_targets,
                target_columns,
                "test",
            )
            print(
                "Fusion test metrics: "
                f"{test_metrics_text}; MC samples={mc_samples}",
                flush=True,
            )
        checkpoint = workdir / "fusion_head.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "target_columns": target_columns,
                "kermt_embedding_types": list(embedding_types),
                "input_dim": embeddings.shape[1],
                "hidden_dim": int(self.fusion.get("hidden_dim", 512)),
                "dropout": float(self.fusion.get("dropout", 0.0)),
                "regression_loss": str(
                    self.fusion.get("regression_loss", "mse")
                ),
                "lr_scheduler": str(
                    self.fusion.get("lr_scheduler", "cosine")
                ),
                "resume": bool(self.fusion.get("resume", False)),
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
