"""Configuration loading for the standalone ADMET deployment desktop."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from chemflow.config import PROJECT_ROOT


@dataclass(frozen=True)
class ADMEDeploymentConfig:
    source: Path
    multitask_checkpoint: str = ""
    mlm_checkpoint: str = ""
    device: str = "auto"
    batch_size: int = 64
    num_workers: int = 0
    applicability_domain: bool = True
    embedding_dimensions: int = 128
    calibration_confidence: float = 0.90
    similarity_radius: int = 2
    similarity_bits: int = 2048
    min_tanimoto_similarity: float = 0.35
    max_embedding_distance: float = 0.35
    max_log_interval_width: float = 1.0
    ood_score_threshold: float = 0.95


def find_default_adme_config() -> Path | None:
    """Find the automatically loaded deployment config in priority order."""
    configured = os.environ.get("CHEMFLOW_ADMET_CONFIG", "").strip()
    if configured:
        path = Path(os.path.expandvars(configured)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"CHEMFLOW_ADMET_CONFIG does not point to a file: {path}"
            )
        return path

    candidates = (
        Path.home() / ".config" / "chemflow" / "admet_desktop.toml",
        PROJECT_ROOT / "example" / "admet_desktop" / "config.toml",
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _resolve_config_path(value: object, base_directory: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    expanded = os.path.expandvars(os.path.expanduser(text))
    if "$" in expanded:
        raise ValueError(f"Checkpoint path contains an unknown variable: {text}")
    path = Path(expanded)
    if not path.is_absolute():
        path = base_directory / path
    return str(path.resolve())


def load_adme_deployment_config(path: str | Path) -> ADMEDeploymentConfig:
    source = Path(os.path.expandvars(str(path))).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ADMET deployment config does not exist: {source}")
    with source.open("rb") as stream:
        raw = tomllib.load(stream)

    checkpoints = raw.get("checkpoints", {})
    inference = raw.get("inference", {})
    quality = raw.get("quality_control", {})
    if not all(isinstance(section, dict) for section in (checkpoints, inference, quality)):
        raise TypeError(
            "[checkpoints], [inference], and [quality_control] must be TOML tables."
        )

    device = str(inference.get("device", "auto")).strip().lower()
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("inference.device must be auto, cpu, cuda, or mps.")

    config = ADMEDeploymentConfig(
        source=source,
        multitask_checkpoint=_resolve_config_path(
            checkpoints.get("multitask", ""), source.parent
        ),
        mlm_checkpoint=_resolve_config_path(
            checkpoints.get("mlm", ""), source.parent
        ),
        device=device,
        batch_size=int(inference.get("batch_size", 64)),
        num_workers=int(inference.get("num_workers", 0)),
        applicability_domain=bool(inference.get("applicability_domain", True)),
        embedding_dimensions=int(inference.get("embedding_dimensions", 128)),
        calibration_confidence=float(
            inference.get("calibration_confidence", 0.90)
        ),
        similarity_radius=int(inference.get("similarity_radius", 2)),
        similarity_bits=int(inference.get("similarity_bits", 2048)),
        min_tanimoto_similarity=float(
            quality.get("min_tanimoto_similarity", 0.35)
        ),
        max_embedding_distance=float(
            quality.get("max_embedding_distance", 0.35)
        ),
        max_log_interval_width=float(
            quality.get("max_log_interval_width", 1.0)
        ),
        ood_score_threshold=float(quality.get("ood_score_threshold", 0.95)),
    )
    if config.batch_size < 1 or config.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative.")
    if not 1 <= config.embedding_dimensions <= 128:
        raise ValueError("embedding_dimensions must be between 1 and 128.")
    if not 0.50 <= config.calibration_confidence <= 0.99:
        raise ValueError("calibration_confidence must be between 0.50 and 0.99.")
    if config.similarity_radius < 1 or config.similarity_bits < 64:
        raise ValueError("Invalid Morgan fingerprint settings.")
    if not 0.0 <= config.min_tanimoto_similarity <= 1.0:
        raise ValueError("min_tanimoto_similarity must be between 0 and 1.")
    if not 0.0 <= config.max_embedding_distance <= 2.0:
        raise ValueError("max_embedding_distance must be between 0 and 2.")
    if config.max_log_interval_width <= 0.0:
        raise ValueError("max_log_interval_width must be positive.")
    if not 0.5 < config.ood_score_threshold < 1.0:
        raise ValueError("ood_score_threshold must be between 0.5 and 1.")
    return config
