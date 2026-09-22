"""Resolve and audit the official NVIDIA KERMT v2 pretrained artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable


KERMT_REPO_ID = "nvidia/NV-KERMT-70M-v2"
KERMT_REVISION = "7df5eb3179235fdea1e8124db73215da33d77dce"
KERMT_CHECKPOINT = "kermt_contrastive_v2.0.pt"
KERMT_ARTIFACTS = (
    KERMT_CHECKPOINT,
    "pretrain_atom_vocab.json",
    "pretrain_bond_vocab.json",
    "pretrain_smiles_vocab.pkl",
    "LICENSE",
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache_dir() -> Path:
    """Return ChemFlow's stable cache directory for the official KERMT model."""
    cache_root = os.environ.get("CHEMFLOW_CACHE_DIR")
    root = Path(cache_root).expanduser() if cache_root else Path.home() / ".cache" / "chemflow"
    return root / "kermt" / "NV-KERMT-70M-v2"


def ensure_pretrained_artifacts(
    *,
    cache_dir: str | Path | None = None,
    repo_id: str = KERMT_REPO_ID,
    revision: str = KERMT_REVISION,
    local_files_only: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    """Download missing official artifacts and return their resolved paths.

    Hugging Face performs its own atomic download and locking. Supplying a
    stable ``local_dir`` gives ChemFlow readable paths while safely allowing
    simultaneous Slurm jobs to reuse the same completed files.
    """
    destination = (
        default_cache_dir()
        if cache_dir is None
        else Path(cache_dir).expanduser().resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Automatic KERMT download requires huggingface_hub. Install "
            "ChemFlow with the 'kermt' optional dependencies."
        ) from exc

    log("KERMT pretrained source: Hugging Face Hub")
    log(f"KERMT repository: {repo_id}")
    log(f"KERMT revision: {revision}")
    log(f"KERMT cache directory: {destination}")

    resolved: dict[str, Path] = {}
    for filename in KERMT_ARTIFACTS:
        expected = destination / filename
        was_cached = expected.is_file() and expected.stat().st_size > 0
        action = "Using cached" if was_cached else "Downloading"
        log(f"{action} KERMT artifact: {filename}")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=destination,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            mode = "local cache" if local_files_only else "Hugging Face Hub"
            raise RuntimeError(
                f"Could not resolve KERMT artifact {filename!r} from {mode}. "
                f"Repository: {repo_id}; revision: {revision}; cache: "
                f"{destination}"
            ) from exc
        path = Path(downloaded).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Resolved KERMT artifact is missing or empty: {path}")
        resolved[filename] = path
        log(f"Resolved KERMT artifact: {path} ({path.stat().st_size:,} bytes)")

    checkpoint = resolved[KERMT_CHECKPOINT]
    log(f"KERMT checkpoint SHA256: {sha256(checkpoint)}")
    log("KERMT pretrained artifacts loaded successfully.")
    return resolved
