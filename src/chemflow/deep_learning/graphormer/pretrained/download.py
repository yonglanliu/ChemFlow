"""Download and cache the official Microsoft Graphormer checkpoint."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from filelock import FileLock


GRAPHORMER_WEIGHTS_URL = (
    "https://huggingface.co/clefourrier/graphormer-base-pcqm4mv1/"
    "resolve/main/pytorch_model.bin"
)
GRAPHORMER_WEIGHTS_SHA256 = (
    "8d636207c28c3fa9b05bd0107c78f5e4d7768cefec2fd3489172ba3940041342"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_pretrained_path() -> Path:
    cache_root = os.environ.get("CHEMFLOW_CACHE_DIR")
    root = (
        Path(cache_root).expanduser()
        if cache_root
        else Path.home() / ".cache" / "chemflow"
    )
    return root / "graphormer" / "graphormer-base-pcqm4mv1.pt"


def ensure_pretrained_weights(
    path: str | Path | None = None,
    *,
    url: str = GRAPHORMER_WEIGHTS_URL,
    sha256: str | None = None,
) -> Path:
    """Return a local checkpoint, downloading it once when necessary."""
    destination = (
        default_pretrained_path()
        if path is None
        else Path(path).expanduser().resolve()
    )

    def validate_existing() -> Path | None:
        if not destination.is_file():
            return None
        if destination.stat().st_size == 0:
            raise ValueError(f"Graphormer checkpoint is empty: {destination}")
        if sha256 and _sha256(destination) != sha256:
            raise ValueError(
                "Graphormer checkpoint checksum mismatch for "
                f"{destination}. Remove it and retry, or configure the "
                "correct pretrained_sha256 for a custom checkpoint."
            )
        return destination

    existing = validate_existing()
    if existing is not None:
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with FileLock(lock_path):
        existing = validate_existing()
        if existing is not None:
            return existing

        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size == 0:
                raise ValueError("Downloaded Graphormer checkpoint is empty.")
            if sha256 and _sha256(temporary) != sha256:
                raise ValueError(
                    "Downloaded Graphormer checkpoint failed SHA-256 validation."
                )
            temporary.replace(destination)
        except (OSError, URLError) as error:
            raise RuntimeError(
                "Could not download the Graphormer pretrained checkpoint from "
                f"{url!r}. Check network/DNS access, or download the checkpoint "
                "manually and set GraphormerConfig.pretrained_path. To train "
                "without pretrained weights, set use_pretrained = false."
            ) from error
        finally:
            if temporary.exists():
                temporary.unlink()

    return destination
