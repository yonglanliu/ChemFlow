"""Download and validate the official CheMeleon message-passing weights.

CheMeleon is released by Jackson Burns and collaborators under the MIT
license. The published model and reference implementation are available at:

https://github.com/JacksonBurns/chemeleon
https://zenodo.org/records/15460715
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from urllib.request import urlopen

from filelock import FileLock


CHEMELEON_WEIGHTS_URL = (
    "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
)
CHEMELEON_WEIGHTS_SHA256 = (
    "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_pretrained_path() -> Path:
    cache_root = os.environ.get("CHEMFLOW_CACHE_DIR")
    if cache_root:
        root = Path(cache_root).expanduser()
    else:
        root = Path.home() / ".cache" / "chemflow"
    return root / "chemeleon" / "chemeleon_mp.pt"


def ensure_pretrained_weights(
    path: str | Path | None = None,
    *,
    url: str = CHEMELEON_WEIGHTS_URL,
    sha256: str | None = CHEMELEON_WEIGHTS_SHA256,
) -> Path:
    """Return a validated local weight file, downloading it when absent."""
    destination = (
        default_pretrained_path()
        if path is None
        else Path(path).expanduser().resolve()
    )

    def validate_existing() -> Path | None:
        if not destination.is_file():
            return None
        if sha256 and _sha256(destination) != sha256:
            raise ValueError(
                "CheMeleon weight checksum mismatch for "
                f"{destination}. Remove the file and retry, or provide the "
                "correct pretrained_sha256 for a custom checkpoint."
            )
        return destination

    existing = validate_existing()
    if existing is not None:
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with FileLock(lock_path):
        # Every DDP worker reaches this function. Only the first worker should
        # download; the others validate and reuse the completed cache entry.
        existing = validate_existing()
        if existing is not None:
            return existing

        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urlopen(url) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)

            if sha256 and _sha256(temporary) != sha256:
                raise ValueError(
                    "Downloaded CheMeleon weights failed SHA-256 validation."
                )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    return destination
