"""Small YAML/config helpers shared by every AlchemFold config module.

These used to live in the Boltz config module, which meant unrelated code
(screening, PROTAC docking) imported a Boltz module just to merge a dict.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_namespace(value: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace (lists handled elementwise)."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Co-folding config must be a YAML mapping: {path}")
    return data


def _resolve(base_dir: Path, value: Any) -> Any:
    if not value:
        return value
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return str(candidate)


SUPPORTED_OUTPUT_FORMATS = {"pdb", "mmcif"}
SUPPORTED_ACCELERATORS = {"gpu", "cpu", "tpu"}
