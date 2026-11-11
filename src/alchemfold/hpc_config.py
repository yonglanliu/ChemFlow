"""User-customizable HPC run settings for AlchemFold docking.

Walltime **presets**, selectable **GPU types**, and SLURM **resource defaults**
are read from a YAML file so each cluster can tune them without editing code.
The file is looked up in this order (first one found wins):

  1. ``$CHEMFLOW_HPC_CONFIG``
  2. ``<PROJECT_ROOT>/env/hpc.yaml``   (shipped example; edit in place)
  3. ``~/.config/chemflow/hpc.yaml``   (per-user override)

A missing file — or any missing/invalid key — falls back to the built-in
defaults below. Expected YAML shape::

    defaults:        # SLURM resources (walltime comes from the preset)
      partition: gpu
      account: ""
      cpus_per_task: 8
      mem: 32G
      conda_env: alchemfold
    gpu_types: ["", a100, v100x, p100, v100, l40]
    default_preset: docking-standard
    presets:         # name -> SLURM --time (all single-node single-GPU)
      docking-quick:    "01:00:00"
      docking-standard: "04:00:00"
      docking-long:     "1-00:00:00"

A dedicated ``docking:`` section (same keys: presets/defaults/gpu_types/
default_preset) overrides the top-level values for the docking path, letting the
shared top-level ``presets`` stay tuned for OpenFE while docking gets short ones.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

try:  # chemflow is the sibling package; fall back to a path guess if absent.
    from chemflow.config import PROJECT_ROOT
except Exception:  # pragma: no cover
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PARTITION = "gpu"

_BUILTIN: dict[str, Any] = {
    "defaults": {
        "partition": DEFAULT_PARTITION,
        "account": "",
        "cpus_per_task": 8,
        "mem": "32G",
        "conda_env": "alchemfold",
    },
    "gpu_types": ["", "a100", "v100x", "p100", "v100", "l40", "gpuh200"],
    "default_preset": "docking-standard",
    "presets": {
        # Co-folding / docking is single-GPU inference and short per complex, so
        # these walltimes are tuned for docking (not multi-day FE jobs).
        "docking-quick": "01:00:00",      # 1 hour — small/medium complex, smoke test
        "docking-standard": "04:00:00",   # 4 hours — typical single docking run (default)
        "docking-long": "1-00:00:00",     # 1 day — large complex / many diffusion samples / MSA build
        "docking-xlong": "2-00:00:00",    # 2 days — very large complex or a big ligand_file batch
    },
}


def config_path() -> Path | None:
    """Return the HPC config file that will be used, or None (built-ins only)."""
    env = os.environ.get("CHEMFLOW_HPC_CONFIG", "").strip()
    candidates = [
        Path(env).expanduser() if env else None,
        PROJECT_ROOT / "env" / "hpc.yaml",
        Path.home() / ".config" / "chemflow" / "hpc.yaml",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    return None


def load_settings(section: str = "docking") -> dict[str, Any]:
    """Load HPC settings, merging a YAML file (if any) over the built-ins.

    ``section`` selects the per-workflow override block (``docking`` for single
    docking, ``screening`` for virtual screening); it overrides the top-level
    values, which stay shared with OpenFE.
    """
    settings = copy.deepcopy(_BUILTIN)
    if section == "screening":
        settings["default_preset"] = "screen-1gpu-x20"
        settings["presets"] = {
            # Screening = a SLURM array of shards. 'shards' is how many array tasks
            # (= GPUs running in parallel); 'gpus_per_task' > 1 enables Boltz DDP
            # inside each shard.
            "screen-1gpu-x4": {"time": "1-00:00:00", "gpus_per_node": 1, "shards": 4},
            "screen-1gpu-x20": {"time": "1-00:00:00", "gpus_per_node": 1, "shards": 20},
            "screen-1gpu-x50": {"time": "1-00:00:00", "gpus_per_node": 1, "shards": 50},
            "screen-4gpu-x5": {"time": "1-00:00:00", "gpus_per_node": 4, "shards": 5},
        }
    path = config_path()
    if path is None:
        return settings
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return settings
    if not isinstance(data, dict):
        return settings
    if section == "docking":
        # Top-level presets/defaults/gpu_types are the shared (OpenFE-style) ones;
        # screening ignores them and uses only its own section.
        if isinstance(data.get("defaults"), dict):
            settings["defaults"].update(data["defaults"])
        if isinstance(data.get("presets"), dict) and data["presets"]:
            # A preset value is either a walltime string (single-node single-GPU) or
            # a mapping of resources (time/nodes/ntasks_per_node/gpus_per_node/...).
            norm: dict[str, Any] = {}
            for key, value in data["presets"].items():
                norm[str(key)] = value if isinstance(value, dict) else str(value)
            settings["presets"] = norm
        if data.get("default_preset"):
            settings["default_preset"] = str(data["default_preset"])
    else:
        if isinstance(data.get("defaults"), dict):
            settings["defaults"].update(data["defaults"])
    if isinstance(data.get("gpu_types"), list):
        settings["gpu_types"] = ["" if g is None else str(g) for g in data["gpu_types"]]
    # A dedicated section (`docking:` / `screening:`, mirrors the `openfe:` section)
    # overrides the top-level values for that path.
    dock = data.get(section)
    if isinstance(dock, dict):
        if isinstance(dock.get("defaults"), dict):
            settings["defaults"].update(dock["defaults"])
        if isinstance(dock.get("presets"), dict) and dock["presets"]:
            settings["presets"] = {
                str(k): (v if isinstance(v, dict) else str(v))
                for k, v in dock["presets"].items()
            }
        if isinstance(dock.get("gpu_types"), list):
            settings["gpu_types"] = ["" if g is None else str(g) for g in dock["gpu_types"]]
        if dock.get("default_preset"):
            settings["default_preset"] = str(dock["default_preset"])
    return settings


def presets(section: str = "docking") -> dict[str, Any]:
    """``{preset_name: walltime-or-resources}`` from the current config."""
    return load_settings(section)["presets"]


def gpu_types(section: str = "docking") -> list[str]:
    """Selectable GPU types ("" = no type constraint / any GPU)."""
    return load_settings(section)["gpu_types"]


def resource_defaults(section: str = "docking") -> dict[str, Any]:
    """Default SLURM resources (partition/account/cpus_per_task/mem/conda_env)."""
    return load_settings(section)["defaults"]


def default_preset(section: str = "docking") -> str:
    """The configured default preset, guaranteed to exist in ``presets()``."""
    settings = load_settings(section)
    chosen = settings["default_preset"]
    available = settings["presets"]
    return chosen if chosen in available else next(iter(available))


def preset_resources(name: str, section: str = "docking") -> dict[str, Any]:
    """Return the fully-resolved SLURM resources for a preset.

    A preset may be a plain walltime string (single-node single-GPU) or a
    mapping overriding any of: ``time, nodes, ntasks_per_node, gpus_per_node,
    gpu_type, cpus_per_task, mem, partition, account, conda_env``. Missing keys
    fall back to the ``defaults`` section, then to built-in single-node values.
    Screening presets may also set ``shards`` (number of SLURM array tasks, i.e.
    GPUs running in parallel). Raises ``KeyError`` if the preset name is unknown.
    """
    settings = load_settings(section)
    presets = settings["presets"]
    if name not in presets:
        raise KeyError(name)
    raw = presets[name]
    raw = {"time": raw} if not isinstance(raw, dict) else dict(raw)
    defaults = settings["defaults"]

    def pick(key: str, fallback: Any) -> Any:
        if key in raw and raw[key] is not None:
            return raw[key]
        if key in defaults and defaults[key] is not None:
            return defaults[key]
        return fallback

    return {
        "time": str(raw.get("time") or defaults.get("time") or "1-00:00:00"),
        "nodes": int(pick("nodes", 1)),
        "ntasks_per_node": int(pick("ntasks_per_node", 1)),
        "gpus_per_node": int(pick("gpus_per_node", 1)),
        "gpu_type": str(pick("gpu_type", "") or ""),
        "cpus_per_task": int(pick("cpus_per_task", 8)),
        "mem": str(pick("mem", "32G")),
        "partition": str(pick("partition", DEFAULT_PARTITION)),
        "account": str(pick("account", "") or ""),
        "conda_env": str(pick("conda_env", "alchemfold") or ""),
        "shards": int(raw.get("shards") or 0),
    }
