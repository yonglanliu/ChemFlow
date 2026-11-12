"""AlchemFold's view of the shared HPC presets.

The presets themselves now live in the top-level :mod:`hpc` package — one YAML
per module in ``src/hpc/presets/`` (``docking.yaml``, ``screening.yaml``, …) —
so docking, screening, OpenFE, GROMACS and OpenMM all read the same loader and
the same user override file:

    1. ``$CHEMFLOW_HPC_CONFIG``
    2. ``<PROJECT_ROOT>/env/hpc.yaml``
    3. ``~/.config/chemflow/hpc.yaml``

This module stays as AlchemFold's stable surface (``preset_resources`` and
friends are called from the CLI and the desktop), translating between the
``section`` wording used here and the ``module`` wording used by
:mod:`hpc.presets`, and returning the same dict shape as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hpc import presets as _presets

DEFAULT_PARTITION = "gpu"


def config_path() -> Path | None:
    """The user HPC config file in use, or None (built-ins only)."""
    return _presets.config_path()


def load_settings(section: str = "docking") -> dict[str, Any]:
    """Resolved ``{defaults, presets, gpu_types, default_preset}`` for a section."""
    return _presets.settings(section)


def presets(section: str = "docking") -> dict[str, Any]:
    """``{preset_name: walltime-or-resources}`` from the current config."""
    return _presets.settings(section)["presets"]


def gpu_types(section: str = "docking") -> list[str]:
    """Selectable GPU types ("" = no type constraint / any GPU)."""
    return _presets.gpu_types(section)


def resource_defaults(section: str = "docking") -> dict[str, Any]:
    """Default SLURM resources before any preset is applied."""
    return _presets.settings(section)["defaults"]


def default_preset(section: str = "docking") -> str:
    """The configured default preset, guaranteed to exist in ``presets()``."""
    return _presets.default_name(section)


def preset_resources(name: str, section: str = "docking") -> dict[str, Any]:
    """Fully-resolved SLURM resources for a preset.

    Same keys as always: ``time, nodes, ntasks_per_node, gpus_per_node,
    gpu_type, cpus_per_task, mem, partition, account, conda_env, shards``.
    Raises ``KeyError`` if the preset name is unknown.
    """
    return _presets.resources(section, name).as_alchemfold()
