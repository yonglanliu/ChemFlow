"""Qt-free config core for the OpenMM Modeling & MD desktop page.

Builds the four YAML configs consumed by ``chemflow openmm modeling`` — soluble
system build, membrane system build, membrane relaxation, and the MD run — by
merging the desktop form's overrides over the engine defaults (imported from
``chemflow.fe.openmm_modeling``) and emitting nested YAML. Also provides a simple
single-job SLURM script for running a step on a GPU node.

No OpenMM / Qt imports, so it is fully testable.
"""

from __future__ import annotations

import copy
from typing import Any

from chemflow.fe.openmm_modeling.md_config import DEFAULTS as MD_DEFAULTS
from chemflow.fe.openmm_modeling.membrane_build_config import DEFAULTS as MEMBRANE_DEFAULTS
from chemflow.fe.openmm_modeling.membrane_relax_config import (
    DEFAULT_FORCEFIELDS,
    DEFAULTS as RELAX_DEFAULTS,
    SUPPORTED_PLATFORMS,
)
from chemflow.fe.openmm_modeling.soluble_build_config import DEFAULTS as SOLUBLE_DEFAULTS

__all__ = [
    "SOLUBLE_DEFAULTS",
    "MEMBRANE_DEFAULTS",
    "RELAX_DEFAULTS",
    "MD_DEFAULTS",
    "DEFAULT_FORCEFIELDS",
    "SUPPORTED_PLATFORMS",
    "STEPS",
    "build_config",
    "to_yaml",
    "parse_forcefields",
    "format_forcefields",
    "slurm_script",
    "HPC_DEFAULTS",
]

# CLI step name -> (defaults dict, `chemflow openmm modeling <subcommand>`)
STEPS = {
    "soluble": (SOLUBLE_DEFAULTS, "build-soluble"),
    "membrane": (MEMBRANE_DEFAULTS, "build-membrane"),
    "relax": (RELAX_DEFAULTS, "relax-membrane"),
    "md": (MD_DEFAULTS, "md"),
}

HPC_DEFAULTS: dict[str, Any] = {
    "partition": "gpu",
    "cpus_per_task": 8,
    "gpus": 1,
    "gpu_type": "",
    "memory": "32G",
    "walltime": "48:00:00",
    "modules": "",
    "conda_init": "",
    "conda_environment": "",
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into a copy of ``base`` (None is ignored)."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_config(step: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge form overrides over the engine defaults for ``step``."""
    if step not in STEPS:
        raise ValueError(f"unknown openmm step {step!r}; choose from {sorted(STEPS)}")
    defaults, _ = STEPS[step]
    return _deep_merge(defaults, overrides or {})


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in ' :#"\'[]{}'):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _emit(obj: Any, indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                _emit(value, indent + 1, lines)
            elif isinstance(value, (list, tuple)):
                _emit_list(key, value, indent, lines)
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
    else:  # pragma: no cover - top level is always a dict here
        lines.append(f"{pad}{_yaml_scalar(obj)}")


def _emit_list(key: str, value, indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    # Nested list (e.g. box vectors) -> inline flow; list of scalars -> block.
    if any(isinstance(v, (list, tuple, dict)) for v in value):
        inline = "[" + ", ".join(
            ("[" + ", ".join(_yaml_scalar(x) for x in v) + "]") if isinstance(v, (list, tuple))
            else _yaml_scalar(v)
            for v in value
        ) + "]"
        lines.append(f"{pad}{key}: {inline}")
    elif not value:
        lines.append(f"{pad}{key}: []")
    else:
        lines.append(f"{pad}{key}:")
        for item in value:
            lines.append(f"{pad}  - {_yaml_scalar(item)}")


def to_yaml(conf: dict[str, Any]) -> str:
    """Emit a nested YAML mapping (dicts, lists, and box-vector lists-of-lists)."""
    lines: list[str] = []
    _emit(conf, 0, lines)
    return "\n".join(lines) + "\n"


def parse_forcefields(text: str) -> list[str] | None:
    """Parse force fields from newline/comma-separated text; ''/None -> None."""
    if text is None or not str(text).strip():
        return None
    parts = [p.strip() for p in str(text).replace(",", "\n").splitlines()]
    return [p for p in parts if p] or None


def format_forcefields(items) -> str:
    """Render a force-field list one per line for a multiline field."""
    return "\n".join(str(i) for i in (items or []))


def slurm_script(step: str, config_path, hpc: dict[str, Any] | None = None) -> str:
    """A single-job SLURM script running one ``chemflow openmm modeling`` step."""
    hpc = {**HPC_DEFAULTS, **(hpc or {})}
    subcommand = STEPS.get(step, (None, step))[1]
    config_path = str(config_path)
    gpus = int(hpc.get("gpus", 0))
    gpu_type = str(hpc.get("gpu_type", "")).strip()
    gres = (f"gpu:{gpu_type}:{gpus}" if gpu_type else f"gpu:{gpus}") if gpus > 0 else ""

    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name=openmm_{step}",
        f"#SBATCH --partition={hpc['partition']}",
        f"#SBATCH --time={hpc['walltime']}",
    ]
    if gres:
        header.append(f"#SBATCH --gres={gres}")
    header += [
        f"#SBATCH --cpus-per-task={hpc['cpus_per_task']}",
        f"#SBATCH --mem={hpc['memory']}",
    ]

    activate: list[str] = []
    modules = str(hpc.get("modules", "")).replace(",", " ").split()
    if modules:
        activate.append("module purge")
        activate += [f"module load {m}" for m in modules]
    if hpc.get("conda_init"):
        activate.append(f"source {hpc['conda_init']}")
    if hpc.get("conda_environment"):
        activate.append(f"conda activate {hpc['conda_environment']}")

    body = (
        f'\nexport OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"\n'
        f'CONFIG="{config_path}"\n'
        f'chemflow openmm modeling {subcommand} run "${{CONFIG}}"\n'
    )
    return "\n".join(header + ["", *activate]) + "\n" + body
