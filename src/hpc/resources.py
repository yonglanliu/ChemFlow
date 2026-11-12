"""Canonical SLURM resource record, and the key aliases every module invented.

The same quantity is spelled differently in each corner of the repo —
``walltime``/``time``, ``memory``/``mem``, ``ntasks``/``tasks``/
``ntasks_per_node``, ``gpus``/``gpus_per_node``, ``conda_environment``/
``conda_env``. Rather than pick a winner and break three config formats,
:func:`normalize_keys` maps all of them onto one vocabulary, and the ``as_*``
views hand each consumer back the spelling it already expects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# alias -> canonical
ALIASES: dict[str, str] = {
    "walltime": "time",
    "memory": "mem",
    "tasks": "ntasks",
    "ntasks_per_node": "ntasks",
    "gpus_per_node": "gpus",
    "gpus_per_task": "gpus",
    "conda_environment": "conda_env",
    "conda_envname": "conda_env",
    "cpus": "cpus_per_task",
}

CANONICAL_KEYS = (
    "partition", "account", "nodes", "ntasks", "cpus_per_task", "gpus",
    "gpu_type", "mem", "time", "conda_env", "conda_init", "modules",
    "constraint", "replicates", "shards",
)


def normalize_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite any known alias to its canonical name (unknown keys pass through).

    Canonical names win: if a mapping somehow carries both ``time`` and
    ``walltime``, the explicit canonical value is kept.
    """
    out: dict[str, Any] = {}
    for key, value in (data or {}).items():
        canonical = ALIASES.get(str(key), str(key))
        if canonical in out and str(key) in ALIASES:
            continue            # don't let an alias clobber an explicit canonical
        out[canonical] = value
    return out


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


@dataclass
class Resources:
    """One job's SLURM allocation, in canonical units."""

    partition: str = "gpu"
    account: str = ""
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 8
    gpus: int = 1
    gpu_type: str = ""
    mem: str = "32G"
    time: str = "1-00:00:00"
    conda_env: str = ""
    conda_init: str = ""
    modules: str = ""
    constraint: str = ""
    replicates: int = 1
    shards: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Resources":
        """Build from any dialect; unknown keys are kept in ``extra``."""
        values = normalize_keys(data)
        known = {f: values.pop(f) for f in list(values) if f in CANONICAL_KEYS}
        res = cls()
        for key, value in known.items():
            if value is None:
                continue
            current = getattr(res, key)
            setattr(res, key, _as_int(value, current) if isinstance(current, int)
                    else str(value))
        res.extra = {k: v for k, v in values.items() if v is not None}
        return res

    def merged(self, override: dict[str, Any] | None) -> "Resources":
        """A copy with ``override`` (any dialect) applied on top."""
        if not override:
            return Resources.from_mapping(self.as_dict())
        data = {**self.as_dict(), **normalize_keys(override)}
        return Resources.from_mapping(data)

    # -- views -----------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        data.update(extra)
        return data

    def as_alchemfold(self) -> dict[str, Any]:
        """The shape ``alchemfold.hpc_config.preset_resources`` has always returned."""
        return {
            "time": self.time,
            "nodes": self.nodes,
            "ntasks_per_node": self.ntasks,
            "gpus_per_node": self.gpus,
            "gpu_type": self.gpu_type,
            "cpus_per_task": self.cpus_per_task,
            "mem": self.mem,
            "partition": self.partition,
            "account": self.account,
            "conda_env": self.conda_env,
            "shards": self.shards,
        }

    def as_openfe(self) -> dict[str, Any]:
        """The shape the OpenFE / OpenMM desktop ``slurm:`` block uses."""
        return {
            "partition": self.partition,
            "account": self.account,
            "nodes": self.nodes,
            "ntasks": self.ntasks,
            "cpus_per_task": self.cpus_per_task,
            "gpus": self.gpus,
            "gpu_type": self.gpu_type,
            "memory": self.mem,
            "walltime": self.time,
            "replicates": self.replicates,
            "modules": self.modules,
            "conda_init": self.conda_init,
            "conda_environment": self.conda_env,
        }

    def as_gromacs(self) -> dict[str, Any]:
        """The shape the GROMACS TOML ``[slurm]`` table uses."""
        return {
            "partition": self.partition,
            "nodes": self.nodes,
            "tasks": self.ntasks,
            "cpus_per_task": self.cpus_per_task,
            "gpus": self.gpus,
            "gpu_type": self.gpu_type,
            "constraint": self.constraint,
            "memory": self.mem,
            "walltime": self.time,
            "conda_init": self.conda_init,
            "conda_environment": self.conda_env,
        }

    # -- rendering helpers ------------------------------------------------
    def gres(self) -> str:
        """``gpu:<type>:<n>`` / ``gpu:<n>`` / ``""`` when no GPU is requested."""
        if self.gpus <= 0:
            return ""
        return f"gpu:{self.gpu_type}:{self.gpus}" if self.gpu_type else f"gpu:{self.gpus}"


# canonical -> the OpenFE/OpenMM `slurm:` block spelling
CANONICAL_TO_OPENFE = {
    "time": "walltime",
    "mem": "memory",
    "conda_env": "conda_environment",
    "ntasks": "ntasks",
    "gpus": "gpus",
    "partition": "partition",
    "account": "account",
    "nodes": "nodes",
    "cpus_per_task": "cpus_per_task",
    "gpu_type": "gpu_type",
    "replicates": "replicates",
    "modules": "modules",
    "conda_init": "conda_init",
}


def to_openfe_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Translate a canonical mapping to OpenFE ``slurm:`` keys, dropping unknowns.

    Used when only the keys a preset actually declares should be applied, so
    selecting a walltime preset does not silently reset a hand-edited CPU count.
    """
    out: dict[str, Any] = {}
    for key, value in normalize_keys(data).items():
        mapped = CANONICAL_TO_OPENFE.get(key)
        if mapped and value is not None:
            out[mapped] = value
    return out
