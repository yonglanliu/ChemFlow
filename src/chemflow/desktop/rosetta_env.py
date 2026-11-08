"""Locate the interpreter for PyRosetta-backed PROTAC commands (pyrosetta-dock, KIC).

Thin wrapper over :func:`chemflow.desktop.env_locate.find_env_python`.
"""

from __future__ import annotations

from chemflow.desktop.env_locate import find_env_python

_ROSETTA_ENV_NAMES = ("pyrosetta", "rosetta")
_ROSETTA_ENV_VARS = ("CHEMFLOW_ROSETTA_PYTHON", "PYROSETTA_PYTHON")


def find_rosetta_python(preferred: str | None = None) -> str:
    return find_env_python(_ROSETTA_ENV_NAMES, _ROSETTA_ENV_VARS, preferred=preferred)
