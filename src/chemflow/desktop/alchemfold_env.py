"""Locate the interpreter for the AlchemFold (Boltz-2 / Chai-1) co-folding env.

Thin wrapper over :func:`chemflow.desktop.env_locate.find_env_python`.
"""

from __future__ import annotations

from chemflow.desktop.env_locate import find_env_python

_ALCHEMFOLD_ENV_NAMES = ("alchemfold",)
_ALCHEMFOLD_ENV_VARS = ("CHEMFLOW_ALCHEMFOLD_PYTHON",)


def find_alchemfold_python(preferred: str | None = None) -> str:
    return find_env_python(_ALCHEMFOLD_ENV_NAMES, _ALCHEMFOLD_ENV_VARS, preferred=preferred)
