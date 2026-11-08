"""Locate the interpreter for free-energy (OpenFE / AlchemForge) commands.

Thin wrapper over :func:`chemflow.desktop.env_locate.find_env_python`.
"""

from __future__ import annotations

from chemflow.desktop.env_locate import find_env_python

_FE_ENV_NAMES = ("alchemforge", "openfe", "fep")
_FE_ENV_VARS = ("CHEMFLOW_FE_PYTHON", "ALCHEMFORGE_PYTHON")


def find_fe_python(preferred: str | None = None) -> str:
    return find_env_python(_FE_ENV_NAMES, _FE_ENV_VARS, preferred=preferred)
