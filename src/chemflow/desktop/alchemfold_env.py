"""Locate the interpreter that runs the AlchemFold CLI.

AlchemFold is a ChemFlow package, not an engine: it writes each engine's input,
launches it and collects the results. The engines live in their OWN conda envs
(``boltz``, ``chai``, ``diffdock``) and are resolved per run by
:func:`alchemfold._exec.prefer_envs_for`, so AlchemFold itself just needs an
interpreter that can import it — normally the ``chemflow`` env you are already
in.

Order: an explicit path (argument or ``CHEMFLOW_ALCHEMFOLD_PYTHON``); the
current interpreter if ``alchemfold`` imports here; then a legacy conda env
named ``alchemfold`` for installs that predate the per-engine split.
"""

from __future__ import annotations

import importlib.util
import sys

from chemflow.desktop.env_locate import find_env_python

_ALCHEMFOLD_ENV_NAMES = ("alchemfold",)          # legacy, kept as a fallback
_ALCHEMFOLD_ENV_VARS = ("CHEMFLOW_ALCHEMFOLD_PYTHON",)


def find_alchemfold_python(preferred: str | None = None) -> str:
    import os
    from pathlib import Path

    for candidate in (preferred, *(os.environ.get(v) for v in _ALCHEMFOLD_ENV_VARS)):
        if candidate and Path(candidate).is_file():
            return candidate
    try:
        if importlib.util.find_spec("alchemfold") is not None:
            return sys.executable
    except (ImportError, ValueError):
        pass
    return find_env_python(_ALCHEMFOLD_ENV_NAMES, _ALCHEMFOLD_ENV_VARS)
