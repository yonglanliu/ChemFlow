"""Locate the interpreter for GROMACS/PMX commands (chemflow ligand-rbfe / rbfe …).

GROMACS desktop runs use the ``alchemforge`` conda env first (it carries OpenFE
1.x, Open Babel, ACPYPE, AmberTools, GROMACS and, once installed, PMX), so
network planning gets a working OpenFE. A dedicated ``gmxfe``/``gromacs`` env is
only a fallback; $CHEMFLOW_GMX_PYTHON overrides everything.
Thin wrapper over :func:`chemflow.desktop.env_locate.find_env_python`.
"""

from __future__ import annotations

import sys

from chemflow.desktop.env_locate import find_env_python
from chemflow.desktop.fe_env import find_fe_python

_GMX_ENV_NAMES = ("alchemforge", "gmxfe", "gromacs", "gmx")
_GMX_ENV_VARS = ("CHEMFLOW_GMX_PYTHON",)


def find_gmx_python(preferred: str | None = None) -> str:
    """Interpreter for GROMACS commands.

    Order: explicit path / $CHEMFLOW_GMX_PYTHON, then the ``alchemforge`` env, then
    a dedicated gmxfe/gromacs env; if none exists, fall back to the shared compute
    env used for OpenFE (alchemforge/openfe).
    """
    result = find_env_python(_GMX_ENV_NAMES, _GMX_ENV_VARS, preferred=preferred)
    if result == sys.executable:          # no dedicated GROMACS env found
        return find_fe_python()
    return result

