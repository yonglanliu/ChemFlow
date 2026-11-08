"""Deprecated shim — the OpenFE CLI moved to :mod:`chemflow.cli.openfe`.

The implementation now lives in ``chemflow.fe.openfe`` / ``chemflow.cli.openfe``
(migrated out of AlchemForge). This module re-exports ``add_openfe_parser`` so the
legacy ``alchemforge openfe`` command keeps working as an alias; new code should
use ``chemflow openfe ...``.
"""

from __future__ import annotations

from chemflow.cli.openfe import add_openfe_parser  # noqa: F401
