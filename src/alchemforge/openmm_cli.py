"""Deprecated shim — the OpenMM modeling/MD CLI moved to :mod:`chemflow.cli.openmm`.

The implementation now lives in ``chemflow.fe.openmm_modeling`` /
``chemflow.cli.openmm`` (migrated out of AlchemForge). This re-exports
``add_openmm_parser`` so the legacy ``alchemforge openmm`` command keeps working
as an alias; new code should use ``chemflow openmm ...``.
"""

from __future__ import annotations

from chemflow.cli.openmm import add_openmm_parser  # noqa: F401
