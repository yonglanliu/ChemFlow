"""Deprecated shim — the network CLI moved to :mod:`chemflow.cli.network`.

The implementation now lives in ``chemflow.fe.openfe.utils.network_edit`` /
``chemflow.cli.network`` (migrated out of AlchemForge). This re-exports
``add_network_parser`` so the legacy ``alchemforge network`` command keeps working
as an alias; new code should use ``chemflow network ...``.
"""

from __future__ import annotations

from chemflow.cli.network import add_network_parser  # noqa: F401
