"""``chemflow gmx-env`` — alias of ``chemflow fe-env``.

GROMACS/PMX RBFE, OpenFE and OpenMM all run from the single ``alchemforge`` conda
env, so there is one installer. ``gmx-env`` is kept only so existing scripts and
the AlchemForge desktop keep working; it shares the pinned package list, CUDA
handling and PMX step with :mod:`chemflow.cli.fe_env`.

    chemflow gmx-env status                # same as: chemflow fe-env status
    chemflow gmx-env install --cuda 12.6   # same as: chemflow fe-env install --cuda 12.6
"""

from __future__ import annotations

from chemflow.cli.fe_env import (  # noqa: F401  (re-exported for callers/tests)
    FE_CONDA_SPECS,
    add_fe_env_parser,
    cuda_specs,
    install_commands,
    pmx_command,
)


def add_gmx_env_parser(subparsers) -> None:
    add_fe_env_parser(
        subparsers, name="gmx-env",
        help="Alias of fe-env: create/link the compute conda env (OpenFE + GROMACS + PMX)")
