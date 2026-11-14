"""AlchemFold entry points exposed on the ``chemflow`` command.

Three subcommands, all thin forwarders to ``python -m alchemfold`` — AlchemFold
remains the single docking/screening implementation, these just save you from
activating its env or remembering its interpreter path::

    chemflow dock generate --engine boltz --protein t.fasta --copies 4 -o dock.yaml
    chemflow dock run dock.yaml
    chemflow screen generate --library enamine_real ...
    chemflow alchemfold gpu-test --gpus 2 --boltz

``chemflow dock``/``chemflow screen`` are exact aliases for ``chemflow
alchemfold dock``/``… screen``; every flag, including the per-engine
``--boltz-env`` / ``--chai-env`` overrides, is passed through untouched.  The
engine itself (Boltz-2, Chai-1, DiffDock-L) is resolved from its own conda env
by AlchemFold at run time, so no engine stack is needed here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _alchemfold_python() -> str:
    """Interpreter that can ``import alchemfold`` (usually this one)."""
    try:
        from chemflow.desktop.alchemfold_env import find_alchemfold_python

        return find_alchemfold_python()
    except Exception:
        return sys.executable


def _forward(rest: list[str], prefix: tuple[str, ...] = ()) -> int:
    rest = list(rest)
    if rest and rest[0] == "--":
        rest = rest[1:]
    return subprocess.call([_alchemfold_python(), "-m", "alchemfold", *prefix, *rest])


def run_alchemfold(args: argparse.Namespace) -> int:
    return _forward(args.alchemfold_args)


def run_dock(args: argparse.Namespace) -> int:
    return _forward(args.alchemfold_args, prefix=("dock",))


def run_screen(args: argparse.Namespace) -> int:
    return _forward(args.alchemfold_args, prefix=("screen",))


def _passthrough_parser(subparsers, name: str, help_text: str, func):
    parser = subparsers.add_parser(
        name,
        help=help_text,
        add_help=False,  # let `--help` reach AlchemFold's own parser
    )
    parser.add_argument("alchemfold_args", nargs=argparse.REMAINDER,
                        help="arguments passed through to `python -m alchemfold`")
    parser.set_defaults(func=func)
    return parser


def add_alchemfold_parser(subparsers) -> None:
    _passthrough_parser(
        subparsers, "alchemfold",
        "Run the AlchemFold CLI (dock / screen / gpu-test …)",
        run_alchemfold,
    )
    _passthrough_parser(
        subparsers, "dock",
        "Co-folding docking with Boltz-2 / Chai-1 / DiffDock-L (alias for `alchemfold dock`)",
        run_dock,
    )
    _passthrough_parser(
        subparsers, "screen",
        "Virtual screening with Boltz-2 / Chai-1 / DiffDock-L (alias for `alchemfold screen`)",
        run_screen,
    )
