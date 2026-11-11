"""``chemflow alchemfold <args…>`` — run the AlchemFold CLI from the chemflow command.

Forwards everything after ``alchemfold`` to ``python -m alchemfold`` in the
``alchemfold`` conda env (found the same way the desktop finds it), so you do not
have to activate that env or remember its interpreter path::

    chemflow alchemfold gpu-test --gpus 2 --boltz
    chemflow alchemfold screen generate --library enamine_real ...
    chemflow alchemfold screen stage screen.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_alchemfold(args: argparse.Namespace) -> int:
    rest = list(args.alchemfold_args)
    if rest and rest[0] == "--":
        rest = rest[1:]
    try:
        from chemflow.desktop.alchemfold_env import find_alchemfold_python

        python = find_alchemfold_python()
    except Exception:
        python = sys.executable
    return subprocess.call([python, "-m", "alchemfold", *rest])


def add_alchemfold_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "alchemfold",
        help="Run the AlchemFold CLI (dock / screen / gpu-test …) in the alchemfold env",
        add_help=False,  # let `--help` reach alchemfold's own parser
    )
    parser.add_argument("alchemfold_args", nargs=argparse.REMAINDER,
                        help="arguments passed through to `python -m alchemfold`")
    parser.set_defaults(func=run_alchemfold)
