"""ChemFlow entry point for optional free-energy workflows."""

from __future__ import annotations

import argparse

from alchemforge.gromacs.abfe.workflow import add_abfe_gromacs_parser
from alchemforge.gromacs.resFEP.workflow import add_resfep_parser
from alchemforge.gromacs.rbfe.workflow import add_ligand_rbfe_parser
from alchemforge.gromacs.septop.workflow import add_septop_gromacs_parser


def add_free_energy_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "free-energy",
        help="Run physics-based free-energy workflows",
    )
    workflows = parser.add_subparsers(dest="free_energy_workflow", required=True)
    add_resfep_parser(workflows)
    add_abfe_gromacs_parser(workflows)
    add_ligand_rbfe_parser(workflows)
    add_septop_gromacs_parser(workflows)
    return parser
