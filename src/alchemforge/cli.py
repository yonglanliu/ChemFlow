"""Command-line interface for AlchemForge."""

from __future__ import annotations

import argparse

from alchemforge.gpu_test import add_gpu_test_parser
from alchemforge.generate_cli import add_generate_parser
from alchemforge.openfe_cli import add_openfe_parser
from alchemforge.gromacs.alanine_scan.workflow import add_alanine_scan_parser
from alchemforge.gromacs.abfe.workflow import add_abfe_gromacs_parser
from alchemforge.gromacs.rbfe.prep_cli import add_rbfe_prep_parser
from alchemforge.gromacs.rbfe.workflow import add_ligand_rbfe_parser
from alchemforge.gromacs.resFEP.workflow import add_resfep_parser
from alchemforge.gromacs.septop.workflow import add_septop_gromacs_parser
from alchemforge.structure.pdbfix import add_pdbfix_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alchemforge",
        description="Reproducible alchemical free-energy workflows",
    )
    workflows = parser.add_subparsers(dest="workflow", required=True)
    add_resfep_parser(workflows)
    add_ligand_rbfe_parser(workflows)
    add_rbfe_prep_parser(workflows)
    add_alanine_scan_parser(workflows)
    add_abfe_gromacs_parser(workflows)
    add_septop_gromacs_parser(workflows)
    add_openfe_parser(workflows)
    add_pdbfix_parser(workflows)
    add_generate_parser(workflows)
    add_gpu_test_parser(workflows)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
