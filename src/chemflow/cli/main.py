from __future__ import annotations

import argparse
import importlib
import os
import sys

from chemflow.cli.gpu_test import add_gpu_test_parser

# ============================================================
# Build CLI parser
# ============================================================

# (module, function) for every subcommand parser. Registered independently so a
# command whose optional dependencies are missing is simply skipped — this lets
# the free-energy commands run in a lean OpenFE/GROMACS env without the ML stack
# (torch/chemprop), and the ML commands run without the FE stack.
_SUBCOMMANDS = [
    # ML / data-science (need the training stack)
    ("chemflow.cli.train", "add_train_parser"),
    ("chemflow.cli.generate", "add_generate_parser"),
    ("chemflow.cli.predict", "add_predict_parser"),
    ("chemflow.cli.search", "add_search_parser"),
    ("chemflow.cli.uncertainty", "add_uncertainty_parser"),
    # Physics-based free energy (OpenFE / GROMACS / OpenMM)
    ("chemflow.cli.free_energy", "add_free_energy_parser"),
    ("chemflow.cli.openfe", "add_openfe_parser"),
    ("chemflow.cli.network", "add_network_parser"),
    ("chemflow.cli.openmm", "add_openmm_parser"),
    ("chemflow.cli.tools", "add_tools_parser"),
    ("chemflow.fe.gromacs.rbfe.workflow", "add_ligand_rbfe_parser"),
    ("chemflow.fe.gromacs.rbfe.prep_cli", "add_rbfe_prep_parser"),
    ("chemflow.fe.gromacs.abfe.workflow", "add_abfe_gromacs_parser"),
    ("chemflow.fe.gromacs.septop.workflow", "add_septop_gromacs_parser"),
    ("chemflow.fe.gromacs.resFEP.workflow", "add_resfep_parser"),
    ("chemflow.fe.gromacs.alanine_scan.workflow", "add_alanine_scan_parser"),
    # PROTAC + environment setup
    ("chemflow.cli.protac", "add_protac_parser"),
    ("chemflow.cli.fe_env", "add_fe_env_parser"),
    ("chemflow.cli.alchemfold_env", "add_alchemfold_env_parser"),
    ("chemflow.cli.gmx_env", "add_gmx_env_parser"),
    ("chemflow.cli.rosetta_env", "add_rosetta_env_parser"),
    ("chemflow.cli.reinvent_env", "add_reinvent_env_parser"),
]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="chemflow",
        description="ChemFlow command line interface",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    debug = bool(os.environ.get("CHEMFLOW_CLI_DEBUG"))
    for module, func_name in _SUBCOMMANDS:
        try:
            mod = importlib.import_module(module)
            getattr(mod, func_name)(subparsers)
        except Exception as exc:  # optional deps missing in this env -> skip command
            if debug:
                print(f"[chemflow] skipped {module} ({exc})", file=sys.stderr)

    add_gpu_test_parser(subparsers)   # always available (lightweight)
    return parser



def build_gpu_test_parser():
    """Build the lightweight parser used by ``chemflow gpu-test``."""

    parser = argparse.ArgumentParser(
        prog="chemflow",
        description="ChemFlow command line interface",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_gpu_test_parser(subparsers)
    return parser


# ============================================================
# Main
# ============================================================

def main(argv: list[str] | None = None):

    # Apply the tools-config links (chemflow_tools.yaml) so external-tool env
    # vars are set automatically; explicit shell values still take precedence.
    try:
        from chemflow.tools_config import apply_env
        apply_env()
    except Exception:
        pass

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = (
        build_gpu_test_parser()
        if arguments and arguments[0] == "gpu-test"
        else build_parser()
    )

    args = parser.parse_args(arguments)

    if not hasattr(
        args,
        "func",
    ):

        parser.print_help()

        return

    return args.func(
        args
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    raise SystemExit(main())
