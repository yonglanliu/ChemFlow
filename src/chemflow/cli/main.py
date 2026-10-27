from __future__ import annotations

import argparse
import sys

from chemflow.cli.gpu_test import add_gpu_test_parser

# ============================================================
# Build CLI parser
# ============================================================

def build_parser():

    # Keep heavyweight model and plotting imports out of module import time so
    # the dedicated GPU diagnostic can start without loading the training stack.
    from chemflow.cli.free_energy import add_free_energy_parser
    from chemflow.cli.generate import add_generate_parser
    from chemflow.cli.predict import add_predict_parser
    from chemflow.cli.search import add_search_parser
    from chemflow.cli.train import add_train_parser
    from chemflow.cli.uncertainty import add_uncertainty_parser

    parser = argparse.ArgumentParser(

        prog="chemflow",

        description=(
            "ChemFlow command line interface"
        ),
    )


    subparsers = parser.add_subparsers(

        dest="command",

        required=True,
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    add_train_parser(
        subparsers
    )


    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    add_generate_parser(
        subparsers
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    add_predict_parser(
        subparsers
    )


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    add_search_parser(
        subparsers
    )


    # --------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------

    add_uncertainty_parser(
        subparsers
    )


    # --------------------------------------------------------
    # Physics-based free-energy workflows
    # --------------------------------------------------------

    add_free_energy_parser(
        subparsers
    )

    # --------------------------------------------------------
    # GPU diagnostics
    # --------------------------------------------------------

    add_gpu_test_parser(
        subparsers
    )


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
