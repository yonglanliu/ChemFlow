from __future__ import annotations

import argparse

from chemflow.cli.generate import (
    add_generate_parser,
)

from chemflow.cli.predict import (
    add_predict_parser,
)

from chemflow.cli.search import (
    add_search_parser,
)

from chemflow.cli.train import (
    add_train_parser,
)

from chemflow.cli.uncertainty import (
    add_uncertainty_parser,
)


# ============================================================
# Build CLI parser
# ============================================================

def build_parser():

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


    return parser


# ============================================================
# Main
# ============================================================

def main():

    parser = build_parser()

    args = parser.parse_args()

    if not hasattr(
        args,
        "func",
    ):

        parser.print_help()

        return

    args.func(
        args
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()