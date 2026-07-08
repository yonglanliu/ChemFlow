from __future__ import annotations

import argparse

from src.cli.train import add_train_parser


def build_parser():
    parser = argparse.ArgumentParser(
        prog="chemflow",
        description="ChemFlow command line interface",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_train_parser(subparsers)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()