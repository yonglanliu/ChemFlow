"""Launch Chemprop after registering ChemFlow's additional losses."""

from __future__ import annotations


def main() -> None:
    # Importing this module registers the additional names before Chemprop
    # constructs its argument parser from LossFunctionRegistry.
    from chemflow.deep_learning.chemprop import losses as _losses  # noqa: F401
    from chemprop.cli.main import main as chemprop_main

    chemprop_main()


if __name__ == "__main__":
    main()
