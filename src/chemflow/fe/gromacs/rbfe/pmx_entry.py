"""Run PMX with the active AlchemForge Python environment."""

from __future__ import annotations

import sys


def main() -> int:
    """Dispatch to PMX after validating its Python-side dependencies."""
    try:
        from rdkit import Chem  # noqa: F401

        import pmx  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "PMX/RDKit import failed in the active interpreter "
            f"{sys.executable}: {error}"
        ) from error

    try:
        from pmx.scripts.cli import entry_point
    except ImportError as error:
        raise SystemExit(
            "The installed PMX package has no pmx.scripts.cli entry point in "
            f"{sys.executable}: {error}"
        ) from error

    result = entry_point()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
