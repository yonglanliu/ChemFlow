"""Launch a desktop backend job in its own Unix process group."""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        raise SystemExit(
            "Usage: python -m alchemforge.desktop.job_runner SCRIPT CONFIG"
        )
    script, config = arguments
    if os.name == "posix":
        os.setsid()
    os.execvp("bash", ["bash", script, config])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
