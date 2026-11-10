"""Resolve engine executables when the env isn't activated.

The AlchemFold CLI is often launched by an absolute interpreter path (e.g.
``/data/<user>/conda/envs/alchemfold/bin/python -m alchemfold ...``) or by the
desktop, *without* ``conda activate``. In that case the env's ``bin/`` is not on
``PATH``, so a bare console-script name like ``boltz`` / ``chai-lab`` fails with
``FileNotFoundError``. :func:`resolve_command` rewrites the leading token to the
matching executable in the running interpreter's own ``bin/`` (the same env), or
anything found on ``PATH``, before we hand the command to ``subprocess``.
"""

from __future__ import annotations

import os
import shutil
import sys


def resolve_command(command: list[str]) -> list[str]:
    """Return ``command`` with its first token resolved to a real executable.

    If the first token already contains a path separator it is left untouched.
    Otherwise we look for it (1) next to ``sys.executable`` (the active env's
    ``bin/``), then (2) on ``PATH``. If neither is found the token is returned
    unchanged so the original ``FileNotFoundError`` still surfaces clearly.
    """
    if not command:
        return command
    first = command[0]
    if os.sep in first or (os.altsep and os.altsep in first):
        return command

    bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    candidate = os.path.join(bin_dir, first)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return [candidate, *command[1:]]

    found = shutil.which(first)
    if found:
        return [found, *command[1:]]

    # Also try the env bin without the executable bit check (rare), else keep as-is.
    if os.path.exists(candidate):
        return [candidate, *command[1:]]
    return command
