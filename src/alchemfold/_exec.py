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


def _sibling_env_executables(name: str):
    """Yield ``bin/<name>`` paths in sibling conda envs of the running interpreter.

    The desktop runs in the ``chemflow`` env, but the docking engines (``boltz`` /
    ``chai-lab``) live in the ``alchemfold`` env. When ``sys.executable`` is inside
    a conda ``envs/<x>/bin`` tree, search every sibling env's ``bin/`` for ``name``,
    trying an env literally called ``alchemfold`` first.
    """
    exe = os.path.abspath(sys.executable)
    parts = exe.split(os.sep)
    if "envs" not in parts:
        return
    envs_root = os.sep.join(parts[: len(parts) - 1 - parts[::-1].index("envs") + 1])
    try:
        env_names = sorted(os.listdir(envs_root))
    except OSError:
        return
    # Prefer an env named 'alchemfold', then the rest alphabetically.
    env_names.sort(key=lambda n: (n != "alchemfold", n))
    for env_name in env_names:
        candidate = os.path.join(envs_root, env_name, "bin", name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            yield candidate


def resolve_command(command: list[str]) -> list[str]:
    """Return ``command`` with its first token resolved to a real executable.

    If the first token already contains a path separator it is left untouched.
    Otherwise we look for it (1) next to ``sys.executable`` (the active env's
    ``bin/``), then (2) on ``PATH``, then (3) in a sibling conda env (preferring
    ``alchemfold``) — so a desktop run from the ``chemflow`` env still finds
    ``boltz`` / ``chai-lab`` in the ``alchemfold`` env. If nothing matches the
    token is returned unchanged so the original ``FileNotFoundError`` still
    surfaces clearly.
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

    # A sibling conda env (e.g. alchemfold) may hold the engine executable even
    # though the running interpreter (e.g. chemflow) does not.
    for sibling in _sibling_env_executables(first):
        return [sibling, *command[1:]]

    # Also try the env bin without the executable bit check (rare), else keep as-is.
    if os.path.exists(candidate):
        return [candidate, *command[1:]]
    return command
