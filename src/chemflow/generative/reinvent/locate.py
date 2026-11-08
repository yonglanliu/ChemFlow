"""Locate the REINVENT4 ``reinvent`` executable.

REINVENT4 is installed in its own conda env (putting it in the chemflow env
breaks chemflow's dependency pins), so the bare name ``reinvent`` is usually not
on the PATH of the process that launches ChemFlow Studio. This module finds the
binary in the common conda layouts so the desktop can pre-fill it, and callers
get a clear path to set rather than a bare "not found".
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _candidate_roots() -> list[Path]:
    """Directories that may contain conda envs (each an ``<env>/bin/reinvent``)."""
    roots: list[Path] = []
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        p = Path(prefix)
        # Active env is .../envs/<name>; its siblings live in .../envs.
        if p.parent.name == "envs":
            roots.append(p.parent)
        # base env: .../ (envs is a subdir)
        roots.append(p / "envs")
    home = Path.home()
    for base in ("miniforge3", "miniconda3", "anaconda3", "mambaforge", "miniconda", "anaconda"):
        roots.append(home / base / "envs")
    roots.append(Path("/opt/homebrew/Caskroom/miniforge/base/envs"))
    roots.append(Path("/opt/miniconda3/envs"))
    roots.append(Path("/opt/anaconda3/envs"))
    # De-duplicate while keeping order.
    seen, unique = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def find_reinvent_command(preferred: str | None = None) -> str:
    """Return a usable ``reinvent`` command, or ``preferred``/``"reinvent"`` if none found.

    Resolution order: an explicit ``preferred`` or ``$REINVENT_COMMAND`` that
    exists (absolute) or resolves on PATH; then ``reinvent`` on PATH; then an env
    literally named ``reinvent`` under a known conda root; then any env that has a
    ``reinvent`` binary.
    """
    for cand in (preferred, os.environ.get("REINVENT_COMMAND")):
        if not cand:
            continue
        if os.path.isabs(cand) and Path(cand).is_file():
            return cand
        found = shutil.which(cand)
        if found:
            return found

    found = shutil.which("reinvent")
    if found:
        return found

    for root in _candidate_roots():
        exe = root / "reinvent" / "bin" / "reinvent"
        if exe.is_file():
            return str(exe)
    for root in _candidate_roots():
        if root.is_dir():
            for env in sorted(root.iterdir()):
                exe = env / "bin" / "reinvent"
                if exe.is_file():
                    return str(exe)
    return preferred or "reinvent"


def not_found_message(command: str) -> str:
    """A consistent, actionable message when ``command`` cannot be run."""
    return (
        f"REINVENT4 executable not found: {command!r}.\n\n"
        "REINVENT4 installs in its own conda env, so it is usually not on PATH here. "
        "Set the 'REINVENT4 executable' field to its full path, e.g.\n"
        "  <conda-prefix>/envs/reinvent/bin/reinvent\n"
        "Find it with:  conda activate reinvent && which reinvent\n"
        "If you haven't installed it yet, use environment-reinvent.yaml."
    )
