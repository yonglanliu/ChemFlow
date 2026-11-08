"""Generic: find the Python interpreter of a dedicated conda env.

Heavy external tools (OpenFE, PyRosetta, REINVENT4) each live best in their own
conda env. The desktop runs their commands through that env's interpreter. This
resolves it from an explicit path, environment variables, or a conda env whose
name matches, falling back to the current interpreter (same-env install).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        p = Path(prefix)
        if p.parent.name == "envs":
            roots.append(p.parent)
        roots.append(p / "envs")
    home = Path.home()
    for base in ("miniforge3", "miniconda3", "anaconda3", "mambaforge", "miniconda", "anaconda"):
        roots.append(home / base / "envs")
    roots.append(Path("/opt/homebrew/Caskroom/miniforge/base/envs"))
    roots.append(Path("/opt/miniconda3/envs"))
    roots.append(Path("/opt/anaconda3/envs"))
    seen, unique = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def find_env_python(env_names, env_vars=(), preferred: str | None = None) -> str:
    """Return the interpreter for a dedicated env, or ``sys.executable``.

    ``env_names``: conda env names to look for (e.g. ("openfe", "fep")).
    ``env_vars``: environment variables that may hold an explicit python path.
    """
    for cand in (preferred, *(os.environ.get(v) for v in env_vars)):
        if cand and Path(cand).is_file():
            return cand
    for root in _candidate_roots():
        for name in env_names:
            exe = root / name / "bin" / "python"
            if exe.is_file():
                return str(exe)
    return sys.executable
