"""Locate the DiffDock-L env + repo (own conda env, separate from alchemfold).

Thin wrapper over :func:`chemflow.desktop.env_locate.find_env_python`, plus the location that
``chemflow diffdock-env install`` recorded. Used by the desktop to show whether DiffDock is
installed; the actual run happens in the alchemfold engine, which resolves it the same way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from chemflow.desktop.env_locate import find_env_python

_DIFFDOCK_ENV_NAMES = ("diffdock",)
_DIFFDOCK_ENV_VARS = ("CHEMFLOW_DIFFDOCK_PYTHON",)


def _recorded() -> dict:
    from chemflow.tools_config import _recorded_diffdock

    return _recorded_diffdock()


def find_diffdock_python(preferred: str | None = None) -> str | None:
    """Interpreter of the diffdock env, or ``None`` if it is not installed."""
    rec = _recorded().get("python")
    found = find_env_python(_DIFFDOCK_ENV_NAMES, _DIFFDOCK_ENV_VARS, preferred=preferred or rec)
    # find_env_python falls back to sys.executable when nothing matches — that is NOT DiffDock.
    return None if found == sys.executable and not (rec and Path(rec).is_file()) else found


def find_diffdock_dir() -> str | None:
    """The cloned DiffDock repo folder, or ``None``."""
    cands = [os.environ.get("CHEMFLOW_DIFFDOCK_DIR"), _recorded().get("dir")]
    for parent in Path(__file__).resolve().parents:
        if (parent / "env" / "environment-diffdock.yaml").is_file():
            cands.append(str(parent / "DiffDock"))
            break
    for cand in cands:
        if cand and (Path(cand).expanduser() / "inference.py").is_file():
            return str(Path(cand).expanduser())
    return None


def diffdock_status() -> tuple[bool, str]:
    """``(installed, one-line message)`` for the AI docking page."""
    python, repo = find_diffdock_python(), find_diffdock_dir()
    if python and repo:
        return True, f"DiffDock-L found — env: {python} | repo: {repo}"
    missing = " and ".join(n for n, v in (("the diffdock env", python), ("the DiffDock repo", repo)) if not v)
    return False, (f"DiffDock-L not found ({missing}). Install it on the Environments page "
                   "(card 'DiffDock-L'), then run Warm-up once.")
