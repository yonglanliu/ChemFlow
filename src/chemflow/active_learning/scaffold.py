"""Write the cluster-side helper assets (VSW template + wrappers) for a project.

`chemflow active-learn scaffold` copies these into the user's working directory so
they don't have to hand-copy scripts: the VSW input template, the oracle wrapper
(ligand-prep + dock + parse), the pose-score extractor, and a job monitor.
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path

_ASSETS = Path(__file__).parent / "assets"

# Which assets each oracle needs. (Glide/Vina share the VSW-style helpers today;
# only the VSW template is engine-specific.)
_SETS = {
    "vsw": ["vsw_template.inp", "vsw_dock.sh", "extract_scores.py", "vsw_monitor.py"],
    "glide": ["vsw_template.inp", "vsw_dock.sh", "extract_scores.py", "vsw_monitor.py"],
    "seed": ["seed_select.py"],
    "all": ["vsw_template.inp", "vsw_dock.sh", "extract_scores.py",
            "vsw_monitor.py", "seed_select.py"],
}


def write_assets(dest, which: str = "all", *, force: bool = False) -> list[Path]:
    """Copy the asset set ``which`` into ``dest``. Returns the files written."""
    names = _SETS.get(which)
    if names is None:
        raise ValueError(f"unknown asset set {which!r}; choose one of {sorted(_SETS)}")
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in dict.fromkeys(names):          # dedupe, keep order
        src = _ASSETS / name
        dst = dest / name
        if dst.exists() and not force:
            print(f"  skip (exists): {dst}  (use --force to overwrite)")
            continue
        shutil.copyfile(src, dst)
        if name.endswith(".sh"):
            dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        written.append(dst)
        print(f"  wrote {dst}")
    return written
