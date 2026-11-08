"""Screen docked ternary models: rank by interface score, then check linkability.

Ties the RosettaDock/PyRosetta output to the linker-design decision in one step:
read ``score.csv``, take the best models by ``I_sc``, and for each measure the
through-space distance between the two warhead attachment atoms (which were
carried into every model). The models that are both low-``I_sc`` and linkable are
the ones to design a linker for.

Pure Python (math via linkability helpers); no RDKit/Rosetta needed.
"""

from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from chemflow.protac.linkability import DEFAULT_MAX_DIST, DEFAULT_MIN_DIST, attachment_coord, distance_verdict


@dataclass
class ScreenRow:
    model: str
    i_sc: float | None
    total: float | None
    distance: float | None
    verdict: str


def _parse_atom_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"atom spec must be RESNAME:ATOMNAME, got {spec!r}")
    resname, atom = spec.split(":", 1)
    if not resname.strip() or not atom.strip():
        raise ValueError(f"atom spec must be RESNAME:ATOMNAME, got {spec!r}")
    return resname.strip(), atom.strip()


def _read_scores(csv_path: Path) -> list[tuple[str, float | None, float | None]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        model_c = cols.get("model")
        isc_c = cols.get("i_sc")
        total_c = cols.get("total_score") or cols.get("total")
        if not model_c:
            raise ValueError(f"{csv_path} has no 'model' column (columns: {reader.fieldnames})")
        rows = []
        for raw in reader:
            def _num(col):
                if not col or raw.get(col) in (None, ""):
                    return None
                try:
                    return float(raw[col])
                except ValueError:
                    return None
            rows.append((raw[model_c].strip(), _num(isc_c), _num(total_c)))
    return rows


def screen_models(
    dock_dir: str | Path,
    atom1: str,
    atom2: str,
    *,
    top: int = 20,
    min_dist: float = DEFAULT_MIN_DIST,
    max_dist: float = DEFAULT_MAX_DIST,
    models_dir: str | Path | None = None,
    linkable_dir: str | Path | None = None,
) -> list[ScreenRow]:
    """Rank ``score.csv`` by I_sc, then measure warhead distance per model.

    ``dock_dir`` may be a folder (uses ``<dir>/score.csv``) or a path to a
    ``score.csv`` file directly. Model PDBs are looked up in ``models_dir`` if
    given, else next to the CSV. ``atom1`` / ``atom2`` are ``RESNAME:ATOMNAME``
    for the two warhead attachment atoms. Writes ``screen.csv`` and returns the
    rows (best I_sc first).
    """
    dock_dir = Path(dock_dir).expanduser()
    csv_path = dock_dir / "score.csv" if dock_dir.is_dir() else dock_dir
    if not csv_path.is_file():
        raise FileNotFoundError(f"score.csv not found: {csv_path}")
    base = Path(models_dir).expanduser() if models_dir else csv_path.parent
    if not base.is_dir():
        raise FileNotFoundError(f"models directory not found: {base}")
    res1, name1 = _parse_atom_spec(atom1)
    res2, name2 = _parse_atom_spec(atom2)

    scored = _read_scores(csv_path)
    # Best (most negative) I_sc first; missing I_sc sinks to the bottom.
    scored.sort(key=lambda r: (r[1] is None, r[1] if r[1] is not None else math.inf))
    scored = scored[:top] if top else scored

    rows: list[ScreenRow] = []
    for model, i_sc, total in scored:
        pdb = base / model
        distance = None
        verdict = "model not found"
        if pdb.is_file():
            text = pdb.read_text(encoding="utf-8", errors="replace")
            c1 = attachment_coord(text, res1, name1)
            c2 = attachment_coord(text, res2, name2)
            if c1 is None or c2 is None:
                verdict = f"attachment atom not found ({atom1 if c1 is None else atom2})"
            else:
                distance = math.dist(c1, c2)
                verdict = distance_verdict(distance, min_dist=min_dist, max_dist=max_dist)
        rows.append(ScreenRow(model, i_sc, total, distance, verdict))

    out = csv_path.parent / "screen.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "I_sc", "total_score", "warhead_distance", "verdict"])
        for r in rows:
            writer.writerow([
                r.model,
                "" if r.i_sc is None else f"{r.i_sc:.2f}",
                "" if r.total is None else f"{r.total:.2f}",
                "" if r.distance is None else f"{r.distance:.2f}",
                r.verdict,
            ])

    # Collect the linkable models' PDBs into their own folder.
    link_dir = Path(linkable_dir).expanduser() if linkable_dir else csv_path.parent / "linkable"
    copied = 0
    for r in rows:
        if r.verdict == "linkable":
            src = base / r.model
            if src.is_file():
                link_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, link_dir / src.name)
                copied += 1
    return rows
