"""Export REINVENT4 results for downstream ChemFlow / Alchemforge steps.

Reads a REINVENT output CSV (sampling or staged-learning summary), keeps the
top-N unique molecules and writes ``.smi`` + ``.csv``; with ``sdf=True`` also
3D-embeds them (RDKit ETKDG + MMFF) into an SDF ready for ``alchemforge rbfe``
or ``alchemfold dock``.
"""

from __future__ import annotations

import csv
from pathlib import Path

_SCORE_COLUMNS = ("score", "total_score", "total score")


def _find(header: list[str], candidates) -> str | None:
    lowered = {h.strip().lower(): h for h in header}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def read_results(csv_path: str | Path) -> list[dict]:
    """Return ``[{smiles, score|None, **row}]`` from a REINVENT CSV."""
    path = Path(csv_path).expanduser()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        smi_col = _find(header, ("smiles",))
        if smi_col is None:
            raise ValueError(f"no SMILES column in {path} (columns: {header})")
        score_col = _find(header, _SCORE_COLUMNS)
        rows = []
        for raw in reader:
            smiles = (raw.get(smi_col) or "").strip()
            if not smiles:
                continue
            score = None
            if score_col and raw.get(score_col, "") not in ("", None):
                try:
                    score = float(raw[score_col])
                except ValueError:
                    score = None
            rows.append({"smiles": smiles, "score": score, **raw})
    return rows


def _canonical(smiles: str) -> str | None:
    try:
        from rdkit import Chem
    except ImportError:  # string-level dedupe only
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def select_top(rows: list[dict], top: int = 100, min_score: float | None = None) -> list[dict]:
    """Top-``top`` unique molecules by score (input order if no scores)."""
    has_scores = any(r["score"] is not None for r in rows)
    ordered = sorted(
        rows, key=lambda r: (r["score"] is None, -(r["score"] or 0.0))
    ) if has_scores else list(rows)
    seen: set[str] = set()
    picked: list[dict] = []
    for row in ordered:
        if min_score is not None and (row["score"] is None or row["score"] < min_score):
            continue
        canon = _canonical(row["smiles"])
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        picked.append({**row, "canonical_smiles": canon})
        if len(picked) >= top:
            break
    return picked


def write_sdf(rows: list[dict], path: Path, seed: int = 7) -> int:
    """Embed + MMFF-optimise molecules; returns the number written."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    written = 0
    writer = Chem.SDWriter(str(path))
    try:
        for i, row in enumerate(rows, start=1):
            mol = Chem.MolFromSmiles(row["canonical_smiles"])
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
                continue
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass
            name = f"gen{i:04d}"
            mol.SetProp("_Name", name)
            mol.SetProp("SMILES", row["canonical_smiles"])
            if row["score"] is not None:
                mol.SetProp("reinvent_score", f"{row['score']:.4f}")
            writer.write(mol)
            written += 1
    finally:
        writer.close()
    return written


def export_results(csv_path, out_prefix=None, *, top: int = 100,
                   min_score: float | None = None, sdf: bool = False) -> dict:
    """Write ``<prefix>.smi`` / ``.csv`` (and ``.sdf``); return the paths."""
    src = Path(csv_path).expanduser().resolve()
    prefix = Path(out_prefix).expanduser() if out_prefix else src.with_name(src.stem + "_top")
    picked = select_top(read_results(src), top=top, min_score=min_score)
    if not picked:
        raise ValueError("no molecules passed the selection")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    smi_path = prefix.with_suffix(".smi")
    with smi_path.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(picked, start=1):
            handle.write(f"{row['canonical_smiles']} gen{i:04d}\n")
    csv_out = prefix.with_suffix(".csv")
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "smiles", "score"])
        for i, row in enumerate(picked, start=1):
            writer.writerow([f"gen{i:04d}", row["canonical_smiles"],
                             "" if row["score"] is None else row["score"]])
    result = {"smi": smi_path, "csv": csv_out, "count": len(picked)}
    if sdf:
        sdf_path = prefix.with_suffix(".sdf")
        result["sdf_count"] = write_sdf(picked, sdf_path)
        result["sdf"] = sdf_path
    return result
