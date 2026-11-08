"""Parse REINVENT4 LinkInvent sampling output into linker candidates.

LinkInvent's sampling CSV has the columns
``SMILES, SMILES_state, Warheads, Linker, NLL`` where ``SMILES`` is the complete
assembled molecule, ``Linker`` the generated linker (with ``*`` attachment
points) and ``NLL`` the model's negative log-likelihood (lower = the model finds
the linker more natural). Pure Python: no RDKit needed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LinkerCandidate:
    index: int                        # 1-based rank in the list
    protac_smiles: str                # complete molecule
    linker_smiles: str
    warheads: str = ""
    nll: float | None = None
    scores: dict[str, Any] = field(default_factory=dict)
    fit_molblock: str | None = None   # best 3D pose of the whole PROTAC, if computed
    notes: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"linker{self.index:04d}"


def _column(header: list[str], *names: str) -> str | None:
    lowered = {h.strip().lower(): h for h in header}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def read_candidates(csv_path: str | Path, *, limit: int | None = None) -> list[LinkerCandidate]:
    """Read LinkInvent output; drop duplicates and unusable rows; sort by NLL."""
    path = Path(csv_path).expanduser()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        smiles_col = _column(header, "SMILES")
        linker_col = _column(header, "Linker")
        if smiles_col is None or linker_col is None:
            raise ValueError(
                f"{path} does not look like LinkInvent output (needs SMILES and Linker "
                f"columns; found {header}). Was model: linkinvent used?"
            )
        warhead_col = _column(header, "Warheads")
        nll_col = _column(header, "NLL")
        state_col = _column(header, "SMILES_state")

        seen: set[tuple[str, str]] = set()
        rows: list[tuple[float, str, str, str]] = []
        for raw in reader:
            smiles = (raw.get(smiles_col) or "").strip()
            linker = (raw.get(linker_col) or "").strip()
            if not smiles or not linker:
                continue
            if state_col and (raw.get(state_col) or "").strip() not in ("", "1"):
                continue                               # 1 = valid in REINVENT4
            key = (smiles, linker)
            if key in seen:
                continue
            seen.add(key)
            try:
                nll = float(raw[nll_col]) if nll_col and raw.get(nll_col) not in (None, "") else None
            except ValueError:
                nll = None
            rows.append((nll if nll is not None else float("inf"), smiles, linker,
                         (raw.get(warhead_col) or "").strip() if warhead_col else ""))

    rows.sort(key=lambda r: r[0])
    candidates = [
        LinkerCandidate(
            index=i,
            protac_smiles=smiles,
            linker_smiles=linker,
            warheads=warheads,
            nll=None if nll == float("inf") else nll,
        )
        for i, (nll, smiles, linker, warheads) in enumerate(rows, start=1)
    ]
    return candidates[:limit] if limit else candidates
