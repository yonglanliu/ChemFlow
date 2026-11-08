"""Library-based linker design: search a curated set of real linkers.

Instead of inventing linkers with LinkInvent, this mode takes a library of
*bifunctional* linkers — each a SMILES with exactly two attachment points
(``*``), e.g. Enamine's PROTAC linker catalogue — assembles each one between the
two clicked warheads, and feeds the assembled PROTACs into the same scoring +
3D-fit pipeline used for the generated linkers.

The library file is supplied by the user (Enamine catalogues are proprietary and
cannot be bundled or fetched here). Supported inputs: ``.smi``/``.txt``/``.csv``
(one SMILES per line, optional name in the second column/field) and ``.sdf``.

RDKit is imported lazily, so the library can be *loaded* and textually filtered
without it; assembly needs RDKit.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Sequence

from chemflow.generative.reinvent.linker.heads import HeadSpec
from chemflow.generative.reinvent.linker.results import LinkerCandidate


# ---------------------------------------------------------------------------
# Loading a linker library (textual: keeps entries with two attachment points)
# ---------------------------------------------------------------------------

def _looks_bifunctional(smiles: str) -> bool:
    """A linker needs exactly two attachment points (``*``/``[*]``/``[*:n]``)."""
    return smiles.count("*") == 2


def load_linker_library(path: str | Path, *, limit: int | None = None) -> list[tuple[str, str]]:
    """Return ``[(smiles, name), ...]`` for linkers with two attachment points.

    ``.sdf`` files are parsed with RDKit (lazy); line formats (``.smi``/``.csv``/
    ``.txt``) are read textually: the first whitespace/comma field is the SMILES,
    an optional second field is the name.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"linker library not found: {path}")

    entries: list[tuple[str, str]] = []
    suffix = path.suffix.lower()
    if suffix in (".sdf", ".sd"):
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(str(path))
        for i, mol in enumerate(supplier, start=1):
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol)
            if not _looks_bifunctional(smiles):
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"linker{i:05d}"
            entries.append((smiles, name or f"linker{i:05d}"))
            if limit and len(entries) >= limit:
                break
        return entries

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        delimiter = "," if suffix == ".csv" else None
        reader = csv.reader(handle, delimiter=delimiter) if delimiter else (
            (line.split() for line in handle))
        for i, row in enumerate(reader, start=1):
            if not row:
                continue
            smiles = str(row[0]).strip()
            if not smiles or smiles.startswith("#") or smiles.lower() in ("smiles", "smi"):
                continue
            if not _looks_bifunctional(smiles):
                continue
            name = str(row[1]).strip() if len(row) > 1 and str(row[1]).strip() else f"linker{i:05d}"
            entries.append((smiles, name))
            if limit and len(entries) >= limit:
                break
    return entries


# ---------------------------------------------------------------------------
# Assembly: warhead + bifunctional linker + warhead  ->  whole PROTAC
# ---------------------------------------------------------------------------

def _attach_head(fragment, head):
    """Join ``head`` (one dummy) onto the first dummy of ``fragment``.

    Any further dummies on ``fragment`` are preserved, so calling this twice on a
    two-dummy linker (once per warhead) yields the complete molecule.
    """
    from rdkit import Chem

    n_frag = fragment.GetNumAtoms()
    combo = Chem.CombineMols(fragment, head)
    rw = Chem.RWMol(combo)

    frag_dummy = next(a.GetIdx() for a in rw.GetAtoms()
                      if a.GetAtomicNum() == 0 and a.GetIdx() < n_frag)
    head_dummy = next(a.GetIdx() for a in rw.GetAtoms()
                      if a.GetAtomicNum() == 0 and a.GetIdx() >= n_frag)
    frag_nbr = rw.GetAtomWithIdx(frag_dummy).GetNeighbors()[0].GetIdx()
    head_nbr = rw.GetAtomWithIdx(head_dummy).GetNeighbors()[0].GetIdx()

    rw.AddBond(frag_nbr, head_nbr, Chem.BondType.SINGLE)
    for idx in sorted((frag_dummy, head_dummy), reverse=True):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def assemble_protac(head1_smiles: str, head2_smiles: str, linker_smiles: str) -> str:
    """Connect two warheads (one ``*`` each) through a linker (two ``*``)."""
    from rdkit import Chem

    linker = Chem.MolFromSmiles(linker_smiles)
    head1 = Chem.MolFromSmiles(head1_smiles)
    head2 = Chem.MolFromSmiles(head2_smiles)
    if linker is None:
        raise ValueError(f"invalid linker SMILES: {linker_smiles}")
    if head1 is None or head2 is None:
        raise ValueError("invalid warhead SMILES")
    if sum(1 for a in linker.GetAtoms() if a.GetAtomicNum() == 0) != 2:
        raise ValueError(f"linker needs exactly two attachment points: {linker_smiles}")

    intermediate = _attach_head(linker, head1)      # one attachment point left
    full = _attach_head(intermediate, head2)        # fully assembled
    return Chem.MolToSmiles(full, isomericSmiles=False)


# ---------------------------------------------------------------------------
# Search + orchestration
# ---------------------------------------------------------------------------

def search_library(
    head1: HeadSpec,
    head2: HeadSpec,
    linkers: Sequence[tuple[str, str]],
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> list[LinkerCandidate]:
    """Assemble each library linker with the two warheads into a candidate.

    Optional ``min_length``/``max_length`` filter on the linker's effective bond
    length (shortest path between attachment points) keep only linkers that can
    plausibly span the warhead geometry.
    """
    from chemflow.generative.reinvent.linker.scores import linker_metrics

    candidates: list[LinkerCandidate] = []
    seen: set[str] = set()
    skipped = 0
    for smiles, name in linkers:
        if min_length is not None or max_length is not None:
            try:
                length = linker_metrics(smiles).get("linker_length")
            except Exception:
                length = None
            if length is not None:
                if min_length is not None and length < min_length:
                    continue
                if max_length is not None and length > max_length:
                    continue
        try:
            protac = assemble_protac(head1.smiles, head2.smiles, smiles)
        except Exception:
            skipped += 1
            continue
        if protac in seen:
            continue
        seen.add(protac)
        cand = LinkerCandidate(
            index=len(candidates) + 1,
            protac_smiles=protac,
            linker_smiles=smiles,
            warheads=f"{head1.smiles}|{head2.smiles}",
        )
        cand.scores["library_name"] = name
        candidates.append(cand)
        if limit and len(candidates) >= limit:
            break
    if skipped:
        log(f"{skipped} library linker(s) could not be assembled and were skipped")
    return candidates


def design_from_library(
    head1: HeadSpec,
    head2: HeadSpec,
    library_path: str | Path,
    *,
    models=None,
    fit=None,
    fit_top: int = 50,
    min_length: int | None = None,
    max_length: int | None = None,
    limit: int | None = 2000,
    log: Callable[[str], None] = print,
    progress=None,
) -> list[LinkerCandidate]:
    """Library counterpart to ``design_linkers``: search, score, optional 3D fit."""
    from chemflow.generative.reinvent.linker.designer import fit_candidates, score_candidates

    linkers = load_linker_library(library_path)
    log(f"{len(linkers)} bifunctional linkers in {Path(library_path).name}")
    candidates = search_library(head1, head2, linkers, min_length=min_length,
                                max_length=max_length, limit=limit, log=log)
    log(f"{len(candidates)} PROTACs assembled from the library")
    if not candidates:
        return candidates
    log("scoring...")
    score_candidates(candidates, models=models)
    if fit is not None and fit_top > 0:
        subset = candidates[:fit_top]
        log(f"3D pocket fit for the top {len(subset)} library PROTACs...")
        fit_candidates(subset, fit, progress=progress)
    return candidates
