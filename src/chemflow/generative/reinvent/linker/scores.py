"""Scores shown next to each generated linker.

Every function fails soft: a score that cannot be computed is ``None`` and a
note is recorded on the candidate, so one odd molecule never breaks the table.
RDKit (and ChemFlow's predictor) are imported lazily.
"""

from __future__ import annotations

import itertools
import os
import sys
from typing import Any, Iterable, Sequence

from chemflow.generative.reinvent.linker.results import LinkerCandidate

# (key, column label, python format) - the order used by the CLI and the GUIs.
LINKER_COLUMNS = [
    ("linker_length", "Linker length (bonds)", "{:d}"),
    ("linker_heavy", "Linker heavy atoms", "{:d}"),
    ("linker_rotb", "Linker rot. bonds", "{:d}"),
    ("linker_rings", "Linker rings", "{:d}"),
]
PROPERTY_COLUMNS = [
    ("mw", "MW", "{:.1f}"),
    ("logp", "cLogP", "{:.2f}"),
    ("tpsa", "TPSA", "{:.1f}"),
    ("hbd", "HBD", "{:d}"),
    ("hba", "HBA", "{:d}"),
    ("rotb", "Rot. bonds", "{:d}"),
    ("qed", "QED", "{:.2f}"),
    ("sa", "SA score", "{:.2f}"),
]
FIT_COLUMNS = [
    ("fit_ok", "3D fit", "{}"),
    ("fit_clashes", "Clashes", "{:d}"),
    ("fit_min_dist", "Min dist (A)", "{:.2f}"),
    ("fit_strain", "Strain (kcal/mol)", "{:.1f}"),
]


def format_value(value: Any, fmt: str) -> str:
    if value is None:
        return "n/a"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


# ---------------------------------------------------------------------------
# Linker metrics (from the 'Linker' SMILES with '*' attachment points)
# ---------------------------------------------------------------------------

def linker_metrics(linker_smiles: str) -> dict[str, Any]:
    """Length / flexibility of a linker given as e.g. ``*CCOCC*``.

    ``linker_length`` follows REINVENT4's FragmentEffectiveLength: the number of
    bonds on the shortest path between the two atoms that carry the attachment
    points, so the same transform thresholds can be reused as RL rewards.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(linker_smiles)
    if mol is None:
        raise ValueError(f"cannot parse linker SMILES: {linker_smiles}")

    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    anchors = []
    for dummy in dummies:
        neighbours = dummy.GetNeighbors()
        if neighbours:
            anchors.append(neighbours[0].GetIdx())

    length = None
    if len(anchors) >= 2:
        lengths = []
        for i, j in itertools.combinations(anchors, 2):
            lengths.append(0 if i == j else len(Chem.GetShortestPath(mol, i, j)) - 1)
        length = int(min(lengths))

    editable = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in dummies), reverse=True):
        editable.RemoveAtom(idx)                      # cap the attachment points with H
    frag = editable.GetMol()
    Chem.SanitizeMol(frag)

    return {
        "linker_length": length,
        "linker_heavy": int(frag.GetNumHeavyAtoms()),
        "linker_rotb": int(rdMolDescriptors.CalcNumRotatableBonds(frag)),
        "linker_rings": int(rdMolDescriptors.CalcNumRings(frag)),
    }


# ---------------------------------------------------------------------------
# Whole-PROTAC properties
# ---------------------------------------------------------------------------

def _sa_score(mol) -> float | None:
    try:
        from rdkit.Chem import RDConfig

        folder = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if folder not in sys.path:
            sys.path.append(folder)
        import sascorer  # type: ignore

        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def protac_properties(smiles: str) -> dict[str, Any]:
    """Descriptors of the complete molecule (PROTACs sit far outside Lipinski)."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse SMILES: {smiles}")
    try:
        qed = float(QED.qed(mol))
    except Exception:
        qed = None
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
        "rotb": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "qed": qed,
        "sa": _sa_score(mol),
    }


# ---------------------------------------------------------------------------
# ChemFlow model predictions
# ---------------------------------------------------------------------------

def predict_with_chemflow(smiles: Sequence[str], model_path: str, column: str = "prediction") -> list[float | None]:
    """Run one ChemFlow model on many SMILES; ``None`` where featurisation failed."""
    from chemflow.machine_learning.predict.predictor import ChemFlowPredictor

    predictor = ChemFlowPredictor(model_path)
    values: list[float | None] = [None] * len(smiles)
    try:
        frame = predictor.predict(list(smiles))
    except ValueError:
        return values
    if column not in frame.columns:
        raise ValueError(f"column '{column}' not in predictor output {list(frame.columns)}")
    for index, value in zip(frame["input_index"], frame[column]):
        value = float(value)
        values[int(index)] = None if value != value else value      # NaN -> None
    return values


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def score_candidates(
    candidates: Iterable[LinkerCandidate],
    *,
    models: Sequence[dict[str, Any]] | None = None,
    properties: bool = True,
    linker: bool = True,
) -> list[LinkerCandidate]:
    """Fill ``candidate.scores`` in place.

    ``models`` is a list of ``{"name": ..., "path": ..., "column": "prediction"}``;
    each adds a ``model:<name>`` score computed on the complete PROTAC.
    """
    candidates = list(candidates)
    for cand in candidates:
        if linker:
            try:
                cand.scores.update(linker_metrics(cand.linker_smiles))
            except Exception as exc:
                cand.notes.append(f"linker metrics unavailable: {exc}")
        if properties:
            try:
                cand.scores.update(protac_properties(cand.protac_smiles))
            except Exception as exc:
                cand.notes.append(f"properties unavailable: {exc}")

    for spec in models or []:
        name = str(spec["name"])
        try:
            values = predict_with_chemflow(
                [c.protac_smiles for c in candidates], str(spec["path"]),
                str(spec.get("column") or "prediction"),
            )
        except Exception as exc:
            for cand in candidates:
                cand.notes.append(f"model {name} failed: {exc}")
            values = [None] * len(candidates)
        for cand, value in zip(candidates, values):
            cand.scores[f"model:{name}"] = value
    return candidates
