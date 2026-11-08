"""Optional 3D pocket-fit check for generated PROTACs.

Both ligands ("heads") sit at known poses in a common coordinate frame (for
example a ternary-complex model). For each candidate we keep the warhead atoms
fixed at those poses, build conformers of the whole PROTAC around them with
RDKit's constrained distance geometry, relax the linker with the warheads held
in place, and report:

* whether any conformer reproduces both poses at all (a linker that is too short
  or too rigid cannot),
* steric clashes between the linker and the protein heavy atoms,
* an approximate strain energy (MMFF energy of the best constrained conformer
  minus the lowest energy found for the free molecule).

This is a quick filter to remove poor linkers, not a binding or degradation
prediction. If the two heads are NOT in a common frame (two separate crystal
structures) the full check is impossible; :func:`exit_clearance` still tells you
whether each exit vector points into open space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from chemflow.generative.reinvent.linker.heads import HeadSpec
from chemflow.generative.reinvent.linker.results import LinkerCandidate

_WATER = {"HOH", "WAT", "H2O", "TIP", "TIP3", "SOL"}


@dataclass
class FitContext:
    head1: HeadSpec
    head2: HeadSpec
    protein_xyz: Any = None           # (N, 3) numpy array of protein heavy atoms, or None
    same_frame: bool = True           # False: heads come from separate structures
    n_confs: int = 60                 # conformers per PROTAC (some are rejected, see _align_to_pose)
    n_reference_confs: int = 20
    clash_cutoff: float = 3.0         # Angstrom, linker heavy atom to protein heavy atom
    pose_tolerance: float = 0.75      # Angstrom RMSD allowed when matching the ligand poses
    seed: int = 7


@dataclass
class FitResult:
    ok: bool | None                   # True fits, False does not, None could not be evaluated
    reason: str = ""
    n_embedded: int = 0
    clashes: int | None = None
    min_dist: float | None = None
    strain: float | None = None
    exit_gap: float | None = None     # distance between the two exit atoms in the pose frame
    molblock: str | None = None       # best 3D pose of the whole PROTAC


# ---------------------------------------------------------------------------
# Protein coordinates (pure Python + numpy)
# ---------------------------------------------------------------------------

def read_pdb_heavy_atoms(paths: str | Path | Sequence[str | Path], *, include_hetatm: bool = False):
    """Heavy-atom coordinates of the protein(s) as an ``(N, 3)`` numpy array.

    Only ``ATOM`` records are read by default, so ligands stored as ``HETATM``
    do not register as clashes with their own linker. Waters and hydrogens are
    always skipped.
    """
    import numpy as np

    if isinstance(paths, (str, Path)):
        paths = [paths]
    coords: list[tuple[float, float, float]] = []
    for path in paths:
        with Path(path).expanduser().open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = line[:6].strip()
                if record != "ATOM" and not (include_hetatm and record == "HETATM"):
                    continue
                if line[17:20].strip().upper() in _WATER:
                    continue
                element = line[76:78].strip().upper() if len(line) >= 78 else ""
                if not element:
                    element = "".join(ch for ch in line[12:16].strip() if ch.isalpha())[:1].upper()
                if element in ("H", "D"):
                    continue
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    continue
    return np.array(coords, dtype=float).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Exit-vector clearance (works even when the heads are in different frames)
# ---------------------------------------------------------------------------

def exit_clearance(head: HeadSpec, protein_xyz, *, reach: float = 6.0, step: float = 0.5) -> float | None:
    """Smallest distance from protein atoms to points along the exit vector.

    Samples the ray leaving the attachment atom out to ``reach`` Angstrom. Small
    values (below ~3 A) mean the linker would have to leave straight into the
    protein.
    """
    if head.exit_origin is None or head.exit_direction is None or protein_xyz is None or len(protein_xyz) == 0:
        return None
    import numpy as np

    origin = np.array(head.exit_origin)
    direction = np.array(head.exit_direction)
    best = float("inf")
    for k in range(1, int(reach / step) + 1):
        point = origin + direction * (k * step)
        best = min(best, float(np.min(np.linalg.norm(np.asarray(protein_xyz) - point, axis=1))))
    return best


# ---------------------------------------------------------------------------
# Constrained 3D fit
# ---------------------------------------------------------------------------

def _head_matches(protac, head: HeadSpec, limit: int = 50):
    # The head molecule keeps its '*' as a wildcard: it must match the first linker
    # atom. Build a query where dummy atoms explicitly match any atom; fall back to
    # the raw molecule if the adjustment is unavailable.
    from rdkit import Chem

    queries = []
    try:
        params = Chem.AdjustQueryParameters.NoAdjustments()
        params.makeDummiesQueries = True
        queries.append(Chem.AdjustQueryProperties(Chem.Mol(head.mol), params))
    except Exception:
        pass
    queries.append(head.mol)
    for query in queries:
        matches = protac.GetSubstructMatches(query, uniquify=False, useChirality=False, maxMatches=limit)
        if matches:
            return matches
    return ()


def _fixed_coordinates(head: HeadSpec, match: Sequence[int]):
    """Map protac atom index -> pose coordinate for every non-dummy head atom."""
    from rdkit.Geometry import Point3D

    conf = head.mol.GetConformer()
    mapping = {}
    for query_idx, protac_idx in enumerate(match):
        if head.mol.GetAtomWithIdx(query_idx).GetAtomicNum() == 0:
            continue                                   # attachment atom is free
        pos = conf.GetAtomPosition(query_idx)
        mapping[int(protac_idx)] = Point3D(pos.x, pos.y, pos.z)
    return mapping


def _real_atoms(head: HeadSpec, match: Sequence[int]) -> set[int]:
    """PROTAC atoms matched to real (non-dummy) head atoms."""
    return {a for q, a in enumerate(match)
            if head.mol.GetAtomWithIdx(q).GetAtomicNum() != 0}


def _choose_matches(protac, head1: HeadSpec, head2: HeadSpec):
    """First pair of head matches that do not share atoms, or ``None``."""
    matches1, matches2 = _head_matches(protac, head1), _head_matches(protac, head2)
    for m1 in matches1:
        real1 = _real_atoms(head1, m1)
        for m2 in matches2:
            if not (real1 & _real_atoms(head2, m2)):
                return m1, m2
    return None


def _kabsch(moving, target):
    """Rotation and shift that best map ``moving`` onto ``target`` (both (N, 3))."""
    import numpy as np

    moving_c, target_c = moving.mean(axis=0), target.mean(axis=0)
    covariance = (moving - moving_c).T @ (target - target_c)
    u, _s, vt = np.linalg.svd(covariance)
    sign = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0      # proper rotation only
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    shift = target_c - rotation @ moving_c
    return rotation, shift


def _align_to_pose(molh, conf_ids: Iterable[int], coord_map: dict, tolerance: float) -> list[int]:
    """Move each conformer onto the ligand poses; drop those that do not match.

    RDKit's ``coordMap`` only constrains the *distances* between the pinned
    atoms, so a conformer comes out in an arbitrary frame (and possibly as the
    mirror image). We rigidly superimpose the pinned atoms onto their pose
    coordinates and keep conformers whose RMSD is within ``tolerance``. Mirror
    images (wrong handedness at an unlabelled warhead stereocentre) fail this
    test and are discarded. Kept conformers get their pinned atoms snapped
    exactly onto the pose.
    """
    import numpy as np
    from rdkit.Geometry import Point3D

    indices = list(coord_map.keys())
    target = np.array([[coord_map[i].x, coord_map[i].y, coord_map[i].z] for i in indices])
    kept: list[int] = []
    for cid in conf_ids:
        conf = molh.GetConformer(cid)
        positions = np.array(conf.GetPositions())
        rotation, shift = _kabsch(positions[indices], target)
        moved = (rotation @ positions.T).T + shift
        rmsd = float(np.sqrt(((moved[indices] - target) ** 2).sum(axis=1).mean()))
        if rmsd > tolerance:
            continue
        moved[indices] = target
        for atom_idx in range(moved.shape[0]):
            x, y, z = (float(v) for v in moved[atom_idx])
            conf.SetAtomPosition(atom_idx, Point3D(x, y, z))
        kept.append(cid)
    return kept


def _optimize_confs(molh, conf_ids: Iterable[int], fixed: Sequence[int]):
    from rdkit.Chem import AllChem

    props = AllChem.MMFFGetMoleculeProperties(molh)
    energies: dict[int, float] = {}
    for cid in conf_ids:
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(molh, props, confId=cid)
        else:
            ff = AllChem.UFFGetMoleculeForceField(molh, confId=cid)
        if ff is None:
            continue
        for idx in fixed:
            ff.AddFixedPoint(int(idx))
        ff.Minimize(maxIts=1000)
        energies[cid] = float(ff.CalcEnergy())
    return energies


def _free_minimum_energy(protac, n_confs: int, seed: int) -> float | None:
    """Lowest force-field energy found for the unconstrained molecule."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    ref = Chem.AddHs(protac)
    cids = list(AllChem.EmbedMultipleConfs(ref, numConfs=n_confs, randomSeed=seed, numThreads=0))
    if not cids:
        return None
    energies = _optimize_confs(ref, cids, [])
    return min(energies.values()) if energies else None


def evaluate_fit(protac_smiles: str, ctx: FitContext) -> FitResult:
    """Run the constrained 3D fit for one complete PROTAC."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    head1, head2 = ctx.head1, ctx.head2
    gap = None
    if head1.exit_origin and head2.exit_origin:
        gap = math.dist(head1.exit_origin, head2.exit_origin)

    if not ctx.same_frame:
        return FitResult(None, "heads are in different coordinate frames; a ternary-complex "
                              "model is needed for the 3D fit", exit_gap=None)
    if not (head1.has_3d and head2.has_3d):
        return FitResult(None, "both heads need 3D coordinates for the 3D fit", exit_gap=gap)

    protac = Chem.MolFromSmiles(protac_smiles)
    if protac is None:
        return FitResult(None, "invalid PROTAC SMILES", exit_gap=gap)

    chosen = _choose_matches(protac, head1, head2)
    if chosen is None:
        return FitResult(None, "could not locate both heads inside the generated molecule", exit_gap=gap)
    coord_map = {**_fixed_coordinates(head1, chosen[0]), **_fixed_coordinates(head2, chosen[1])}

    molh = Chem.AddHs(protac)                          # heavy-atom indices are preserved
    conf_ids = list(AllChem.EmbedMultipleConfs(
        molh, numConfs=ctx.n_confs, coordMap=coord_map, randomSeed=ctx.seed,
        useRandomCoords=True, clearConfs=True, numThreads=0,
    ))
    if not conf_ids:
        return FitResult(False, "no conformer reproduces both ligand poses (linker too short or "
                               "too rigid for this geometry)", n_embedded=0, exit_gap=gap)

    embedded = len(conf_ids)
    conf_ids = _align_to_pose(molh, conf_ids, coord_map, ctx.pose_tolerance)
    if not conf_ids:
        return FitResult(False, f"{embedded} conformer(s) built but none matched both ligand poses "
                               f"within {ctx.pose_tolerance} A (geometry too strained, or wrong "
                               "handedness at a warhead stereocentre; try more conformers)",
                         n_embedded=0, exit_gap=gap)

    fixed = list(coord_map.keys())
    energies = _optimize_confs(molh, conf_ids, fixed)
    if not energies:
        return FitResult(None, "force-field setup failed", n_embedded=len(conf_ids), exit_gap=gap)

    import numpy as np

    linker_atoms = [a.GetIdx() for a in molh.GetAtoms()
                    if a.GetAtomicNum() > 1 and a.GetIdx() not in coord_map]
    protein = ctx.protein_xyz if ctx.protein_xyz is not None and len(ctx.protein_xyz) else None

    scored = []
    for cid, energy in energies.items():
        clashes, min_dist = 0, None
        if protein is not None and linker_atoms:
            positions = molh.GetConformer(cid).GetPositions()[linker_atoms]
            dists = np.linalg.norm(positions[:, None, :] - np.asarray(protein)[None, :, :], axis=2)
            clashes = int((dists < ctx.clash_cutoff).sum())
            min_dist = float(dists.min())
        scored.append((clashes, energy, cid, min_dist))
    scored.sort(key=lambda row: (row[0], row[1]))
    clashes, energy, best_cid, min_dist = scored[0]

    free_min = _free_minimum_energy(protac, ctx.n_reference_confs, ctx.seed)
    strain = max(0.0, energy - free_min) if free_min is not None else None

    return FitResult(
        ok=bool(clashes == 0),
        reason="" if clashes == 0 else f"{clashes} linker-protein contacts closer than {ctx.clash_cutoff} A",
        n_embedded=len(conf_ids),
        clashes=clashes if protein is not None else None,
        min_dist=min_dist,
        strain=strain,
        exit_gap=gap,
        molblock=Chem.MolToMolBlock(molh, confId=best_cid),
    )


def fit_candidates(
    candidates: Iterable[LinkerCandidate],
    ctx: FitContext,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> list[LinkerCandidate]:
    """Run :func:`evaluate_fit` on each candidate, storing ``fit_*`` scores."""
    candidates = list(candidates)
    for i, cand in enumerate(candidates, start=1):
        try:
            result = evaluate_fit(cand.protac_smiles, ctx)
        except Exception as exc:                       # never let one molecule stop the batch
            result = FitResult(None, f"3D fit failed: {exc}")
        cand.scores.update({
            "fit_ok": result.ok,
            "fit_clashes": result.clashes,
            "fit_min_dist": result.min_dist,
            "fit_strain": result.strain,
            "fit_n_conformers": result.n_embedded,
        })
        cand.fit_molblock = result.molblock
        if result.reason:
            cand.notes.append(result.reason)
        if progress:
            progress(i, len(candidates))
    return candidates
