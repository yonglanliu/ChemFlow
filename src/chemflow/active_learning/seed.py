"""Seed (round-0) compound selection for active-learning screening.

Round 0 has no surrogate yet, so the seed is chosen by EXPLORATION (coverage),
not prediction. Three methods, selected by ``method``:

  * "random"   — fast random byte-offset sample across the shards.
  * "diverse"  — (DEFAULT) random oversample -> Morgan fingerprints -> RDKit
                 MaxMin pick of the most diverse ``n`` (best scaffold coverage).
  * "scaffold" — random oversample -> Butina cluster -> one representative per
                 cluster (alternative diversity flavor).

An optional ``actives_file`` (SMILES, one per line or 'smiles id') force-includes
known binders in the seed to warm-start the surrogate with positives.

RDKit is imported lazily; if it's missing, "diverse"/"scaffold" fall back to
"random" with a warning. All shard parsing matches the loop (SMILES col / id col;
header skipped; CXSMILES '|...|' blocks stripped).
"""

from __future__ import annotations

import bz2
import gzip
import os
import random
import sys


def _open_text(path):
    """Open a shard as text, transparently decompressing .bz2 / .gz."""
    p = str(path)
    if p.endswith(".bz2"):
        return bz2.open(p, "rt", encoding="utf-8", errors="replace")
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")


def _is_compressed(path) -> bool:
    p = str(path)
    return p.endswith(".bz2") or p.endswith(".gz")


def _parse_record(line: str, smiles_col: int, id_col: int):
    line = line.strip()
    if not line:
        return None
    fields = line.split("\t") if "\t" in line else line.split()
    if max(smiles_col, id_col) >= len(fields):
        return None
    smiles = fields[smiles_col].strip().split()[0]   # strip CXSMILES |...| block
    cid = fields[id_col].strip()
    if not smiles or not cid or smiles.lower() == "smiles":
        return None
    return cid, smiles


def _reservoir_pool(shards, n, smiles_col, id_col, rng, scan_cap=0):
    """Streaming reservoir sample of ``n`` (id, smiles) from (possibly compressed)
    shards — used when byte-offset seeking is impossible (.bz2/.gz).

    Decompresses line by line. ``scan_cap`` bounds how many records are read PER
    SHARD so a seed doesn't have to stream hundreds of GB; with Enamine REAL split
    by heavy-atom count across files, reading a chunk of each shard still spreads
    the pool across the size range. 0 = auto (~40x n per shard).
    """
    per_shard = scan_cap or max(40 * n, 500_000)
    reservoir: list[tuple[str, str]] = []
    seen = 0
    for shard in shards:
        read = 0
        try:
            fh = _open_text(shard)
        except OSError:
            continue
        with fh:
            for line in fh:
                rec = _parse_record(line, smiles_col, id_col)
                if rec is None:
                    continue
                read += 1
                seen += 1
                if len(reservoir) < n:
                    reservoir.append(rec)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < n:
                        reservoir[j] = rec
                if read >= per_shard:
                    break
    # dedupe by id, preserve order
    out: dict[str, str] = {}
    for cid, smi in reservoir:
        out.setdefault(cid, smi)
    return list(out.items())


def _random_offset_pool(shards, n, smiles_col, id_col, rng, scan_cap=0):
    """~uniform random sample of ``n`` (id, smiles).

    Fast random byte-offset seek for plain-text shards; falls back to a streaming
    reservoir sample when any shard is compressed (.bz2/.gz can't be seeked)."""
    if any(_is_compressed(s) for s in shards):
        return _reservoir_pool(shards, n, smiles_col, id_col, rng, scan_cap)
    sizes = [(s, os.path.getsize(s)) for s in shards]
    total = sum(sz for _, sz in sizes)
    if total == 0:
        return []
    out: dict[str, str] = {}
    attempts, cap = 0, n * 30
    while len(out) < n and attempts < cap:
        attempts += 1
        r = rng.randint(0, total - 1)
        shard, off = sizes[0][0], r
        for s, sz in sizes:
            if off < sz:
                shard, off = s, off
                break
            off -= sz
        with open(shard, "rb") as fh:
            fh.seek(off)
            fh.readline()
            raw = fh.readline()
        if not raw:
            continue
        rec = _parse_record(raw.decode("utf-8", "replace"), smiles_col, id_col)
        if rec:
            out[rec[0]] = rec[1]
    return list(out.items())


def _read_actives(path, smiles_col, id_col):
    recs = []
    with _open_text(path) as fh:
        for i, line in enumerate(fh):
            rec = _parse_record(line, smiles_col, id_col)
            if rec is None:                      # allow bare 'SMILES' per line
                s = line.strip().split()
                if s:
                    rec = (f"active_{i}", s[0])
            if rec:
                recs.append(rec)
    return recs


def _fps(records, radius, nbits):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps, keep = [], []
    for cid, smi in records:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(gen.GetFingerprint(mol))
            keep.append((cid, smi))
    return fps, keep


def _diverse(records, n, radius, nbits, seed):
    from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

    fps, keep = _fps(records, radius, nbits)
    if len(keep) <= n:
        return keep
    idx = MaxMinPicker().LazyBitVectorPick(fps, len(fps), n, seed=seed)
    return [keep[i] for i in idx]


def _scaffold(records, n, radius, nbits, cutoff=0.65):
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    fps, keep = _fps(records, radius, nbits)
    if len(keep) <= n:
        return keep
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    picks = []
    for cl in clusters:                      # one representative (centroid) per cluster
        picks.append(keep[cl[0]])
        if len(picks) >= n:
            break
    return picks


def select_seed(shards, n, *, method="diverse", pool=0, smiles_col=0, id_col=1,
                actives_file=None, radius=2, nbits=2048, seed=0, scan_cap=0):
    """Return ``[(id, smiles), ...]`` of up to ``n`` seed compounds.

    ``pool`` is the random oversample MaxMin/Butina pick from (0 = auto, ~20x n).
    ``scan_cap`` bounds records read per shard for compressed libraries (0 = auto).
    Known actives (if any) are force-included and count toward ``n``.
    """
    rng = random.Random(seed)
    actives = _read_actives(actives_file, smiles_col, id_col) if actives_file else []
    need = max(0, n - len(actives))
    method = (method or "diverse").lower()

    if method == "random" or need == 0:
        picks = _random_offset_pool(shards, need, smiles_col, id_col, rng, scan_cap) if need else []
    else:
        p = pool or max(20 * n, 50_000)
        candidates = _random_offset_pool(shards, p, smiles_col, id_col, rng, scan_cap)
        try:
            if method == "diverse":
                picks = _diverse(candidates, need, radius, nbits, seed)
            elif method == "scaffold":
                picks = _scaffold(candidates, need, radius, nbits)
            else:
                raise ValueError(f"unknown seed method: {method!r}")
        except ImportError:
            print(f"[seed] RDKit unavailable; '{method}' falling back to random.",
                  file=sys.stderr)
            picks = candidates[:need]

    # Merge actives first (deduped by id), then fill with the method's picks.
    out: dict[str, str] = {}
    for cid, smi in actives + picks:
        if cid not in out:
            out[cid] = smi
        if len(out) >= n:
            break
    return list(out.items())
