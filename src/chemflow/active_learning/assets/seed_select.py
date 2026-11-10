#!/usr/bin/env python3
"""Diverse (MaxMin) seed selection for active-learning screening.

Picks N chemically diverse compounds from a huge Enamine REAL library WITHOUT
reading all of it: it first draws a large RANDOM pool by seeking to random byte
offsets across the shard files (fast on 10^9+ molecules), computes Morgan
fingerprints, then uses RDKit's MaxMin picker to choose the N most diverse.

Run this on your HPC in the chemflow env (needs RDKit), near screen_db:

    conda activate chemflow
    python seed_select.py \
        --db "screen_db/*.cxsmiles" \
        --n 10000 --pool 300000 \
        --smiles-col 0 --id-col 1 \
        --out seed.smi

Output:
    seed.smi  : 'SMILES<TAB>id' per line  -> feed to VSW / LigPrep (it does prep + dock)
    seed.csv  : 'id,SMILES'               -> convenient record of the picks

Notes:
  * Enamine REAL .cxsmiles is TAB-separated with SMILES in column 0 and the
    idnumber in column 1 (a header line starting 'smiles' is skipped).
  * CXSMILES enhanced-stereo blocks ("SMILES |...|") are stripped to the core
    SMILES so the output has no embedded spaces; LigPrep/Meeko read it fine.
  * --pool is the random oversample MaxMin picks from; bigger pool = more diverse
    seed but more memory/time. ~20-50x N is a good range.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys


def parse_record(line, smiles_col, id_col):
    line = line.strip()
    if not line:
        return None
    fields = line.split("\t") if "\t" in line else line.split()
    if max(smiles_col, id_col) >= len(fields):
        return None
    smiles = fields[smiles_col].strip().split()[0]  # drop CXSMILES |...| block
    cid = fields[id_col].strip()
    if not smiles or not cid or smiles.lower() == "smiles":
        return None
    return cid, smiles


def random_offset_pool(shards, pool, smiles_col, id_col, rng):
    """~uniform random sample of `pool` (id, smiles) via random byte offsets."""
    sizes = [(s, os.path.getsize(s)) for s in shards]
    total = sum(sz for _, sz in sizes)
    if total == 0:
        return []
    out = {}
    attempts = 0
    max_attempts = pool * 30
    while len(out) < pool and attempts < max_attempts:
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
            fh.readline()            # discard partial line
            raw = fh.readline()
        if not raw:
            continue
        rec = parse_record(raw.decode("utf-8", "replace"), smiles_col, id_col)
        if rec:
            out[rec[0]] = rec[1]     # dedupe by id
    return list(out.items())


def maxmin_pick(records, n, radius, nbits, rng_seed):
    """Return the n most diverse records by Morgan-fingerprint MaxMin."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps, keep = [], []
    for cid, smi in records:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        keep.append((cid, smi))
    if not fps:
        raise SystemExit("No valid molecules parsed from the pool.")
    if len(keep) <= n:
        print(f"Pool ({len(keep)} valid) <= n ({n}); taking all.", file=sys.stderr)
        return keep
    picker = MaxMinPicker()
    idx = picker.LazyBitVectorPick(fps, len(fps), n, seed=rng_seed)
    return [keep[i] for i in idx]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Diverse (MaxMin) seed selection.")
    ap.add_argument("--db", required=True, help="glob for library shards, e.g. 'screen_db/*.cxsmiles'")
    ap.add_argument("--n", type=int, default=10000, help="seed size (default 10000)")
    ap.add_argument("--pool", type=int, default=300000, help="random oversample to pick from")
    ap.add_argument("--smiles-col", type=int, default=0)
    ap.add_argument("--id-col", type=int, default=1)
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius")
    ap.add_argument("--nbits", type=int, default=2048, help="fingerprint size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="seed.smi")
    args = ap.parse_args(argv)

    shards = sorted(glob.glob(os.path.expanduser(args.db)))
    if not shards:
        raise SystemExit(f"No shards matched {args.db!r}")
    print(f"Shards: {len(shards)}")
    rng = random.Random(args.seed)

    print(f"Drawing random pool of ~{args.pool} …")
    pool = random_offset_pool(shards, args.pool, args.smiles_col, args.id_col, rng)
    print(f"Pool: {len(pool)} unique compounds")

    print(f"MaxMin picking {args.n} diverse …")
    picks = maxmin_pick(pool, args.n, args.radius, args.nbits, args.seed)
    print(f"Selected: {len(picks)}")

    with open(args.out, "w", encoding="utf-8") as fh:
        for cid, smi in picks:
            fh.write(f"{smi}\t{cid}\n")
    csv_out = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "SMILES"])
        w.writerows((cid, smi) for cid, smi in picks)
    print(f"Wrote {args.out} (SMILES<TAB>id) and {csv_out} (id,SMILES)")
    print("Feed seed.smi into your VSW workflow (LigPrep -> Glide/MM-GBSA).")


if __name__ == "__main__":
    main()
