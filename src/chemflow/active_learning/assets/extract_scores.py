#!/usr/bin/env python3
# Run with:  $SCHRODINGER/run python3 extract_scores.py POSES_pv.maegz OUT.csv [BATCH.smi]
#
# Extracts id, SMILES, GlideScore from a Glide pose-viewer (_pv.maegz) file and
# writes one row per compound (the BEST / most negative GlideScore across its
# LigPrep variants). Columns: id,SMILES,glide_score
#
#   id          = s_vsw_compound_code (the title from your batch .smi)
#   glide_score = r_i_docking_score   (GlideScore; more negative = better)
#   SMILES      = from the input BATCH.smi if given (keeps it consistent with the
#                 library, recommended for the ML surrogate), else regenerated
#                 from the docked 3D pose (the prepared/protonated species).

import sys

from schrodinger.structure import StructureReader
from schrodinger.structutils import analyze

poses = sys.argv[1]
out = sys.argv[2]
batch = sys.argv[3] if len(sys.argv) > 3 else None

# Optional: original input SMILES keyed by id (from 'SMILES<TAB>id' batch file).
orig = {}
if batch:
    with open(batch) as fh:
        for line in fh:
            parts = line.replace("\t", " ").split()
            if len(parts) >= 2:
                orig[parts[1]] = parts[0]

best = {}   # id -> (score, smiles)
for st in StructureReader(poses):
    p = st.property
    score = p.get("r_i_docking_score")
    if score is None:          # receptor / non-docked entries have no score
        continue
    cid = p.get("s_vsw_compound_code") or st.title
    if cid in orig:
        smi = orig[cid]
    else:
        try:
            smi = analyze.generate_smiles(st)
        except Exception:
            smi = ""
    if cid not in best or score < best[cid][0]:
        best[cid] = (score, smi)

with open(out, "w") as fh:
    fh.write("id,SMILES,glide_score\n")
    for cid, (score, smi) in best.items():
        fh.write("%s,%s,%s\n" % (cid, smi, score))

print("wrote %d compounds -> %s" % (len(best), out))
