#!/bin/bash
# vsw_dock.sh BATCH_SMI RECEPTOR_GRID OUT_CSV
#
# Oracle wrapper for the active-learning loop. Each round it:
#   1. renders vsw_template.inp, substituting @LIGANDS@ = this round's batch.smi
#      (SMILES<TAB>id) and @GRID@ = the Glide grid .zip,
#   2. runs Schrodinger VSW (LigPrep -> SampleRings -> QikProp -> Glide SP),
#      BLOCKING until it finishes (-WAIT),
#   3. parses the SP poses to CSV 'id,score' (best GlideScore per compound),
#      keyed on s_vsw_compound_code (= the id/title from the .smi).
#
# Point the al.yaml oracle at it:
#   oracle: vsw
#   oracle_command: "bash /path/to/vsw_dock.sh {batch} {receptor} {out}"
#   modules: [schrodinger]         # or set $SCHRODINGER yourself
#
# Env overrides:
#   VSW_TEMPLATE  path to vsw_template.inp (default: next to this script)
#   VSW_NJOBS     glide/ligprep subjobs (default 1; raise for a real 10k batch)
#   VSW_HOST      host:cpu for subjobs (default localhost:1)

set -euo pipefail

BATCH="$1"; GRID="$2"; OUT="$3"
: "${SCHRODINGER:?set SCHRODINGER or 'module load schrodinger' first}"
TEMPLATE="${VSW_TEMPLATE:-$(cd "$(dirname "$0")" && pwd)/vsw_template.inp}"
# Shared-license-friendly defaults: prep features (LigPrep/MacroModel/Epik) have
# only ~60 tokens shared cluster-wide, so cap prep concurrency low (16); Glide has
# ~155 so 32 is fine. Override any with VSW_HOST (all stages) or the per-stage vars.
HOST="${VSW_HOST:-}"
HOST_GLIDE="${VSW_HOST_GLIDE:-${HOST:-singlecpu-long:32}}"
HOST_LIGPREP="${VSW_HOST_LIGPREP:-${HOST:-singlecpu-long:16}}"
HOST_MMOD="${VSW_HOST_MMOD:-${HOST:-singlecpu-long:16}}"
NJOBS="${VSW_NJOBS:-32}"
MAX_RETRIES="${VSW_MAX_RETRIES:-3}"

WORKDIR="$(cd "$(dirname "$BATCH")" && pwd)"
BATCH_ABS="$WORKDIR/$(basename "$BATCH")"
GRID_ABS="$(readlink -f "$GRID" 2>/dev/null || echo "$GRID")"
OUT_ABS="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
JOB="vsw"

cd "$WORKDIR"
sed -e "s|@LIGANDS@|${BATCH_ABS}|g" -e "s|@GRID@|${GRID_ABS}|g" "$TEMPLATE" > "${JOB}.inp"

# -WAIT so this returns only when docking is complete (required by the AL loop).
"${SCHRODINGER}/vsw" "${JOB}.inp" -OVERWRITE \
    -host_ligprep "${HOST_LIGPREP}" -host_glide "${HOST_GLIDE}" \
    -host_mmod "${HOST_MMOD}" \
    -adjust -NJOBS "${NJOBS}" -max_retries "${MAX_RETRIES}" -TMPLAUNCHDIR -WAIT

# Locate the SP pose output (pose-viewer .maegz).
POSES="$(ls ${JOB}*SP_OUT_1*pv.maegz 2>/dev/null | head -1 || true)"
[ -z "$POSES" ] && POSES="$(ls ${JOB}*SP_OUT_1*.maegz* 2>/dev/null | head -1 || true)"
[ -z "$POSES" ] && { echo "ERROR: no SP_OUT_1 output found in $WORKDIR" >&2; exit 1; }

# Best GlideScore per compound -> id,score
"${SCHRODINGER}/run" python3 - "$POSES" "$OUT_ABS" <<'PY'
import sys
from schrodinger.structure import StructureReader
poses, out = sys.argv[1], sys.argv[2]
best = {}
for st in StructureReader(poses):
    p = st.property
    cid = p.get("s_vsw_compound_code") or st.title
    sc = p.get("r_i_docking_score")
    if cid is None or sc is None:
        continue
    if cid not in best or sc < best[cid]:   # more negative = better
        best[cid] = sc
with open(out, "w") as fh:
    fh.write("id,score\n")
    for cid, sc in best.items():
        fh.write("%s,%s\n" % (cid, sc))
print("parsed %d compounds -> %s" % (len(best), out))
PY
