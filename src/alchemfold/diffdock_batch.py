"""Batched DiffDock-L library screening: ONE ``inference`` call per shard.

Mirrors the Boltz batched path. All ligands of the shard go into one multi-row DiffDock CSV, the
model is loaded once, then outputs are consolidated into:

* ``poses_dir/<ligand_id>.sdf``  - the top-ranked (rank1) pose of every ligand, one folder
* ``results_csv``               - one row per ligand: smiles, status, ``confidence``, pose

``confidence`` is DiffDock's pose-quality score (higher is better) — it is NOT a binding
affinity. Re-running resumes: ligands already ``ok`` in ``results_csv`` are skipped.
"""

from __future__ import annotations

import csv
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from alchemfold.engines import diffdock as engine

FIELDS = ["ligand_id", "smiles", "status", "confidence", "pose", "trajectory"]


def _load_done(results_csv: Path) -> dict[str, dict]:
    if not results_csv.is_file():
        return {}
    with results_csv.open(newline="", encoding="utf-8") as fh:
        return {r["ligand_id"]: r for r in csv.DictReader(fh) if r.get("status") == "ok"}


def _collect(lig: dict, raw: Path, poses_dir: Path, traj_dir: Path | None = None) -> dict:
    row = {"ligand_id": lig["id"], "smiles": lig["smiles"], "status": "failed",
           "confidence": "", "pose": "", "trajectory": ""}
    d = raw / lig["id"]
    if not d.is_dir():
        return row
    ranked = sorted(d.glob("rank1_confidence*.sdf")) or sorted(d.glob("rank1.sdf"))
    if not ranked:
        return row
    src = ranked[0]
    m = re.search(r"confidence(-?\d+\.?\d*)", src.name)
    dest = poses_dir / f"{lig['id']}.sdf"
    shutil.copy(src, dest)
    row.update(status="ok", confidence=float(m.group(1)) if m else "", pose=str(dest))
    if traj_dir is not None:
        traj = d / "rank1_reverseprocess.pdb"      # multi-frame PDB of the best pose's denoising
        if traj.is_file():
            traj_dir.mkdir(parents=True, exist_ok=True)
            tdest = traj_dir / f"{lig['id']}.pdb"
            try:
                from alchemfold.diffdock_traj import clean_file

                clean_file(traj, tdest)       # chronological, no frozen hydrogens
            except Exception:
                shutil.copy(traj, tdest)      # fall back to the raw file
            row["trajectory"] = str(tdest)
    return row


def run_library(conf, library: list[dict], *, dry_run: bool = False) -> int:
    workdir = Path(conf.workdir).expanduser().resolve()
    raw = workdir / "diffdock_raw"
    poses_dir = Path(getattr(conf, "poses_dir", None) or workdir / "poses")
    results_csv = Path(getattr(conf, "results_csv", None) or workdir / "results.csv")
    save_traj = bool(getattr(conf, "diffdock_trajectory", False))
    traj_dir = Path(getattr(conf, "trajectories_dir", None) or workdir / "trajectories") if save_traj else None
    for d in (poses_dir, results_csv.parent):
        d.mkdir(parents=True, exist_ok=True)

    done = _load_done(results_csv)
    todo = [l for l in library if l["id"] not in done]
    print(f"Engine: diffdock (batched); {len(library)} ligand(s), "
          f"{len(done)} already done, {len(todo)} to run")
    if not todo:
        return 0
    receptor = getattr(conf, "receptor_pdb", None)
    if not receptor or not Path(receptor).is_file():
        raise ValueError(f"receptor_pdb not found: {receptor!r} (DiffDock needs a receptor structure)")

    csv_path = engine.write_library_csv(conf, todo, workdir / "diffdock_input.csv")
    cmd = engine.build_command(conf, {"protein_ligand_csv": csv_path}, raw)
    cwd = engine.build_cwd(conf)
    from alchemfold._exec import resolve_command

    cmd = resolve_command(cmd)
    warnings = engine.preflight(conf)
    for w in warnings:
        print(f"WARNING (first-run start-up): {w}", flush=True)
    if len(todo) > 2000:
        print(f"WARNING: {len(todo)} ligands in ONE DiffDock process on ONE GPU. A screen splits them "
              "into shards that run in parallel (`alchemfold screen ...`, e.g. 500 ligands per shard) "
              "and is resumable per shard — much faster and safer for big libraries.", flush=True)
    print(f"Receptor: {receptor}; {len(todo)} ligand(s); "
          f"samples/ligand={engine.samples_per_complex(conf)}, "
          f"inference_steps={getattr(conf, 'diffdock_inference_steps', 20)}")
    print(f"Running (in {cwd}): {' '.join(shlex.quote(c) for c in cmd)}")
    if dry_run:
        return 0

    t0 = time.time()
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in engine.build_env(conf).items()})
    rc = subprocess.run(cmd, cwd=cwd, check=False, env=env).returncode
    elapsed = time.time() - t0
    print(f"DiffDock finished in {elapsed / 60:.1f} min: "
          f"{elapsed / max(1, len(todo)):.1f} s per ligand (wall), {elapsed / 3600:.2f} GPU-hours used")
    if warnings or len(todo) < 20:
        print("  NOTE: this wall time includes DiffDock's start-up (model + ESM2 load"
              + (", and the one-time table build / downloads warned about above" if warnings else "")
              + "). The real per-ligand docking cost is the last 'it [..s/it]' line of DiffDock's log; "
                "it is only meaningful for a batch of many ligands.")

    rows = list(done.values())
    n_ok = n_fail = 0
    for lig in todo:
        row = _collect(lig, raw, poses_dir, traj_dir)
        n_ok += row["status"] == "ok"
        n_fail += row["status"] != "ok"
        rows.append(row)
    with results_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Done: {n_ok} ok, {n_fail} failed. Results: {results_csv}; poses: {poses_dir}")

    if not getattr(conf, "keep_raw", False):
        shutil.rmtree(raw, ignore_errors=True)
    return rc if rc != 0 else (1 if n_fail and not n_ok else 0)
