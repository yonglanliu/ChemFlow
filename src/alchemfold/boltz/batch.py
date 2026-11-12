"""Batched Boltz library screening: one ``boltz predict`` call per shard.

Instead of one process + one folder per ligand, all ligand YAMLs are written
flat into one inputs folder and Boltz is launched ONCE (model loaded once; the
protein MSA is computed once and reused). Outputs are consolidated into:

* ``poses_dir/<ligand_id>.pdb|cif``  - every predicted pose in one folder
* ``results_csv``                    - one row per ligand (scores + pose path +
                                       status ok/failed), so failures are visible

Re-running resumes: ligands already ``ok`` in ``results_csv`` are skipped.
"""

from __future__ import annotations

import csv
import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from alchemfold.boltz.run import build_boltz_input, build_command
from alchemfold.boltz.engine import _pocket_constraint

SCORE_COLUMNS = [
    "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
    "complex_plddt", "complex_iplddt",
    "affinity_pred_value", "affinity_probability_binary",
]
FIELDS = ["ligand_id", "smiles", "status", *SCORE_COLUMNS, "pose"]


def _write_yaml(conf, lig: dict, inputs_dir: Path, chain_id: str) -> Path:
    import yaml

    conf_i = _clone_for_ligand(conf, lig, chain_id)
    data = build_boltz_input(conf_i)
    constraint = _pocket_constraint(conf_i)
    if constraint is not None:
        data["constraints"] = [constraint]
    path = inputs_dir / f"{lig['id']}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _clone_for_ligand(conf, lig: dict, chain_id: str):
    import copy
    from types import SimpleNamespace

    c = copy.deepcopy(conf)
    c.ligands = [SimpleNamespace(id=chain_id, smiles=lig["smiles"])]
    if getattr(c, "pocket", None):
        c.pocket.binder = chain_id
    if getattr(c, "predict_affinity", False):
        c.affinity_binder = chain_id
    return c


def _share_msa(conf, first: dict, chain_id: str, workdir: Path) -> None:
    """Compute the protein MSA once (via the server) and reuse it for all ligands."""
    proteins = getattr(conf, "proteins", []) or []
    if not getattr(conf, "use_msa_server", True):
        return
    if all(getattr(p, "msa", None) for p in proteins):
        return
    seed_in = workdir / "_msa_seed" / "in"
    seed_out = workdir / "_msa_seed" / "out"
    seed_in.mkdir(parents=True, exist_ok=True)
    import copy

    seed_conf = copy.deepcopy(conf)
    seed_conf.devices = 1  # one ligand: no point spinning up DDP
    # This run exists only to build the MSA (its prediction is discarded), so make the
    # model part as cheap as possible: no affinity head, 1 diffusion step, 1 recycle.
    seed_conf.predict_affinity = False
    seed_conf.sampling_steps = 1
    seed_conf.recycling_steps = 1
    seed_conf.diffusion_samples = 1
    _write_yaml(seed_conf, first, seed_in, chain_id)  # no affinity property in the YAML
    cmd = _resolved(build_command(seed_conf, seed_in / f"{first['id']}.yaml", seed_out), 1,
                    seed_conf)
    print("Computing shared protein MSA once ...")
    subprocess.run(cmd, check=False)

    csvs = sorted(seed_out.glob("**/msa/*.csv"),
                  key=lambda p: int(re.search(r"_(\d+)\.csv$", p.name).group(1))
                  if re.search(r"_(\d+)\.csv$", p.name) else 0)
    seqs: list[str] = []
    for p in proteins:
        if str(p.sequence) not in seqs:
            seqs.append(str(p.sequence))
    if len(csvs) != len(seqs):
        print(f"  (could not locate shared MSA: found {len(csvs)} for {len(seqs)} "
              "unique sequence(s); falling back to the MSA server per ligand)")
        return
    msa_dir = workdir / "msa"
    msa_dir.mkdir(exist_ok=True)
    by_seq = {}
    for seq, src in zip(seqs, csvs):
        dest = msa_dir / f"protein_{len(by_seq)}.csv"
        shutil.copy(src, dest)
        by_seq[seq] = str(dest)
    for p in proteins:
        if not getattr(p, "msa", None):
            p.msa = by_seq[str(p.sequence)]
    conf.use_msa_server = False
    shutil.rmtree(workdir / "_msa_seed", ignore_errors=True)


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _collect(lig: dict, raw: Path, poses_dir: Path) -> dict:
    row = {"ligand_id": lig["id"], "smiles": lig["smiles"], "status": "failed",
           **{k: "" for k in SCORE_COLUMNS}, "pose": ""}
    pred_dirs = list(raw.glob(f"**/predictions/{lig['id']}"))
    if not pred_dirs:
        return row
    d = pred_dirs[0]
    poses = sorted(d.glob(f"{lig['id']}_model_0.*"))
    poses = [p for p in poses if p.suffix in (".pdb", ".cif")]
    if not poses:
        return row
    dest = poses_dir / f"{lig['id']}{poses[0].suffix}"
    shutil.copy(poses[0], dest)
    scores = {}
    scores.update(_read_json(d / f"confidence_{lig['id']}_model_0.json"))
    scores.update(_read_json(d / f"affinity_{lig['id']}.json"))
    for k in SCORE_COLUMNS:
        if isinstance(scores.get(k), (int, float)):
            row[k] = scores[k]
    row.update(status="ok", pose=str(dest))
    return row


def _load_done(results_csv: Path) -> dict[str, dict]:
    if not results_csv.is_file():
        return {}
    with results_csv.open(newline="", encoding="utf-8") as fh:
        return {r["ligand_id"]: r for r in csv.DictReader(fh) if r.get("status") == "ok"}


def _resolved(cmd, devices: int = 1, conf=None):
    from alchemfold._exec import prefer_envs_for, resolve_command, wrap_srun

    prefer = prefer_envs_for(conf) if conf is not None else ("alchemfold",)
    return wrap_srun(resolve_command(cmd, prefer), devices)


def run_boltz_library(conf, library: list[dict], *, dry_run: bool = False) -> int:
    return run_library(conf, library, dry_run=dry_run)


def run_library(conf, library: list[dict], *, dry_run: bool = False) -> int:
    from alchemfold.dock_run import _screen_chain_id

    workdir = Path(conf.workdir).expanduser().resolve()
    inputs_dir = workdir / "inputs"
    raw = workdir / "boltz_raw"
    poses_dir = Path(getattr(conf, "poses_dir", None) or workdir / "poses")
    results_csv = Path(getattr(conf, "results_csv", None) or workdir / "results.csv")
    for d in (inputs_dir, poses_dir, results_csv.parent):
        d.mkdir(parents=True, exist_ok=True)

    done = _load_done(results_csv)
    todo = [l for l in library if l["id"] not in done]
    print(f"Engine: boltz (batched); {len(library)} ligand(s), "
          f"{len(done)} already done, {len(todo)} to run")
    if not todo:
        return 0

    chain_id = _screen_chain_id(conf)
    if not dry_run:
        _share_msa(conf, todo[0], chain_id, workdir)
    for lig in todo:
        _write_yaml(conf, lig, inputs_dir, chain_id)

    cmd = _resolved(build_command(conf, inputs_dir, raw),
                    int(getattr(conf, "devices", 1) or 1), conf)
    if dry_run:
        print(" ".join(shlex.quote(c) for c in cmd))
        return 0
    n_res = sum(len(str(getattr(p, "sequence", "") or "")) for p in (getattr(conf, "proteins", []) or []))
    print(f"Protein: {n_res} residue(s) total; {len(todo)} ligand(s) in this run "
          f"(diffusion_samples={getattr(conf, 'diffusion_samples', 1)}, "
          f"sampling_steps={getattr(conf, 'sampling_steps', 200)}, "
          f"recycling_steps={getattr(conf, 'recycling_steps', 3)}, "
          f"affinity={bool(getattr(conf, 'predict_affinity', False))}, "
          f"devices={getattr(conf, 'devices', 1)})")
    print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, check=False).returncode
    elapsed = time.time() - t0
    n_gpu = max(1, int(getattr(conf, "devices", 1) or 1))
    print(f"Boltz finished in {elapsed / 60:.1f} min: "
          f"{elapsed / max(1, len(todo)):.1f} s per ligand (wall), "
          f"{elapsed * n_gpu / 3600:.2f} GPU-hours used")

    rows = list(done.values())
    n_ok = n_fail = 0
    # Collecting results is where each ligand is confirmed finished, so report
    # "N/total compounds" here — including the ones already done on an earlier
    # run, so the count reflects the whole library rather than just this batch.
    from chemflow.utils.progress import Progress

    bar = Progress(len(library), label="collecting", unit="compound")
    bar.update(len(done))  # resumed from results.csv
    for lig in todo:
        row = _collect(lig, raw, poses_dir)
        ok = row["status"] == "ok"
        n_ok += ok
        n_fail += not ok
        rows.append(row)
        bar.update(1, failed=0 if ok else 1, note=str(lig.get("id", "")))
    bar.close()
    with results_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Done: {n_ok} ok, {n_fail} failed, {len(done)} already complete "
          f"({len(rows)}/{len(library)} compounds total). "
          f"Results: {results_csv}; poses: {poses_dir}")

    if not getattr(conf, "keep_raw", False):
        shutil.rmtree(raw, ignore_errors=True)
        shutil.rmtree(inputs_dir, ignore_errors=True)
    return rc if rc != 0 else (1 if n_fail and not n_ok else 0)
