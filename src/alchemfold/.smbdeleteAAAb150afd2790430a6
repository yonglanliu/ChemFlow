"""AlchemFold virtual screening: shard a library, dock on a SLURM array, rank.

``stage`` splits the combined ligand library into shards, writes a per-shard
docking config, a SLURM **array** job (one task per shard, no concurrency cap),
and a dependent **ranking** job. ``rank`` scrapes a score from every predicted
complex, sorts them, and writes the top hits.

Run headlessly::

    python -m alchemfold.screen --config screen.yaml           # stage
    python -m alchemfold.screen --config screen.yaml --rank    # rank
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from alchemfold.dock_config import load_ligands_from_file
from alchemfold.screen_config import load_config, validate_config

# Score keys we know how to read, with the sort direction that means "better".
# True => higher is better (descending); False => lower is better (ascending).
_SCORE_DIRECTION = {
    "confidence": True,
    "confidence_score": True,
    "ptm": True,
    "iptm": True,
    "plddt": True,
    "aggregate_score": True,
    "affinity_probability_binary": True,
    "affinity_pred_value": False,  # predicted ΔG-like, lower = stronger binder
}

_AUTO_METRIC = {
    "boltz": ("affinity_pred_value", "confidence_score"),
    "chai": ("aggregate_score", "ptm", "confidence"),
    "diffdock": ("confidence",),
}


# --------------------------------------------------------------------------- shard


def combine_ligands(conf) -> list[dict]:
    """Merge all library files into one deduplicated [{id, smiles}] list."""
    merged: list[dict] = []
    seen: set[str] = set()
    for path in getattr(conf, "ligand_files", []) or []:
        for entry in load_ligands_from_file(path, getattr(conf, "ligand_file_format", "auto")):
            cid, base, k = entry["id"], entry["id"], 2
            while cid in seen:
                cid = f"{base}_{k}"
                k += 1
            seen.add(cid)
            merged.append({"id": cid, "smiles": entry["smiles"]})
    return merged


def split_shards(ligands: list[dict], n_shards: int, shard_size: int) -> list[list[dict]]:
    """Split into shards by count (n_shards) or by size (shard_size)."""
    total = len(ligands)
    if total == 0:
        return []
    if n_shards and n_shards > 0:
        n = min(n_shards, total)
        # Round-robin keeps shard sizes balanced.
        shards: list[list[dict]] = [[] for _ in range(n)]
        for i, lig in enumerate(ligands):
            shards[i % n].append(lig)
        return [s for s in shards if s]
    size = shard_size if shard_size and shard_size > 0 else total
    return [ligands[i:i + size] for i in range(0, total, size)]


def _shard_dock_yaml(conf, shard_smi: Path, shard_workdir: Path) -> dict:
    """Build a single-shard docking config (screened as a library)."""
    engine = str(conf.engine)
    cfg: dict = {
        "engine": engine,
        "workdir": str(shard_workdir),
        "out_dir": "docking_output",
        "output_format": getattr(conf, "output_format", "pdb"),
        "use_msa_server": bool(getattr(conf, "use_msa_server", True)),
        "seed": int(getattr(conf, "seed", 1)),
        "accelerator": str(getattr(conf, "accelerator", "gpu")),
        "devices": int(getattr(conf, "devices", 1)),
        "diffusion_samples": int(getattr(conf, "diffusion_samples", 1)),
        "recycling_steps": int(getattr(conf, "recycling_steps", 3)),
        "sampling_steps": int(getattr(conf, "sampling_steps", 200)),
        "predict_affinity": bool(getattr(conf, "predict_affinity", False)),
        "ligand_file": str(shard_smi),
        "ligand_file_format": "smiles",
        "proteins": [
            {k: v for k, v in (("id", getattr(p, "id", "A")),
                               ("sequence", getattr(p, "sequence", None)),
                               ("msa", getattr(p, "msa", None))) if v is not None}
            for p in (getattr(conf, "proteins", []) or [])
        ],
    }
    for key in ("receptor_pdb", "diffdock_command", "diffdock_inference_steps",
                "diffdock_model_dir", "diffdock_config", "boltz_command", "chai_command"):
        value = getattr(conf, key, None)
        if value:
            cfg[key] = value
    pocket = getattr(conf, "pocket", None)
    if pocket:
        cfg["pocket"] = {
            "max_distance": float(getattr(pocket, "max_distance", 6.0)),
            "contacts": [{"chain": c.chain, "residue": int(c.residue)}
                         for c in (getattr(pocket, "contacts", []) or [])],
        }
    return cfg


def _slurm_gres(slurm) -> str:
    g = int(getattr(slurm, "gpus_per_task", 1))
    if g <= 0:
        return ""
    gtype = str(getattr(slurm, "gpu_type", "") or "").strip()
    prefix = f"{gtype}:" if gtype else ""
    return f"#SBATCH --gres=gpu:{prefix}{g}\n"


def _conda_activate(slurm) -> str:
    env = str(getattr(slurm, "conda_env", "") or "").strip()
    if not env:
        return ""
    return (
        'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
        f"conda activate {env}\n"
    )


def stage(config_path) -> int:
    conf = load_config(config_path)
    validate_config(conf)
    config_path = Path(config_path).expanduser().resolve()

    workdir = Path(conf.workdir).expanduser().resolve()
    shard_dir = workdir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    ligands = combine_ligands(conf)
    if not ligands:
        raise ValueError("no ligands found in ligand_files")
    shards = split_shards(ligands, int(conf.n_shards), int(conf.shard_size))
    n = len(shards)
    width = max(4, len(str(n - 1)))

    for i, shard in enumerate(shards):
        tag = str(i).zfill(width)
        smi = shard_dir / f"shard_{tag}.smi"
        smi.write_text(
            "".join(f"{lig['smiles']} {lig['id']}\n" for lig in shard), encoding="utf-8"
        )
        shard_work = shard_dir / f"work_{tag}"
        import yaml

        (shard_dir / f"shard_{tag}.yaml").write_text(
            yaml.safe_dump(_shard_dock_yaml(conf, smi, shard_work), sort_keys=False),
            encoding="utf-8",
        )

    slurm = conf.slurm
    account = f"#SBATCH --account={slurm.account}\n" if str(getattr(slurm, "account", "")).strip() else ""
    python = sys.executable

    array_sh = workdir / "dock_array.sh"
    array_sh.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name=screen_dock\n"
        f"#SBATCH --partition={slurm.partition}\n"
        f"{account}"
        f"#SBATCH --array=0-{n - 1}\n"
        f"#SBATCH --nodes={int(slurm.nodes)}\n"
        f"#SBATCH --cpus-per-task={int(slurm.cpus_per_task)}\n"
        f"{_slurm_gres(slurm)}"
        f"#SBATCH --mem={slurm.mem}\n"
        f"#SBATCH --time={slurm.time}\n"
        "set -euo pipefail\n"
        "trap 'echo \"[FAILED] line $LINENO: $BASH_COMMAND (exit $?)\" >&2' ERR\n"
        f"{_conda_activate(slurm)}"
        f'SHARD=$(printf "%0{width}d" "$SLURM_ARRAY_TASK_ID")\n'
        f'exec {python} -m alchemfold.dock_run --config "{shard_dir}/shard_${{SHARD}}.yaml"\n',
        encoding="utf-8",
    )
    array_sh.chmod(array_sh.stat().st_mode | 0o100)

    rank_sh = workdir / "rank.sh"
    rank_sh.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=screen_rank\n"
        f"#SBATCH --partition={slurm.partition}\n"
        f"{account}"
        "#SBATCH --nodes=1\n"
        "#SBATCH --cpus-per-task=2\n"
        "#SBATCH --mem=8G\n"
        "#SBATCH --time=2:00:00\n"
        "set -euo pipefail\n"
        f"{_conda_activate(slurm)}"
        f'exec {python} -m alchemfold.screen --config "{config_path}" --rank\n',
        encoding="utf-8",
    )
    rank_sh.chmod(rank_sh.stat().st_mode | 0o100)

    submit_sh = workdir / "submit_screen.sh"
    submit_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'cd "{workdir}"\n'
        "if ! ARRAY_ID=$(sbatch --parsable dock_array.sh); then\n"
        '    echo "ERROR: docking array submission failed" >&2; exit 1\n'
        "fi\n"
        'echo "Docking array job: $ARRAY_ID"\n'
        "sbatch --dependency=afterok:${ARRAY_ID} rank.sh\n"
        'echo "Ranking job queued (runs after the array completes)."\n',
        encoding="utf-8",
    )
    submit_sh.chmod(submit_sh.stat().st_mode | 0o100)

    print(f"Staged {len(ligands)} ligand(s) into {n} shard(s) under {shard_dir}")
    print(f"Submit the screen (array + ranking) with:\n    bash {submit_sh}")
    return 0


# --------------------------------------------------------------------------- rank


def _parse_diffdock_confidence(ligand_dir: Path) -> tuple[float | None, Path | None]:
    best, best_path = None, None
    for sdf in ligand_dir.rglob("*.sdf"):
        m = re.search(r"confidence(-?\d+\.?\d*)", sdf.name)
        if m:
            value = float(m.group(1))
            if best is None or value > best:
                best, best_path = value, sdf
    return best, best_path


def _scrape_json_scores(ligand_dir: Path) -> dict[str, float]:
    found: dict[str, float] = {}
    for jf in ligand_dir.rglob("*.json"):
        try:
            data = json.loads(jf.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            k = str(key).lower()
            if k in _SCORE_DIRECTION and isinstance(value, (int, float)):
                # Keep the best value seen for each key.
                better = _SCORE_DIRECTION[k]
                if k not in found or (value > found[k]) == better:
                    found[k] = float(value)
    return found


def _pick_metric(conf, scores: dict[str, float], diffdock_conf: float | None):
    requested = str(getattr(conf.ranking, "metric", "auto"))
    if requested != "auto":
        if requested == "confidence" and diffdock_conf is not None and not scores:
            return "confidence", diffdock_conf
        return requested, scores.get(requested)
    # auto: try the engine's priority list, then diffdock filename confidence.
    for key in _AUTO_METRIC.get(str(conf.engine), ()):  # noqa: B007
        if key in scores:
            return key, scores[key]
    if diffdock_conf is not None:
        return "confidence", diffdock_conf
    # fall back to any known score
    for key, value in scores.items():
        return key, value
    return "none", None


def rank(config_path) -> int:
    conf = load_config(config_path)
    validate_config(conf)
    workdir = Path(conf.workdir).expanduser().resolve()
    shard_dir = workdir / "shards"

    rows: list[dict] = []
    for ligands_root in shard_dir.glob("work_*/ligands"):
        for ligand_dir in sorted(p for p in ligands_root.iterdir() if p.is_dir()):
            scores = _scrape_json_scores(ligand_dir)
            dd_conf, dd_path = _parse_diffdock_confidence(ligand_dir)
            metric, value = _pick_metric(conf, scores, dd_conf)
            if value is None:
                continue
            structure = None
            for pattern in ("*.pdb", "*.cif", "*.sdf"):
                hits = sorted(ligand_dir.rglob(pattern))
                if hits:
                    structure = dd_path if (dd_path and pattern == "*.sdf") else hits[0]
                    break
            rows.append({
                "ligand_id": ligand_dir.name,
                "shard": ligands_root.parent.name,
                "metric": metric,
                "score": value,
                "structure": str(structure) if structure else "",
            })

    if not rows:
        print("No scored predictions found to rank (check the docking outputs).",
              file=sys.stderr)
        return 1

    # Sort: order 'auto' uses the metric's natural direction.
    order = str(getattr(conf.ranking, "order", "auto"))
    metric_name = rows[0]["metric"]
    descending = _SCORE_DIRECTION.get(metric_name, True) if order == "auto" else (order == "desc")
    rows.sort(key=lambda r: r["score"], reverse=descending)

    top_k = int(conf.ranking.top_k)
    out_csv = workdir / str(getattr(conf.ranking, "output", "ranking.csv"))
    import csv as _csv

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.writer(handle)
        writer.writerow(["rank", "ligand_id", "shard", "metric", "score", "structure"])
        for i, r in enumerate(rows, start=1):
            writer.writerow([i, r["ligand_id"], r["shard"], r["metric"], r["score"], r["structure"]])

    top_dir = workdir / "top"
    top_dir.mkdir(exist_ok=True)
    for r in rows[:top_k]:
        if r["structure"] and Path(r["structure"]).is_file():
            dest = top_dir / f"{r['ligand_id']}{Path(r['structure']).suffix}"
            try:
                shutil.copy(r["structure"], dest)
            except OSError:
                pass

    print(f"Ranked {len(rows)} ligand(s) by '{metric_name}' "
          f"({'desc' if descending else 'asc'}).")
    print(f"Full ranking: {out_csv}")
    print(f"Top {min(top_k, len(rows))} structures copied to: {top_dir}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AlchemFold virtual screening (shard + SLURM array + rank).",
    )
    parser.add_argument("--config", required=True, help="Screening YAML configuration")
    parser.add_argument("--rank", action="store_true",
                        help="Rank completed results instead of staging")
    args = parser.parse_args(argv)
    return rank(args.config) if args.rank else stage(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
