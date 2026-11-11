"""Active-learning screening loop (orchestrator).

Deep-Docking-style loop for ultra-large libraries: a cheap surrogate (Chemprop)
scores the whole sharded library each round; Glide (the oracle) only docks the
batch that acquisition selects. The oracle / train / predict steps are **pluggable
command templates** in the config, so the Glide wrapper and the ChemFlow
train/predict CLI slot in without hard-coding either.

Optional **DiffDock pose gate** between "select the batch" and "run the oracle(s)":
the (cheap) DiffDock-L docking scores every selected compound; only the compounds with a
high pose confidence (plus a small random *audit* sample of the rest) go to the expensive
oracle(s). The DiffDock confidence is also recorded for ALL compounds as an extra column
(`diffdock_confidence`) in labels.csv — a multitask training target whose oracle columns are
simply empty for compounds that were gated out.

Only stdlib + PyYAML are imported here; the heavy work happens in the subprocess
commands you configure. Use ``--dry-run`` to print the plan without executing.
"""

from __future__ import annotations

import csv
import random
import shlex
import subprocess
from pathlib import Path

from chemflow.active_learning.acquisition import Candidate, select_batch, top_hits

_DEFAULTS = {
    "workdir": ".",                     # run from the current directory …
    "job_name": "al_run",              # … everything is written into <workdir>/<job_name>/
    "library": "custom",                # enamine_real | zinc20 | custom (see _LIBRARY_PRESETS)
    "library_shards": "shards/*.smi",   # id<TAB>smiles per line (gz ok via your cmds)
    "rounds": 5,
    "seed_size": 2000,
    "batch_size": 2000,
    "strategy": "ucb",                  # greedy | ucb | thompson | random
    "kappa": 1.0,
    "lower_is_better": True,            # docking score: more negative = better
    "top_n": 500,                       # how many hits to emit at the end
    "seed": 0,
    # Library file columns. Enamine REAL .cxsmiles = SMILES in col 0, ID in col 1
    # (whitespace/tab separated). A header line (first field 'smiles') is skipped.
    "smiles_column": 0,
    "id_column": 1,
    # --- Seed (round-0) selection ---
    "seed_method": "diverse",           # random | diverse (MaxMin) | scaffold (Butina)
    "seed_pool": 0,                     # oversample the diverse/scaffold pick draws from (0 = ~20x seed_size)
    "seed_actives": "",                 # optional SMILES file of known binders to force into the seed
    # --- Oracle (docking) ---
    "oracle": "glide",                  # glide | vina | custom
    "receptor": "",                     # Glide grid .zip, or Vina receptor .pdbqt
    "modules": [],                      # `module load` these before the oracle (HPC lmod)
    # Command templates. Placeholders in {braces} are filled each round:
    #   oracle:  {batch} .smi, {receptor}, {out} csv the wrapper must write (id,score)
    #   train:   {labels} cumulative id,score csv, {model} output model dir
    #   predict: {model}, {library} shard glob, {out} csv (id,pred[,sigma])
    "oracle_command": "",               # empty → use the built-in template for `oracle`
    "train_command": "chemflow train chemprop {config}",
    "train_config": "",                 # chemprop config.toml (positional) -> {config}; point its dataset at labels.csv
    "predict_command": "chemflow predict --model {model} --library {library} --out {out} --uncertainty",
    # --- Optional DiffDock-L pose GATE (between batch selection and the oracle) ---
    # Compounds with a LOW DiffDock confidence are not sent to the expensive oracle (they are
    # "not evaluated", NOT confirmed non-binders). A random `gate_audit_fraction` of them is sent
    # anyway so the gate's recall can be measured; the DiffDock confidence is stored for all.
    "gate": "",                         # "" = off | diffdock
    "gate_receptor": "",                # receptor PDB DiffDock docks into
    "gate_command": ("chemflow alchemfold dock-batch --ligands {batch} --receptor {receptor} "
                     "--out {out} --workdir {workdir} --samples {samples} --steps {steps}"),
    "gate_modules": [],                 # `module load` these before the gate command
    "gate_samples": 10,                 # DiffDock poses per compound
    "gate_steps": 20,                   # DiffDock denoising (reverse-diffusion) steps
    "gate_pass_fraction": 0.3,          # keep the top fraction of each batch by confidence ...
    "gate_threshold": None,             # ... or an absolute confidence cutoff (overrides the fraction)
    "gate_audit_fraction": 0.03,        # random share of the REJECTED sent to the oracle anyway
    "gate_seed": True,                  # also gate the round-0 seed batch
}

GATE_METRIC = "diffdock_confidence"     # labels.csv column holding the DiffDock confidence

# Built-in oracle wrapper contracts (you provide the wrapper scripts): each reads
# the batch .smi (SMILES<TAB>id), runs **ligand preparation then docking**, and
# writes {out} as CSV 'id,score'. Enamine REAL is raw SMILES, so prep is required.
#   - glide : LigPrep -> Glide SP/XP
#   - vina  : Meeko/OpenBabel -> pdbqt -> AutoDock Vina
#   - vsw   : Schrodinger Virtual Screening Workflow (LigPrep + HTVS->SP->XP in one)
_BUILTIN_ORACLE = {
    "glide": "glide_dock.sh {batch} {receptor} {out}",
    "vina": "vina_dock.sh {batch} {receptor} {out}",
    "vsw": "vsw_dock.sh {batch} {receptor} {out}",
    # ML co-folding affinity (AlchemFold / Boltz-2): {receptor} is the protein
    # sequence/FASTA; the wrapper co-folds each ligand and writes id,score (pIC50).
    "boltz2": "boltz2_screen.sh {batch} {receptor} {out}",
}

# Library presets: pick a vendor to get its default shard glob + column layout, or
# "custom" to use the explicit library_shards / smiles_column / id_column you set.
# Paths are the on-disk layout under screen_db/ (Enamine REAL ships .cxsmiles.bz2).
#   - enamine_real : screen_db/enamine_real/*.cxsmiles.bz2  (SMILES col 0, id col 1)
#   - zinc20       : screen_db/zinc20/*.smi                  (SMILES col 0, zinc_id col 1)
_LIBRARY_PRESETS = {
    "enamine_real": {"library_shards": "screen_db/enamine_real/*.cxsmiles.bz2",
                     "smiles_column": 0, "id_column": 1},
    "zinc20":       {"library_shards": "screen_db/zinc20/*.smi",
                     "smiles_column": 0, "id_column": 1},
    "custom":       {},
}


def load_config(path) -> dict:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    conf = dict(_DEFAULTS)
    # Apply the library preset's defaults first, so explicit user keys still win.
    lib = str(data.get("library", conf["library"])).lower()
    preset = _LIBRARY_PRESETS.get(lib)
    if preset is None:
        raise ValueError(
            f"library must be one of {sorted(_LIBRARY_PRESETS)}; got {lib!r}")
    conf.update(preset)
    conf.update({k: v for k, v in data.items() if v is not None})
    conf["library"] = lib
    return conf


def _chemflow_root() -> Path | None:
    """Repo/install root that contains `screen_db/` (walk up from the package)."""
    try:
        import chemflow
        p = Path(chemflow.__file__).resolve()
    except Exception:
        return None
    for parent in p.parents:
        if (parent / "screen_db").is_dir():
            return parent
    return None


def _shard_paths(pattern: str, base: Path) -> list[Path]:
    p = Path(pattern)
    if p.is_absolute():
        return sorted(Path(p.parent).glob(p.name))
    hits = sorted(base.glob(pattern))
    if hits:
        return hits
    # Fallback: resolve a relative `screen_db/...` glob against the ChemFlow install
    # root, so `active-learn run/seed` works regardless of the current directory.
    root = _chemflow_root()
    if root is not None:
        hits = sorted(root.glob(pattern))
    return hits


def _parse_record(line: str, smiles_col: int, id_col: int):
    """Parse one library line into (id, smiles), or None for header/blank lines."""
    line = line.strip()
    if not line:
        return None
    fields = line.split("\t") if "\t" in line else line.split()
    if max(smiles_col, id_col) >= len(fields):
        return None
    smiles = fields[smiles_col].strip()
    cid = fields[id_col].strip()
    if not smiles or not cid or smiles.lower() == "smiles":  # skip header
        return None
    return cid, smiles


def _random_offset_sample(shards, n: int, smiles_col: int, id_col: int, rng: random.Random):
    """Fast ~uniform sample of ``n`` (id, smiles) records from huge shards.

    Seeks to random byte offsets and reads the NEXT complete line, so it does not
    read the whole (10^9+) library — ideal for seeding. Slight length bias is
    irrelevant for a bootstrap seed. Falls back across shards sized by bytes.
    """
    import os

    sizes = [(s, os.path.getsize(s)) for s in shards]
    total = sum(sz for _, sz in sizes)
    if total == 0:
        return []
    out: dict[str, str] = {}
    attempts = 0
    max_attempts = n * 20
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        r = rng.randint(0, total - 1)
        # pick the shard this global offset lands in
        shard, off = sizes[0][0], r
        for s, sz in sizes:
            if off < sz:
                shard, off = s, off
                break
            off -= sz
        with open(shard, "rb") as fh:
            fh.seek(off)
            fh.readline()                 # discard partial line
            raw = fh.readline()           # next full line
        if not raw:
            continue
        rec = _parse_record(raw.decode("utf-8", "replace"), smiles_col, id_col)
        if rec:
            out[rec[0]] = rec[1]          # dedupe by id
    return list(out.items())


def _fetch_smiles(shards, wanted: set[str], smiles_col: int, id_col: int) -> dict[str, str]:
    """One-pass scan to recover SMILES for a set of ids (fallback; slow on huge libs)."""
    from chemflow.active_learning.seed import _open_text

    found: dict[str, str] = {}
    for shard in shards:
        with _open_text(shard) as fh:
            for line in fh:
                rec = _parse_record(line, smiles_col, id_col)
                if rec and rec[0] in wanted and rec[0] not in found:
                    found[rec[0]] = rec[1]
                    if len(found) == len(wanted):
                        return found
    return found


def _write_batch(path: Path, records) -> None:
    """Write a dock-ready .smi: 'SMILES<TAB>id' per line."""
    path.write_text("".join(f"{smi}\t{cid}\n" for cid, smi in records), encoding="utf-8")


def _read_preds(path: Path):
    """Return (candidates, {id: smiles}) from a surrogate preds.csv.

    The predict step reads the library anyway, so if it carries a 'smiles' column
    we reuse it to build the next batch without re-scanning the whole library.
    """
    cands: list[Candidate] = []
    smiles: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row.get("id") or row.get("ID") or row.get("name")
            if cid is None:
                continue
            pred = float(row.get("pred") or row.get("prediction") or row.get("score"))
            sigma = float(row.get("sigma") or row.get("std") or row.get("uncertainty") or 0.0)
            cands.append(Candidate(cid=str(cid), pred=pred, sigma=sigma))
            smi = row.get("smiles") or row.get("SMILES")
            if smi:
                smiles[str(cid)] = smi
    return cands, smiles


def _read_batch(path: Path) -> list[tuple[str, str]]:
    """Read a batch.smi ('SMILES<TAB>id' per line) back as (id, smiles) records."""
    records: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t") if "\t" in line else line.split()
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                records.append((parts[1].strip(), parts[0].strip()))
    return records


def _read_scores(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row.get("id") or row.get("ID") or row.get("name")
            sc = row.get("score") or row.get("glide_score") or row.get("r_i_docking_score")
            if cid is not None and sc not in (None, ""):
                out[str(cid)] = float(sc)
    return out


def _parse_oracles(conf: dict) -> list[dict]:
    """Normalize the config's oracle(s) into a list of spec dicts.

    Accepts structured ``oracles: [{name, receptor, metric, lower_is_better,
    modules, command}]``, a plain ``oracles: [name, ...]``, or the single-oracle
    top-level fields (oracle/receptor/modules/oracle_command/lower_is_better).
    """
    raw = conf.get("oracles")
    specs: list[dict] = []
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        for o in raw:
            specs.append({
                "name": str(o.get("name", "")).lower(),
                "receptor": o.get("receptor", conf.get("receptor", "")),
                "metric": o.get("metric") or "score",
                "lower_is_better": bool(o.get("lower_is_better", conf.get("lower_is_better", True))),
                "modules": list(o.get("modules") or []),
                "command": (o.get("command") or "").strip(),
            })
    elif isinstance(raw, list) and raw:
        for name in raw:
            specs.append({"name": str(name).lower(), "receptor": conf.get("receptor", ""),
                          "metric": "score", "lower_is_better": bool(conf.get("lower_is_better", True)),
                          "modules": list(conf.get("modules") or []), "command": ""})
    if not specs:
        specs.append({"name": str(conf.get("oracle", "glide")).lower(),
                      "receptor": conf.get("receptor", ""), "metric": "score",
                      "lower_is_better": bool(conf.get("lower_is_better", True)),
                      "modules": list(conf.get("modules") or []),
                      "command": (conf.get("oracle_command") or "").strip()})
    return specs


def _oracle_command_for(spec: dict) -> str:
    tmpl = (spec.get("command") or "").strip()
    if tmpl:
        return tmpl
    name = spec["name"]
    if name not in _BUILTIN_ORACLE:
        raise ValueError(
            f"oracle must be one of {sorted(_BUILTIN_ORACLE)} (or set a command); got {name!r}")
    return _BUILTIN_ORACLE[name]


def _write_labels_multi(path: Path, rows: dict[str, dict], metrics: list[str]) -> None:
    """Write a multi-metric labels table: id + one column per oracle metric."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id"] + metrics)
        for cid in sorted(rows):
            r = rows[cid]
            w.writerow([cid] + [r.get(m, "") for m in metrics])


def _read_labels_multi(path: Path, metrics: list[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row.get("id") or row.get("ID") or row.get("name")
            if not cid:
                continue
            d: dict[str, float] = {}
            for m in metrics:
                v = row.get(m)
                if v not in (None, ""):
                    try:
                        d[m] = float(v)
                    except ValueError:
                        pass
            if not d and row.get("score") not in (None, "") and metrics:
                try:
                    d[metrics[0]] = float(row["score"])
                except ValueError:
                    pass
            if d:
                rows[str(cid)] = d
    return rows


def _run(cmd: str, *, dry_run: bool, modules: list | None = None) -> None:
    """Run a command. If ``modules`` is given, run it in a login shell after
    ``module load``-ing them (HPC lmod), since `module` is a shell function."""
    modules = modules or []
    if modules:
        full = " && ".join([f"module load {m}" for m in modules] + [cmd])
        print("$ " + full)
        if dry_run:
            return
        rc = subprocess.run(["bash", "-lc", full], check=False).returncode
    else:
        print("$ " + cmd)
        if dry_run:
            return
        rc = subprocess.run(shlex.split(cmd), check=False).returncode
    if rc != 0:
        raise RuntimeError(f"command failed (exit {rc}): {cmd}")


def _oracle_template(conf: dict) -> str:
    tmpl = (conf.get("oracle_command") or "").strip()
    if tmpl:
        return tmpl
    oracle = str(conf.get("oracle", "glide")).lower()
    if oracle not in _BUILTIN_ORACLE:
        raise ValueError(
            f"oracle must be one of {sorted(_BUILTIN_ORACLE)} (or set oracle_command); got {oracle!r}")
    return _BUILTIN_ORACLE[oracle]


def _read_gate_scores(path: Path) -> dict[str, float]:
    """``{id: confidence}`` from the gate command's CSV (rows without a pose are skipped)."""
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = row.get("id") or row.get("ID") or row.get("name")
            v = row.get("confidence") or row.get("score")
            if cid and v not in (None, ""):
                try:
                    out[str(cid)] = float(v)
                except ValueError:
                    pass
    return out


def apply_gate(records, conf_by_id: dict[str, float], *, threshold, pass_fraction: float,
               audit_fraction: float, rng: random.Random):
    """Split a batch by DiffDock confidence (higher = better pose).

    Returns ``(passed, audited, report)``. ``passed``/``audited`` are (id, smiles) record lists
    that go to the oracle; ``report`` has one ``(id, smiles, confidence, gate, audit)`` row per
    compound. Compounds without a DiffDock pose are rejected (confidence blank).
    """
    scored = [(cid, smi, conf_by_id[cid]) for cid, smi in records if cid in conf_by_id]
    unscored = [(cid, smi, None) for cid, smi in records if cid not in conf_by_id]
    if threshold is not None:
        keep = {cid for cid, _, c in scored if c >= float(threshold)}
    else:
        n_keep = int(round(max(0.0, min(1.0, float(pass_fraction))) * len(records)))
        ranked = sorted(scored, key=lambda t: t[2], reverse=True)
        keep = {cid for cid, _, _ in ranked[:n_keep]}
    rejected = [(cid, smi, c) for cid, smi, c in scored + unscored if cid not in keep]
    n_audit = int(round(max(0.0, min(1.0, float(audit_fraction))) * len(rejected)))
    audit_ids = {cid for cid, _, _ in rng.sample(rejected, n_audit)} if n_audit else set()
    passed, audited, report = [], [], []
    for cid, smi, c in scored + unscored:
        if cid in keep:
            passed.append((cid, smi)); report.append((cid, smi, c, 1, 0))
        elif cid in audit_ids:
            audited.append((cid, smi)); report.append((cid, smi, c, 0, 1))
        else:
            report.append((cid, smi, c, 0, 0))
    return passed, audited, report


def _spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation (stdlib); None if it is undefined."""
    n = len(x)
    if n < 5:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else None


def _gate_quality(labels_rows, report, primary_metric: str, lower_is_better: bool) -> None:
    """Print how well the gate agrees with the oracle (needs oracle scores for passed + audited)."""
    both = [(r[2], labels_rows.get(r[0], {}).get(primary_metric)) for r in report
            if r[2] is not None and primary_metric in labels_rows.get(r[0], {})]
    audited = [labels_rows[r[0]][primary_metric] for r in report
               if r[4] and primary_metric in labels_rows.get(r[0], {})]
    passed = [labels_rows[r[0]][primary_metric] for r in report
              if r[3] and primary_metric in labels_rows.get(r[0], {})]
    if len(both) >= 5:
        rho = _spearman([c for c, _ in both], [v for _, v in both])
        # confidence is higher=better; the oracle score is better when LOWER (docking) -> expect rho < 0
        good = (-rho if lower_is_better else rho) if rho is not None else None
        if good is not None:
            print(f"  gate check: Spearman(confidence, {primary_metric}) = {rho:+.2f} over {len(both)} "
                  f"compounds ({'agrees' if good > 0 else 'DISAGREES'} with the oracle; "
                  f"|rho| near 0 = the gate is not informative)")
    if passed and audited:
        mp, ma = sum(passed) / len(passed), sum(audited) / len(audited)
        print(f"  gate audit: mean {primary_metric} passed={mp:.2f} (n={len(passed)}) vs "
              f"rejected-audit={ma:.2f} (n={len(audited)}) — if these are close, the gate is not "
              f"separating good from bad")


def _load_gate_seen(outdir: Path) -> set[str]:
    """Ids already evaluated by the gate in earlier rounds (never re-acquire them)."""
    seen: set[str] = set()
    for rep in sorted(outdir.glob("round_*/gate_report.csv")):
        with open(rep, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("id"):
                    seen.add(row["id"])
    return seen


def run_loop(config_path, *, dry_run: bool = False) -> int:
    conf = load_config(config_path)
    base = Path(config_path).expanduser().resolve().parent
    # Run from <workdir> (default: the current dir) and write everything into the
    # <workdir>/<job_name>/ output folder.
    workdir = (base / str(conf["workdir"])).resolve()
    job_name = str(conf.get("job_name") or "").strip()
    outdir = (workdir / job_name).resolve() if job_name else workdir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {outdir}")
    rng = random.Random(int(conf["seed"]))

    shards = _shard_paths(conf["library_shards"], base)
    if not shards and not dry_run:
        raise FileNotFoundError(f"no library shards matched {conf['library_shards']!r}")
    print(f"Library shards: {len(shards)}")

    # Oracle(s): each has its own input + output metric (a training target). The
    # PRIMARY metric drives acquisition; the rest are auxiliary multitask targets.
    oracle_specs = _parse_oracles(conf)
    metrics: list[str] = list(dict.fromkeys(o["metric"] for o in oracle_specs))
    primary_metric = str(conf.get("primary_metric") or metrics[0])
    if primary_metric not in metrics:
        primary_metric = metrics[0]
    primary_lower = next((o["lower_is_better"] for o in oracle_specs
                          if o["metric"] == primary_metric), bool(conf.get("lower_is_better", True)))
    print(f"Oracles: {', '.join(o['name']+'→'+o['metric'] for o in oracle_specs)} "
          f"(primary: {primary_metric}, lower_is_better={primary_lower})")

    gate_on = str(conf.get("gate") or "").strip().lower() == "diffdock"
    if gate_on:
        if not dry_run and not Path(str(conf.get("gate_receptor") or "")).expanduser().is_file():
            raise FileNotFoundError(
                f"gate: diffdock needs gate_receptor (a receptor PDB); got {conf.get('gate_receptor')!r}")
        print(f"DiffDock gate ON: pass top {conf['gate_pass_fraction']:.0%} per batch"
              if conf.get("gate_threshold") is None else
              f"DiffDock gate ON: pass confidence >= {conf['gate_threshold']}",
              f"+ {conf['gate_audit_fraction']:.0%} audit of the rejected")
    elif str(conf.get("gate") or "").strip():
        raise ValueError(f"gate must be '' or 'diffdock'; got {conf.get('gate')!r}")
    # The DiffDock confidence is one more labels column (an auxiliary multitask target).
    label_metrics = metrics + ([GATE_METRIC] if gate_on else [])
    gate_seen: set[str] = _load_gate_seen(outdir) if gate_on else set()

    labels_csv = outdir / "labels.csv"
    labels_rows: dict[str, dict] = _read_labels_multi(labels_csv, label_metrics) if labels_csv.exists() else {}
    scored: dict[str, float] = {cid: r[primary_metric] for cid, r in labels_rows.items()
                                if primary_metric in r}
    smiles_col = int(conf.get("smiles_column", 0))
    id_col = int(conf.get("id_column", 1))

    for rnd in range(int(conf["rounds"])):
        rdir = outdir / f"round_{rnd:02d}"
        rdir.mkdir(exist_ok=True)
        print(f"\n=== Round {rnd} ===")

        # 1. Select this round's batch as (id, smiles) records. RESUME: if this round's batch.smi
        #    already exists, reuse it — re-selecting would pick a different batch (the already-docked
        #    compounds are excluded now) and no longer match the saved oracle/gate scores.
        batch_smi = rdir / "batch.smi"   # 'SMILES<TAB>id' — oracle does ligand prep + dock
        resumed = (not dry_run) and batch_smi.exists() and batch_smi.stat().st_size > 0
        if resumed:
            records = _read_batch(batch_smi)
            print(f"Round {rnd} batch exists — reusing batch.smi ({len(records)} compounds)")
        elif rnd == 0 and not scored:
            if dry_run:
                records = [(f"seed_{i}", "CCO") for i in range(int(conf["seed_size"]))]
            else:
                from chemflow.active_learning.seed import select_seed
                records = select_seed(
                    shards, int(conf["seed_size"]),
                    method=str(conf.get("seed_method", "diverse")),
                    pool=int(conf.get("seed_pool", 0)),
                    smiles_col=smiles_col, id_col=id_col,
                    actives_file=(conf.get("seed_actives") or None),
                    seed=int(conf["seed"]),
                )
            print(f"Seed batch: {len(records)} compounds "
                  f"(method={conf.get('seed_method', 'diverse')})")
        else:
            preds_csv = outdir / f"round_{rnd-1:02d}" / "preds.csv"
            cands, pred_smiles = _read_preds(preds_csv) if preds_csv.exists() else ([], {})
            batch_ids = select_batch(
                cands, k=int(conf["batch_size"]), strategy=conf["strategy"],
                kappa=float(conf["kappa"]), lower_is_better=primary_lower,
                exclude=set(scored) | gate_seen, seed=int(conf["seed"]) + rnd,
            )
            # Build dock-ready records: prefer SMILES carried in preds.csv, else
            # scan the library once for the selected ids.
            missing = [cid for cid in batch_ids if cid not in pred_smiles]
            if missing and not dry_run:
                print(f"  fetching SMILES for {len(missing)} ids from the library …")
                pred_smiles.update(_fetch_smiles(shards, set(missing), smiles_col, id_col))
            records = [(cid, pred_smiles.get(cid, "CCO" if dry_run else "")) for cid in batch_ids]
            records = [(cid, smi) for cid, smi in records if smi]
            print(f"Acquired batch: {len(records)} compounds ({conf['strategy']})")

        if not dry_run and not resumed:
            _write_batch(batch_smi, records)

        # 1b. Optional DiffDock pose gate: dock the whole batch with DiffDock-L (cheap), keep the
        #     high-confidence compounds (+ a random audit sample of the rest) for the oracle(s).
        oracle_batch = batch_smi
        if gate_on and (rnd > 0 or bool(conf.get("gate_seed", True))):
            gate_csv = rdir / "gate_scores.csv"
            if not dry_run and gate_csv.exists() and gate_csv.stat().st_size > 0:
                print(f"  round {rnd} gate already scored — reusing {gate_csv.name}")
            else:
                _run(str(conf["gate_command"]).format(
                    batch=batch_smi, receptor=str(Path(str(conf["gate_receptor"])).expanduser()),
                    out=gate_csv, workdir=rdir / "gate", samples=int(conf["gate_samples"]),
                    steps=int(conf["gate_steps"])),
                    dry_run=dry_run, modules=conf.get("gate_modules") or [])
            if not dry_run:
                conf_by_id = _read_gate_scores(gate_csv)
                passed, audited, report = apply_gate(
                    records, conf_by_id,
                    threshold=conf.get("gate_threshold"),
                    pass_fraction=float(conf["gate_pass_fraction"]),
                    audit_fraction=float(conf["gate_audit_fraction"]),
                    rng=random.Random(int(conf["seed"]) + 1000 + rnd))
                with open(rdir / "gate_report.csv", "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["id", "smiles", "confidence", "gate_pass", "audit"])
                    w.writerows(report)
                for cid, c in conf_by_id.items():
                    labels_rows.setdefault(cid, {})[GATE_METRIC] = c
                gate_seen.update(r[0] for r in report)
                oracle_records = passed + audited
                oracle_batch = rdir / "batch_oracle.smi"
                _write_batch(oracle_batch, oracle_records)
                print(f"Gate: {len(conf_by_id)}/{len(records)} compounds got a DiffDock pose; "
                      f"{len(passed)} passed + {len(audited)} audit -> oracle "
                      f"({len(oracle_records)}/{len(records)} = "
                      f"{len(oracle_records) / max(1, len(records)):.0%} of the batch)")
            else:
                print("  [dry-run] gate would split the batch -> batch_oracle.smi (passed + audit)")
                report = []
        else:
            report = []

        # 2. Oracle(s) on the batch. Each selected oracle docks/scores the SAME batch
        #    with its own input + command and writes its metric column; all metrics are
        #    merged into labels.csv (the multitask training targets). Resumable per
        #    oracle: a finished oracle_scores_<name>.csv is reused across <=10-day jobs.
        single = len(oracle_specs) == 1
        for spec in oracle_specs:
            out_name = "oracle_scores.csv" if single else f"oracle_scores_{spec['name']}.csv"
            scores_csv = rdir / out_name
            if not dry_run and scores_csv.exists() and scores_csv.stat().st_size > 0:
                print(f"  round {rnd} {spec['name']} already scored — reusing {scores_csv.name}")
            else:
                cmd = _oracle_command_for(spec).format(
                    batch=oracle_batch, receptor=spec.get("receptor", ""), out=scores_csv)
                _run(cmd, dry_run=dry_run, modules=spec.get("modules") or [])
            if not dry_run and scores_csv.exists():
                for cid, val in _read_scores(scores_csv).items():
                    labels_rows.setdefault(cid, {})[spec["metric"]] = val
        if not dry_run:
            _write_labels_multi(labels_csv, labels_rows, label_metrics)
            scored = {cid: r[primary_metric] for cid, r in labels_rows.items()
                      if primary_metric in r}
            if report:
                _gate_quality(labels_rows, report, primary_metric, primary_lower)
        print(f"Total scored (primary metric {primary_metric}): {len(scored)}")

        # 3. Train surrogate on all labels; 4. predict over the whole library.
        #    Resumable: skip if this round's predictions already exist.
        model_dir = rdir / "model"
        preds_csv = rdir / "preds.csv"
        if not dry_run and preds_csv.exists() and preds_csv.stat().st_size > 0:
            print(f"  round {rnd} predictions exist — skipping train+predict")
        else:
            _run(conf["train_command"].format(
                labels=labels_csv, model=model_dir, config=conf.get("train_config", "")),
                dry_run=dry_run)
            _run(conf["predict_command"].format(
                model=model_dir, library=conf["library_shards"], out=preds_csv), dry_run=dry_run)

    # Final: emit top hits by oracle score.
    hits = top_hits(scored, n=int(conf["top_n"]), lower_is_better=primary_lower)
    hits_csv = outdir / "top_hits.csv"
    if not dry_run:
        with open(hits_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "score"])
            w.writerows(hits)
    print(f"\nDone. {len(scored)} compounds docked; top {len(hits)} -> {hits_csv}")
    print("Escalate the top hits to Boltz-2 co-fold (alchemfold) then FEP (alchemforge).")
    return 0


def _write_labels(path: Path, scored: dict[str, float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "score"])
        w.writerows(sorted(scored.items()))
