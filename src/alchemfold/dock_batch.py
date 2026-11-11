"""``alchemfold dock-batch`` — dock a ligand list with DiffDock-L and write one scores CSV.

The backend of the active-learning **DiffDock gate**: give it a ``.smi`` batch (``SMILES<TAB>id``,
as the active-learning loop writes) and a receptor PDB; it runs ONE batched DiffDock call (model
loads once) and writes ``--out`` as ``id,smiles,confidence`` (ligands without a pose are omitted).
Poses (rank 1 per ligand) are kept in ``<workdir>/poses/`` so a later rescoring step (Vina/GNINA)
can use them.
"""

from __future__ import annotations

import csv
from pathlib import Path


def run(args) -> int:
    import yaml

    from alchemfold.dock_run import run_from_config

    ligands = Path(args.ligands).expanduser().resolve()
    receptor = Path(args.receptor).expanduser().resolve()
    # Job folder: everything (work files, poses, results and, by default, the scores CSV) is
    # written to <workdir>/<jobname>/ (--workdir overrides the folder, e.g. the AL loop's round dir).
    workdir = (Path(args.workdir).expanduser() if args.workdir
               else Path(args.jobname or "dock_batch")).resolve()
    out = Path(args.out).expanduser().resolve() if args.out else workdir / "gate_scores.csv"
    if not ligands.is_file():
        raise FileNotFoundError(f"ligands file not found: {ligands}")
    if not receptor.is_file():
        raise FileNotFoundError(f"receptor PDB not found: {receptor}")
    workdir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "engine": "diffdock",
        "workdir": str(workdir),
        "receptor_pdb": str(receptor),
        "ligand_file": str(ligands),
        "ligand_file_format": "auto",
        "diffdock_samples": int(args.samples),
        "diffdock_inference_steps": int(args.steps),
        "poses_dir": str(workdir / "poses"),
        "results_csv": str(workdir / "results.csv"),
        "keep_raw": bool(args.keep_raw),
        "diffdock_trajectory": bool(args.trajectory),
        **({"diffdock_batch_size": int(args.batch_size)} if args.batch_size else {}),
        "trajectories_dir": str(workdir / "trajectories"),
        "proteins": [{"id": "A", "sequence": ""}],   # unused by DiffDock (receptor_pdb is used)
    }
    if args.diffdock_python:
        cfg["diffdock_python"] = args.diffdock_python
    if args.diffdock_dir:
        cfg["diffdock_dir"] = args.diffdock_dir
    config = workdir / "dock_batch.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rc = run_from_config(config)

    results = workdir / "results.csv"
    n_ok = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "smiles", "confidence"])
        if results.is_file():
            with results.open(newline="", encoding="utf-8") as rf:
                for row in csv.DictReader(rf):
                    if row.get("status") == "ok" and row.get("confidence") not in (None, ""):
                        w.writerow([row["ligand_id"], row.get("smiles", ""), row["confidence"]])
                        n_ok += 1
    print(f"dock-batch: {n_ok} ligand(s) with a DiffDock pose -> {out}")
    # A run that produced no scores at all is a failure (the loop must not silently continue).
    return rc if rc != 0 else (0 if n_ok else 1)


def add_parser(commands) -> None:
    p = commands.add_parser(
        "dock-batch",
        help="Dock a ligand list with DiffDock-L (one batched call) and write id,confidence "
             "(the active-learning gate backend)",
    )
    p.add_argument("--ligands", required=True, help="Ligand file (.smi 'SMILES<TAB>id', .csv, .cxsmiles)")
    p.add_argument("--receptor", required=True, help="Receptor structure (PDB)")
    p.add_argument("--out", default=None,
                   help="Output CSV id,smiles,confidence (default: <job folder>/gate_scores.csv)")
    p.add_argument("--jobname", default="dock_batch",
                   help="Job folder name: everything is written to ./<jobname>/ (default dock_batch)")
    p.add_argument("--workdir", default=None,
                   help="Use this folder as the job folder instead of ./<jobname>/")
    p.add_argument("--samples", type=int, default=10, help="Poses per ligand (default 10)")
    p.add_argument("--steps", type=int, default=20, help="DiffDock inference steps (default 20)")
    p.add_argument("--trajectory", action="store_true",
                   help="Also save each ligand's best-pose diffusion trajectory (multi-frame PDB) "
                        "to <job folder>/trajectories/")
    p.add_argument("--batch-size", type=int, default=None,
                   help="GPU batch over ONE ligand's poses (DiffDock default 10); only helps if --samples > batch")
    p.add_argument("--keep-raw", action="store_true", help="Keep DiffDock's raw output folders")
    p.add_argument("--diffdock-python", default=None, help="Interpreter of the diffdock env (auto-detected)")
    p.add_argument("--diffdock-dir", default=None, help="The DiffDock repo folder (auto-detected)")
    p.set_defaults(func=lambda a: run(a))
