"""``alchemfold gpu-test`` — check GPUs and (optionally) run a tiny built-in screen.

Run it inside a GPU allocation (e.g. ``sinteractive --gres=gpu:2 --ntasks-per-node=2``)::

    chemflow alchemfold gpu-test                    # GPU visible? which model? speed
    chemflow alchemfold gpu-test --gpus 2           # + multi-GPU srun/NCCL launch probe
    chemflow alchemfold gpu-test --gpus 2 --boltz   # + tiny real Boltz screen (4 ligands)
    chemflow alchemfold gpu-test --boltz --affinity # also exercise affinity prediction
    chemflow alchemfold gpu-test --gpus 2 --boltz --submit   # test the real sbatch array
    chemflow alchemfold gpu-test --diffdock         # tiny DiffDock-L screen through AlchemFold
                                                    #   (op-level DiffDock checks: chemflow diffdock-env gpu-test)

The Boltz example uses a 35-residue protein (villin headpiece) with no MSA server and
4 small molecules, so it needs no internet and finishes in minutes (after the first
model-weights download). It goes through the SAME code path as a real screen: stage ->
shard -> batched ``boltz predict`` (srun when multi-GPU) -> results CSV -> ranking.
Everything is written to ``<workdir>/<jobname>/`` (default ``./gpu_test/``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EXAMPLE_SEQUENCE = "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"  # villin headpiece HP35
EXAMPLE_LIGANDS = [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
]


def _torch_check() -> tuple[bool, int]:
    try:
        import torch
    except Exception as exc:
        print(f"[FAIL] PyTorch import: {exc}")
        return False, 0
    if not torch.cuda.is_available():
        print("[FAIL] PyTorch CUDA: no GPU visible (are you in a GPU allocation? "
              "e.g. sinteractive --gres=gpu:1)")
        return False, 0
    count = torch.cuda.device_count()
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        torch.cuda.set_device(i)
        a = torch.randn(2048, 2048, device=f"cuda:{i}")
        for _ in range(2):
            a @ a
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            a @ a
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 5
        print(f"[PASS] cuda:{i} {props.name}, {props.total_memory / 2**30:.0f} GiB, "
              f"capability {props.major}.{props.minor}, 2048x2048 fp32 matmul {ms:.2f} ms")
    return True, count


def _ddp_probe(n: int) -> bool:
    from alchemfold._exec import wrap_srun

    if not os.environ.get("SLURM_JOB_ID"):
        print("[SKIP] multi-GPU probe: not inside a SLURM job (Lightning launches its own "
              "processes there, so srun is not needed)")
        return True
    base = [sys.executable, "-m", "alchemfold._ddp_probe"]
    cmd = wrap_srun(base, n)
    print("Launching:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print("[FAIL] multi-GPU probe timed out (ranks could not rendezvous)")
        return False
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("[rank")]
    for ln in sorted(lines):
        print("   ", ln)
    ok = proc.returncode == 0 and len(lines) == n and all(ln.endswith("OK") for ln in lines)
    devices = {(ln.split("host=")[1].split()[0], ln.split("device=")[1].split()[0]) for ln in lines}
    if ok and len(devices) < n:
        print(f"[WARN] {n} ranks but only {len(devices)} distinct (host, device) pair(s): ranks "
              "share a GPU. Add --gpus-per-task=1 to the srun call (tell me and I'll add it).")
        ok = False
    if not ok:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        if tail:
            print("    stderr tail:\n      " + "\n      ".join(tail))
        if "processors" in (proc.stderr or "") or "ntasks" in (proc.stderr or "").lower():
            print(f"    Hint: allocate one task per GPU, e.g. "
                  f"sinteractive --gres=gpu:{n} --ntasks-per-node={n}")
    print(f"[{'PASS' if ok else 'FAIL'}] multi-GPU launch (srun, {n} tasks, NCCL all-reduce)")
    return ok


def _boltz_example(args, n: int) -> bool:
    import yaml

    from alchemfold.screen_config import default_yaml_text

    job = (Path(args.workdir).expanduser() / args.jobname).resolve()
    job.mkdir(parents=True, exist_ok=True)
    smi = job / "ligands.smi"
    smi.write_text("".join(f"{s} {name}\n" for name, s in EXAMPLE_LIGANDS), encoding="utf-8")

    cfg = yaml.safe_load(default_yaml_text(
        engine="boltz", library="custom", sequence=EXAMPLE_SEQUENCE, ligand_files=[str(smi)],
        n_shards=1, shard_size=0, jobname=args.jobname, cleanup=False,
        predict_affinity=args.affinity,
        slurm={"gpus_per_task": n, "time": "00:30:00"}))
    cfg["use_msa_server"] = False
    cfg["proteins"] = [{"id": "A", "sequence": EXAMPLE_SEQUENCE, "msa": "empty"}]
    cfg["sampling_steps"] = 50
    cfg["recycling_steps"] = 1
    cfg["diffusion_samples"] = 1
    if args.boltz_command:
        cfg["boltz_command"] = args.boltz_command
    config = job / "screen.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Example job folder: {job}")

    from alchemfold.screen import rank, stage

    stage(config)
    if args.submit:
        rc = subprocess.run(["bash", str(job / "submit_screen.sh")]).returncode
        print(f"[{'PASS' if rc == 0 else 'FAIL'}] sbatch submit. Watch with `squeue -u $USER`; "
              f"when done: python -m alchemfold screen rank {config}")
        return rc == 0

    from alchemfold.dock_run import run_from_config

    t0 = __import__("time").time()
    rc = run_from_config(job / "shards" / "shard_0000.yaml")
    took = __import__("time").time() - t0
    rank(config)
    poses = sorted((job / "poses").glob("*.*")) if (job / "poses").is_dir() else []
    ok = rc == 0 and len(poses) == len(EXAMPLE_LIGANDS)
    print(f"{len(poses)}/{len(EXAMPLE_LIGANDS)} pose(s) in {job / 'poses'}; "
          f"results: {job / 'results'}; ranking: {job / 'ranking.csv'}; "
          f"total {took / 60:.1f} min")
    print(f"[{'PASS' if ok else 'FAIL'}] Boltz example screen ({n} GPU(s) per shard)")
    return ok


def _diffdock_example(args) -> bool:
    """Tiny DiffDock-L screen through the real stage -> run -> rank path (1 GPU)."""
    import yaml

    from alchemfold.diffdock.engine import locate
    from alchemfold.screen_config import default_yaml_text

    python, repo = locate(type("C", (), {})())
    if not python or not repo or not (Path(repo) / "inference.py").is_file():
        print("[FAIL] DiffDock not found — install it (`chemflow diffdock-env install`) and run "
              "`chemflow diffdock-env warmup` first.")
        return False
    receptor = Path(repo) / "data" / "1a0q" / "1a0q_protein_processed.pdb"
    if not receptor.is_file():
        print(f"[FAIL] bundled example receptor not found: {receptor}")
        return False

    jobname = args.jobname + "_diffdock"
    job = (Path(args.workdir).expanduser() / jobname).resolve()
    job.mkdir(parents=True, exist_ok=True)
    smi = job / "ligands.smi"
    smi.write_text("".join(f"{s} {name}\n" for name, s in EXAMPLE_LIGANDS), encoding="utf-8")
    cfg = yaml.safe_load(default_yaml_text(
        engine="diffdock", library="custom", ligand_files=[str(smi)], n_shards=1, shard_size=0,
        jobname=jobname, cleanup=False, receptor_pdb=str(receptor), diffdock_samples=3,
        slurm={"gpus_per_task": 1, "time": "00:30:00"}))
    cfg["diffdock_inference_steps"] = 10
    config = job / "screen.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Example job folder: {job}  (receptor: {receptor})")

    from alchemfold.screen import rank, stage

    stage(config)
    if args.submit:
        rc = subprocess.run(["bash", str(job / "submit_screen.sh")]).returncode
        print(f"[{'PASS' if rc == 0 else 'FAIL'}] sbatch submit. Watch with `squeue -u $USER`; "
              f"when done: python -m alchemfold screen rank {config}")
        return rc == 0

    import time

    from alchemfold.dock_run import run_from_config

    t0 = time.time()
    rc = run_from_config(job / "shards" / "shard_0000.yaml")
    took = time.time() - t0
    rank(config)
    poses = sorted((job / "poses").glob("*.sdf")) if (job / "poses").is_dir() else []
    ok = rc == 0 and len(poses) == len(EXAMPLE_LIGANDS)
    print(f"{len(poses)}/{len(EXAMPLE_LIGANDS)} pose(s) in {job / 'poses'}; ranking: "
          f"{job / 'ranking.csv'}; total {took / 60:.1f} min")
    print(f"[{'PASS' if ok else 'FAIL'}] DiffDock example screen (1 GPU per shard)")
    return ok


def run(args) -> int:
    from alchemfold._gpuinfo import gpu_report

    n = max(1, int(args.gpus))
    print("== AlchemFold GPU test ==")
    for line in gpu_report():
        print(line)
    results: list[tuple[str, bool]] = []

    ok, count = _torch_check()
    results.append(("PyTorch sees GPU(s)", ok))
    if ok and count < n:
        print(f"[FAIL] requested --gpus {n} but only {count} GPU(s) visible")
        results.append((f"{n} GPUs visible", False))
        ok = False
    if ok and n > 1:
        results.append(("multi-GPU srun/NCCL launch", _ddp_probe(n)))
    if args.boltz and (ok or args.boltz_command):
        results.append(("Boltz example screen", _boltz_example(args, n)))
    elif args.boltz:
        print("[SKIP] Boltz example: fix the GPU checks above first")
    if args.diffdock and (ok or args.boltz_command):
        results.append(("DiffDock example screen", _diffdock_example(args)))
    elif args.diffdock:
        print("[SKIP] DiffDock example: fix the GPU checks above first")

    print("\n== Summary ==")
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    all_ok = all(p for _, p in results)
    print("ALCHEMFOLD GPU TEST: " + ("PASSED" if all_ok else "FAILED"))
    return 0 if all_ok else 1
