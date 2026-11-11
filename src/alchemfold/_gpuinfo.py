"""Print where a run is executing: node, SLURM job and the GPU(s) it got.

Lets a user see at the top of every log which GPU model a job landed on (the
single biggest factor in co-folding speed) without logging in to the node.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess

# GPUs without native bfloat16: Boltz's bf16 autocast runs far slower on these.
_NO_BF16 = ("P100", "V100", "P40", "K80", "T4", "GTX", "Quadro P", "Quadro M")


def gpu_report() -> list[str]:
    """Human-readable lines describing the host, SLURM job and visible GPUs."""
    lines = [f"Host: {socket.gethostname()}"]
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        lines.append(f"SLURM job: {job}" + (f" (array task {task})" if task is not None else ""))
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        lines.append(f"CUDA_VISIBLE_DEVICES: {cvd or '(empty)'}")

    exe = shutil.which("nvidia-smi")
    if not exe:
        lines.append("GPU: nvidia-smi not found (CPU node, or driver tools not on PATH)")
        return lines
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive
        lines.append(f"GPU: could not query nvidia-smi ({exc})")
        return lines
    if not out:
        lines.append("GPU: none visible")
        return lines
    names = []
    for row in out.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) >= 4:
            idx, name, mem, drv = parts[:4]
            lines.append(f"GPU {idx}: {name}, {mem}, driver {drv}")
            names.append(name)
        else:
            lines.append(f"GPU: {row}")
    slow = [n for n in names if any(tag in n for tag in _NO_BF16)]
    if slow:
        lines.append(
            f"WARNING: {slow[0]} has no native bfloat16 — Boltz bf16 autocast is much "
            "slower here; an A100/L40/H100 is typically several times faster.")
    return lines


def print_run_info(conf=None) -> None:
    """Print the GPU report (skipped for CPU-only runs)."""
    if conf is not None and str(getattr(conf, "accelerator", "gpu")) != "gpu":
        print("Accelerator: cpu")
        return
    for line in gpu_report():
        print(line, flush=True)
