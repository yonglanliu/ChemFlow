"""GPU detection shared by every module.

Consolidates what used to be ``alchemfold/_gpuinfo.py`` plus the ad-hoc
``nvidia-smi`` heredoc the desktop pasted into generated SLURM scripts, so a
log always opens with the same report: which node, which SLURM job, which GPU.
GPU model is the single biggest factor in co-folding and MD throughput.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess

# GPUs without native bfloat16: bf16 autocast (Boltz, OpenFold-style models)
# runs far slower on these.
NO_BF16 = ("P100", "V100", "P40", "K80", "T4", "GTX", "Quadro P", "Quadro M")


def visible_gpus(timeout: int = 20) -> list[dict[str, str]]:
    """``[{index, name, memory, driver}, ...]`` from nvidia-smi; empty if none."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout, check=False).stdout.strip()
    except Exception:
        return []
    gpus = []
    for row in out.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) >= 4:
            gpus.append({"index": parts[0], "name": parts[1],
                         "memory": parts[2], "driver": parts[3]})
    return gpus


def slow_bf16(names: list[str]) -> list[str]:
    """Which of these GPU names lack native bfloat16."""
    return [n for n in names if any(tag in n for tag in NO_BF16)]


def gpu_report() -> list[str]:
    """Human-readable lines describing the host, SLURM job and visible GPUs."""
    lines = [f"Host: {socket.gethostname()}"]
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        lines.append(f"SLURM job: {job}"
                     + (f" (array task {task})" if task is not None else ""))
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        lines.append(f"CUDA_VISIBLE_DEVICES: {cvd or '(empty)'}")

    if not shutil.which("nvidia-smi"):
        lines.append("GPU: nvidia-smi not found (CPU node, or driver tools not on PATH)")
        return lines
    gpus = visible_gpus()
    if not gpus:
        lines.append("GPU: none visible")
        return lines
    for gpu in gpus:
        lines.append(f"GPU {gpu['index']}: {gpu['name']}, {gpu['memory']}, "
                     f"driver {gpu['driver']}")
    slow = slow_bf16([g["name"] for g in gpus])
    if slow:
        lines.append(
            f"WARNING: {slow[0]} has no native bfloat16 — bf16 autocast is much "
            "slower here; an A100/L40/H100 is typically several times faster.")
    return lines


def print_run_info(conf=None) -> None:
    """Print the GPU report (skipped for CPU-only runs)."""
    if conf is not None and str(getattr(conf, "accelerator", "gpu")) != "gpu":
        print("Accelerator: cpu")
        return
    for line in gpu_report():
        print(line, flush=True)


def report_snippet() -> str:
    """A shell fragment that prints the same report from inside a SLURM script.

    Used by the generated job scripts, which run before this package is
    necessarily importable in the activated env.
    """
    return (
        'echo "Host: $(hostname)"\n'
        'echo "SLURM job: ${SLURM_JOB_ID:-none}'
        '${SLURM_ARRAY_TASK_ID:+ (array task $SLURM_ARRAY_TASK_ID)}"\n'
        "command -v nvidia-smi >/dev/null && nvidia-smi -L || "
        'echo "GPU: nvidia-smi not found"\n'
    )
