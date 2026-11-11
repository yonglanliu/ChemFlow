"""Per-rank probe launched by ``alchemfold gpu-test`` through ``srun``.

Each SLURM task (= one GPU) prints its rank, node and CUDA device, then joins an
NCCL all-reduce. If every rank prints OK with a DIFFERENT device, multi-GPU launch
(srun + one task per GPU) works for Boltz/Lightning DDP.
"""

from __future__ import annotations

import os
import socket
import sys


def main() -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    n_dev = torch.cuda.device_count()
    use_cuda = n_dev > 0
    dev = local if local < n_dev else 0
    if use_cuda:
        torch.cuda.set_device(dev)
    name = torch.cuda.get_device_name(dev) if use_cuda else "cpu"

    os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR")
                          or socket.gethostname())
    os.environ.setdefault("MASTER_PORT", str(29500 + int(os.environ.get("SLURM_JOB_ID", "0")) % 1000))
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world)

    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend, init_method="env://")
    t = torch.tensor([float(rank)], device=f"cuda:{dev}" if use_cuda else "cpu")
    dist.all_reduce(t)
    expected = float(sum(range(world)))
    ok = abs(t.item() - expected) < 1e-6
    print(f"[rank {rank}/{world}] host={socket.gethostname()} device=cuda:{dev} "
          f"({name}) visible={n_dev} all_reduce={t.item():.0f} expected={expected:.0f} "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
