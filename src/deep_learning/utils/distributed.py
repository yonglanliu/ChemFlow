import torch.distributed as dist
import torch
import sys
import os

# ============================================================
# Distributed helpers
# ============================================================

def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_dist_available_and_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_dist_available_and_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def main_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def disable_tqdm() -> bool:
    """Disable tqdm in Slurm logs, non-interactive shells, and non-main ranks."""
    return (
        not sys.stderr.isatty()
        or "SLURM_JOB_ID" in os.environ
        or not is_main_process()
    )

def setup_distributed():
    """
    Supports:
    - Single GPU
    - Single CPU/MPS
    - Multi-GPU with torchrun
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))  # Get world size from environment variable, default to 1 for single GPU/CPU/MPS

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA GPUs.")

        local_rank = int(os.environ["LOCAL_RANK"])  # Get local rank from environment variable set by torchrun
        torch.cuda.set_device(local_rank)  # Set the current CUDA device to the local rank for this process

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

        device = torch.device("cuda", local_rank)
        distributed = True

    else:
        distributed = False

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return device, distributed


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


# ============================================================
# Utils
# ============================================================
def reduce_mean(value: float, device: torch.device) -> float:
    if not is_dist_available_and_initialized():
        return value

    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()

    return tensor.item()



def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    """Needed when resuming optimizer states onto GPU/MPS."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)