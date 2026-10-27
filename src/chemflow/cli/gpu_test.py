"""PyTorch CUDA smoke test for ChemFlow."""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TorchGPUCheckResult:
    """Result of a PyTorch CUDA smoke test."""

    passed: bool
    detail: str


def _torch_import_failure(error: ImportError) -> TorchGPUCheckResult:
    message = str(error).strip() or repr(error)
    if "GLIBCXX_" in message and "libstdc++.so.6" in message:
        return TorchGPUCheckResult(
            False,
            "PyTorch import failed because an incompatible system libstdc++.so.6 "
            "was loaded. Ensure the environment contains libstdcxx-ng and put "
            "$CONDA_PREFIX/lib before system library directories in "
            "LD_LIBRARY_PATH. Original error: "
            + message,
        )
    return TorchGPUCheckResult(False, f"PyTorch import failed: {message}")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def test_torch_gpu(
    device_index: int,
    matrix_size: int,
    iterations: int,
) -> TorchGPUCheckResult:
    """Run synchronized float32 matrix multiplications on one CUDA device."""

    try:
        import torch
    except ImportError as error:
        return _torch_import_failure(error)

    if not torch.cuda.is_available():
        return TorchGPUCheckResult(
            False,
            f"torch.cuda.is_available() is False; PyTorch={torch.__version__}, "
            f"compiled CUDA={torch.version.cuda}",
        )
    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        return TorchGPUCheckResult(
            False,
            f"CUDA device {device_index} was requested, but only {device_count} "
            "visible device(s) were detected",
        )

    try:
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        properties = torch.cuda.get_device_properties(device_index)
        torch.cuda.reset_peak_memory_stats(device)

        left = torch.randn((matrix_size, matrix_size), device=device)
        right = torch.randn((matrix_size, matrix_size), device=device)

        # Warm up CUDA context creation and the matrix-multiplication kernel.
        product = left @ right
        torch.cuda.synchronize(device)

        started = time.perf_counter()
        for _ in range(iterations):
            product = left @ right
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        checksum = float(product.mean().cpu())
        if not math.isfinite(checksum):
            return TorchGPUCheckResult(False, "CUDA calculation produced a nonfinite value")
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        milliseconds = elapsed * 1000.0 / iterations
        capability = torch.cuda.get_device_capability(device_index)
        return TorchGPUCheckResult(
            True,
            f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; "
            f"device {device_index}: {properties.name}; "
            f"compute capability {capability[0]}.{capability[1]}; "
            f"{matrix_size}x{matrix_size} float32 matmul "
            f"{milliseconds:.2f} ms/iteration; peak {peak_mib:.1f} MiB",
        )
    except (OSError, RuntimeError, ValueError) as error:
        return TorchGPUCheckResult(False, f"{type(error).__name__}: {error}")


def run_gpu_test(args: argparse.Namespace) -> int:
    """Run the CLI test and convert its result to a process exit status."""

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    result = test_torch_gpu(args.device, args.matrix_size, args.iterations)
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] PyTorch CUDA: {result.detail}")
    print("CHEMFLOW GPU TEST: PASSED" if result.passed else "CHEMFLOW GPU TEST: FAILED")
    return 0 if result.passed else 1


def add_gpu_test_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register ``chemflow gpu-test``."""

    parser = subparsers.add_parser(
        "gpu-test",
        help="Run a quick PyTorch CUDA calculation",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Visible CUDA device index (default: 0)",
    )
    parser.add_argument(
        "--matrix-size",
        type=_positive_integer,
        default=1024,
        help="Square float32 matrix dimension (default: 1024)",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_integer,
        default=3,
        help="Timed matrix multiplications after warmup (default: 3)",
    )
    parser.set_defaults(func=run_gpu_test)
    return parser

