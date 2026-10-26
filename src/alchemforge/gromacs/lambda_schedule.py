"""Shared lambda-window generation for GROMACS free-energy workflows."""

from __future__ import annotations

import math


def generate_lambda_windows(
    count: int, function: str = "cosine", power: float = 2.0
) -> list[float]:
    """Generate ``count`` monotonically increasing values from zero to one."""
    if count < 2:
        raise ValueError("Lambda window count must be at least 2")
    if power <= 0:
        raise ValueError("Lambda power must be greater than zero")
    name = function.lower()
    fractions = [index / (count - 1) for index in range(count)]
    if name == "linear":
        values = fractions
    elif name == "cosine":
        values = [(1.0 - math.cos(math.pi * value)) / 2.0 for value in fractions]
    elif name == "power":
        values = [value**power for value in fractions]
    else:
        raise ValueError("Lambda function must be one of: cosine, linear, power")
    values[0] = 0.0
    values[-1] = 1.0
    return values
