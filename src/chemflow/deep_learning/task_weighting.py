"""Shared task-loss weighting for partially labeled multitask datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def resolve_task_types(
    task: str,
    target_names: Sequence[str],
    configured: Sequence[str] | Mapping[str, str] | None = None,
) -> list[str]:
    """Resolve one regression/classification type for every target."""
    names = [str(name) for name in target_names]
    overall = str(task).strip().lower()
    if overall in {"regression", "classification"}:
        return [overall] * len(names)
    if overall != "mixed":
        raise ValueError("task must be 'regression', 'classification', or 'mixed'.")
    if configured is None:
        raise ValueError("task='mixed' requires DatasetConfig.task_types.")
    if isinstance(configured, Mapping):
        missing = sorted(set(names) - set(configured))
        extra = sorted(set(configured) - set(names))
        if missing or extra:
            raise ValueError(
                "task_types must match target_column exactly; "
                f"missing={missing}, extra={extra}."
            )
        values = [configured[name] for name in names]
    else:
        values = list(configured)
        if len(values) != len(names):
            raise ValueError(f"Expected {len(names)} task_types, got {len(values)}.")
    resolved = [str(value).strip().lower() for value in values]
    invalid = sorted(set(resolved) - {"regression", "classification"})
    if invalid:
        raise ValueError(f"Unsupported task_types: {invalid}.")
    if len(set(resolved)) < 2:
        raise ValueError("task='mixed' must contain both task types.")
    return resolved


def resolve_task_loss_weights(
    targets: np.ndarray,
    target_names: Sequence[str],
    *,
    strategy: str = "uniform",
    configured: Sequence[float] | Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return label counts and positive task weights normalized to mean one."""
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    names = [str(name) for name in target_names]
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Target matrix width must match the configured task names.")
    counts = np.isfinite(values).sum(axis=0).astype(np.int64)
    if np.any(counts == 0):
        missing = [names[index] for index in np.flatnonzero(counts == 0)]
        raise ValueError(f"Training split has no labels for tasks: {missing}.")

    if configured is not None:
        if isinstance(configured, Mapping):
            missing = sorted(set(names) - set(configured))
            extra = sorted(set(configured) - set(names))
            if missing or extra:
                raise ValueError(
                    "Explicit task_loss_weights must match target_column exactly; "
                    f"missing={missing}, extra={extra}."
                )
            weights = np.asarray([configured[name] for name in names], dtype=float)
        else:
            weights = np.asarray(list(configured), dtype=float)
            if weights.shape != (len(names),):
                raise ValueError(
                    f"Expected {len(names)} explicit task weights, got {len(weights)}."
                )
    else:
        normalized_strategy = str(strategy).strip().lower().replace("-", "_")
        if normalized_strategy in {"uniform", "none"}:
            weights = np.ones(len(names), dtype=float)
        elif normalized_strategy == "inverse_frequency":
            weights = 1.0 / counts.astype(float)
        elif normalized_strategy in {
            "sqrt_inverse_frequency",
            "inverse_sqrt_frequency",
        }:
            weights = 1.0 / np.sqrt(counts.astype(float))
        else:
            raise ValueError(
                "task_loss_weighting must be 'uniform', "
                "'sqrt_inverse_frequency', or 'inverse_frequency'."
            )

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("All task loss weights must be finite and greater than zero.")
    weights = weights / weights.mean()
    return counts, weights.astype(np.float32)
