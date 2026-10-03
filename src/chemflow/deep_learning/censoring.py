"""Shared utilities for one-sided (censored) regression targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch


EXACT = 0
UPPER_BOUND = -1  # ``y < limit``: predictions at or below the limit are valid.
LOWER_BOUND = 1   # ``y > limit``: predictions at or above the limit are valid.

_RELATIONS = {
    "": EXACT,
    "=": EXACT,
    "==": EXACT,
    "eq": EXACT,
    "exact": EXACT,
    "<": UPPER_BOUND,
    "<=": UPPER_BOUND,
    "≤": UPPER_BOUND,
    "lt": UPPER_BOUND,
    "le": UPPER_BOUND,
    ">": LOWER_BOUND,
    ">=": LOWER_BOUND,
    "≥": LOWER_BOUND,
    "gt": LOWER_BOUND,
    "ge": LOWER_BOUND,
}


def resolve_relation_columns(
    frame: pd.DataFrame,
    targets: Sequence[str],
    configured: str | Sequence[str] | Mapping[str, str] | None = None,
) -> list[str | None]:
    """Resolve relation columns in target order.

    With no explicit configuration, ``<target>_relation`` is detected for each
    target. For a single target, the conventional ``relation`` column is also
    detected. Missing relation columns mean exact (uncensored) observations.
    """
    targets = [str(value) for value in targets]
    if configured is None:
        resolved = [
            f"{target}_relation" if f"{target}_relation" in frame.columns else None
            for target in targets
        ]
        if len(targets) == 1 and resolved[0] is None and "relation" in frame.columns:
            resolved[0] = "relation"
        return resolved
    if isinstance(configured, str):
        if len(targets) != 1:
            raise ValueError(
                "A single DatasetConfig.relation_column can only be used with "
                "one target; provide a list or target-to-column table."
            )
        resolved = [configured]
    elif isinstance(configured, Mapping):
        unknown = sorted(set(configured) - set(targets))
        if unknown:
            raise ValueError(f"relation_column contains unknown targets: {unknown}")
        resolved = [configured.get(target) for target in targets]
    else:
        resolved = list(configured)
        if len(resolved) != len(targets):
            raise ValueError("relation_column must contain one column per target.")
    resolved = [str(value).strip() if value is not None else None for value in resolved]
    missing = sorted({value for value in resolved if value and value not in frame.columns})
    if missing:
        raise KeyError(f"Dataset is missing relation columns: {missing}")
    return resolved


def relation_codes(
    frame: pd.DataFrame,
    targets: Sequence[str],
    relation_columns: Sequence[str | None],
    *,
    task_types: Sequence[str] | None = None,
) -> np.ndarray:
    """Return an ``n_rows x n_tasks`` matrix of exact/upper/lower codes."""
    codes = np.zeros((len(frame), len(targets)), dtype=np.int8)
    types = list(task_types or ["regression"] * len(targets))
    for index, (target, column, task_type) in enumerate(
        zip(targets, relation_columns, types)
    ):
        if column is None:
            continue
        if task_type != "regression":
            nonempty = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
            nonexact = nonempty & ~frame[column].astype(str).str.strip().str.lower().isin(
                {"=", "==", "eq", "exact"}
            )
            if nonexact.any():
                raise ValueError(f"Classification target {target!r} cannot be censored.")
            continue
        for row_position, value in enumerate(frame[column].tolist()):
            key = "" if pd.isna(value) else str(value).strip().lower()
            if key not in _RELATIONS:
                raise ValueError(
                    f"Unsupported relation {value!r} for target {target!r}; "
                    "use =, <, <=, >, or >=."
                )
            codes[row_position, index] = _RELATIONS[key]
    return codes


def censor_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    """Clamp predictions that already satisfy their reported one-sided bound."""
    codes = codes.to(device=predictions.device)
    satisfied_upper = (codes < 0) & (predictions <= targets)
    satisfied_lower = (codes > 0) & (predictions >= targets)
    return torch.where(satisfied_upper | satisfied_lower, targets, predictions)


def censor_predictions_from_masks(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lt_mask: torch.Tensor | None,
    gt_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Chemprop-compatible form where lt marks ``<`` and gt marks ``>``."""
    if lt_mask is None and gt_mask is None:
        return predictions
    codes = torch.zeros_like(targets, dtype=torch.int8)
    if lt_mask is not None:
        codes = torch.where(lt_mask.bool(), -torch.ones_like(codes), codes)
    if gt_mask is not None:
        codes = torch.where(gt_mask.bool(), torch.ones_like(codes), codes)
    return censor_predictions(predictions, targets, codes)


def prefixed_targets(
    frame: pd.DataFrame,
    targets: Sequence[str],
    codes: np.ndarray,
) -> pd.DataFrame:
    """Encode bounds as ``<value``/``>value`` for CLI backends."""
    output = frame.copy()
    for index, target in enumerate(targets):
        numeric = pd.to_numeric(output[target], errors="coerce")
        values: list[Any] = []
        for value, code in zip(numeric, codes[:, index]):
            if not np.isfinite(value):
                values.append(np.nan)
            elif code < 0:
                values.append(f"<{value:.17g}")
            elif code > 0:
                values.append(f">{value:.17g}")
            else:
                values.append(float(value))
        output[target] = values
    return output


def parse_prefixed_targets(
    frame: pd.DataFrame,
    targets: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode numeric, ``<value``, and ``>value`` target cells.

    This is primarily used to recover censoring metadata from CLI-oriented
    prepared files (for example Chemprop inputs) during test evaluation.
    """
    values = np.full((len(frame), len(targets)), np.nan, dtype=float)
    codes = np.zeros((len(frame), len(targets)), dtype=np.int8)
    for target_index, target in enumerate(targets):
        for row_index, raw in enumerate(frame[target].tolist()):
            if pd.isna(raw):
                continue
            text = str(raw).strip()
            if not text:
                continue
            if text.startswith("<="):
                codes[row_index, target_index], text = UPPER_BOUND, text[2:]
            elif text.startswith(">="):
                codes[row_index, target_index], text = LOWER_BOUND, text[2:]
            elif text.startswith("<"):
                codes[row_index, target_index], text = UPPER_BOUND, text[1:]
            elif text.startswith(">"):
                codes[row_index, target_index], text = LOWER_BOUND, text[1:]
            try:
                values[row_index, target_index] = float(text.strip())
            except ValueError:
                continue
    return values, codes
