"""Acquisition / batch-selection for active-learning screening (pure functions).

Given per-candidate surrogate predictions (and optional uncertainty), choose the
next batch to send to the expensive oracle (Glide). Docking/Glide scores follow
the convention **lower is better** (more negative GlideScore = stronger), so by
default we minimise. Set ``lower_is_better=False`` for a score where higher wins
(e.g. a pIC50 surrogate).

Everything here is dependency-free (stdlib only) so it unit-tests without torch.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Candidate:
    cid: str
    pred: float
    sigma: float = 0.0
    cluster: int | None = None  # optional group id for diversity capping


def _acq_value(c: Candidate, strategy: str, kappa: float, lower_is_better: bool,
               rng: random.Random) -> float:
    """Return a score where LARGER = more desirable to pick (we select the top-k)."""
    # Normalise so "want to minimise the oracle score" becomes "maximise -pred".
    direction = -1.0 if lower_is_better else 1.0
    mean = direction * c.pred
    if strategy == "greedy":
        return mean
    if strategy == "ucb":
        # Upper-confidence-bound: reward predicted-good AND uncertain.
        return mean + kappa * c.sigma
    if strategy == "thompson":
        # Sample a plausible value from N(pred, sigma) then rank by it.
        return direction * rng.gauss(c.pred, max(c.sigma, 1e-9))
    if strategy == "random":
        return rng.random()
    raise ValueError(f"unknown acquisition strategy: {strategy!r}")


def select_batch(
    candidates: list[Candidate],
    *,
    k: int,
    strategy: str = "ucb",
    kappa: float = 1.0,
    lower_is_better: bool = True,
    exclude: set[str] | None = None,
    max_per_cluster: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Pick ``k`` candidate ids for the next oracle round.

    - ``strategy``: greedy | ucb | thompson | random
    - ``exclude``: ids already scored by the oracle (never re-pick)
    - ``max_per_cluster``: cap picks sharing a ``cluster`` id, for batch diversity
    """
    rng = random.Random(seed)
    exclude = exclude or set()
    pool = [c for c in candidates if c.cid not in exclude]
    ranked = sorted(
        pool,
        key=lambda c: _acq_value(c, strategy, kappa, lower_is_better, rng),
        reverse=True,
    )
    picked: list[str] = []
    per_cluster: dict[int, int] = {}
    for c in ranked:
        if len(picked) >= k:
            break
        if max_per_cluster is not None and c.cluster is not None:
            if per_cluster.get(c.cluster, 0) >= max_per_cluster:
                continue
            per_cluster[c.cluster] = per_cluster.get(c.cluster, 0) + 1
        picked.append(c.cid)
    return picked


def top_hits(
    scored: dict[str, float], *, n: int, lower_is_better: bool = True
) -> list[tuple[str, float]]:
    """Return the ``n`` best (id, oracle_score) pairs from all scored compounds."""
    items = sorted(scored.items(), key=lambda kv: kv[1], reverse=not lower_is_better)
    return items[:n]


def enrichment_factor(
    scored_order: list[str], actives: set[str], *, fraction: float
) -> float:
    """EF at ``fraction`` — how many more actives are in the top fraction vs random.

    ``scored_order`` is candidate ids sorted best→worst by the surrogate; ``actives``
    are the known/true actives. Useful to track whether the loop is improving.
    """
    if not scored_order or not actives or not 0 < fraction <= 1:
        return 0.0
    n_top = max(1, math.floor(len(scored_order) * fraction))
    top = scored_order[:n_top]
    hits_top = sum(1 for cid in top if cid in actives)
    active_rate = len(actives) / len(scored_order)
    expected = active_rate * n_top
    return (hits_top / expected) if expected > 0 else 0.0
