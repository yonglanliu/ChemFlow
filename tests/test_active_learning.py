"""Tests for the active-learning acquisition logic (pure, no torch/Glide needed)."""

from chemflow.active_learning.acquisition import (
    Candidate,
    enrichment_factor,
    select_batch,
    top_hits,
)


def _cands():
    # pred: lower = better (Glide). c5 is best, c1 worst.
    return [
        Candidate("c1", pred=-6.0, sigma=0.1),
        Candidate("c2", pred=-7.0, sigma=0.2),
        Candidate("c3", pred=-8.0, sigma=2.0),   # good + very uncertain
        Candidate("c4", pred=-9.0, sigma=0.1),
        Candidate("c5", pred=-10.0, sigma=0.1),  # best mean
    ]


def test_greedy_picks_best_mean():
    picks = select_batch(_cands(), k=2, strategy="greedy", lower_is_better=True)
    assert picks == ["c5", "c4"]


def test_ucb_rewards_uncertainty():
    # With a large kappa, the uncertain c3 should beat the confident c4.
    picks = select_batch(_cands(), k=2, strategy="ucb", kappa=5.0, lower_is_better=True)
    assert "c3" in picks and "c5" in picks


def test_exclude_already_scored():
    picks = select_batch(_cands(), k=3, strategy="greedy", lower_is_better=True,
                         exclude={"c5", "c4"})
    assert "c5" not in picks and "c4" not in picks
    assert picks[0] == "c3"


def test_max_per_cluster_diversifies():
    cands = [
        Candidate("a1", -10.0, cluster=1),
        Candidate("a2", -9.9, cluster=1),
        Candidate("a3", -9.8, cluster=1),
        Candidate("b1", -8.0, cluster=2),
    ]
    picks = select_batch(cands, k=2, strategy="greedy", lower_is_better=True,
                         max_per_cluster=1)
    assert picks == ["a1", "b1"]  # only one from cluster 1


def test_higher_is_better_direction():
    cands = [Candidate("x", 5.0), Candidate("y", 9.0), Candidate("z", 1.0)]
    picks = select_batch(cands, k=1, strategy="greedy", lower_is_better=False)
    assert picks == ["y"]


def test_top_hits_lower_is_better():
    scored = {"a": -6.0, "b": -9.0, "c": -7.5}
    assert top_hits(scored, n=2) == [("b", -9.0), ("c", -7.5)]


def test_enrichment_factor():
    order = [f"c{i}" for i in range(100)]
    actives = {"c0", "c1", "c2", "c3", "c4"}  # all in the top 10%
    ef = enrichment_factor(order, actives, fraction=0.1)
    assert ef == 10.0  # perfect enrichment: 5 of 5 actives in top 10%
