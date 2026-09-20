from __future__ import annotations

import numpy as np

from chemflow.deep_learning.task_weighting import (
    resolve_task_loss_weights,
    resolve_task_types,
)


def test_mixed_task_types_follow_target_order():
    assert resolve_task_types(
        "mixed",
        ["solubility", "active"],
        {"active": "classification", "solubility": "regression"},
    ) == ["regression", "classification"]


def test_mixed_task_types_require_both_families():
    try:
        resolve_task_types("mixed", ["first", "second"], ["regression", "regression"])
    except ValueError as error:
        assert "both task types" in str(error)
    else:
        raise AssertionError("A homogeneous task_types list must be rejected for task='mixed'.")


def test_inverse_frequency_equalizes_total_labeled_weight():
    targets = np.asarray(
        [[1.0, 1.0], [2.0, np.nan], [3.0, np.nan], [4.0, np.nan]]
    )
    counts, weights = resolve_task_loss_weights(
        targets, ["dense", "sparse"], strategy="inverse_frequency"
    )
    np.testing.assert_array_equal(counts, [4, 1])
    assert np.isclose(counts[0] * weights[0], counts[1] * weights[1])
    assert np.isclose(weights.mean(), 1.0)


def test_explicit_task_weights_follow_target_order():
    targets = np.ones((3, 2))
    _, weights = resolve_task_loss_weights(
        targets,
        ["first", "second"],
        configured={"second": 3.0, "first": 1.0},
    )
    np.testing.assert_allclose(weights, [0.5, 1.5])
