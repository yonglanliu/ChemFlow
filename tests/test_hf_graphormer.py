from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from chemflow.cli.main import build_parser
from chemflow.deep_learning.hf_graphormer.trainer import (
    HuggingFaceGraphormerTrainer,
    _classification_metrics,
    _metrics,
    _normalise_split,
    _validate_split_config,
)


def test_hf_graphormer_cli_is_registered():
    args = build_parser().parse_args(
        ["train", "hf-graphormer", "reference.toml"]
    )
    assert args.model == "hf-graphormer"
    assert args.config == "reference.toml"


def test_hf_graphormer_split_aliases():
    assert _normalise_split("training") == "train"
    assert _normalise_split("VALID") == "val"
    assert _normalise_split("testing") == "test"
    assert _normalise_split("unknown") is None


def test_hf_graphormer_predefined_split_config():
    config = {"split_type": "predefined", "split_column": "split"}
    _validate_split_config(config)
    assert config["split_type"] == "predefined"

    inferred = {"split_type": "scaffold_balanced", "split_column": "partition"}
    _validate_split_config(inferred)
    assert inferred["split_type"] == "predefined"

    try:
        _validate_split_config({"split_type": "predefined"})
    except ValueError as error:
        assert "requires DatasetConfig.split_column" in str(error)
    else:
        raise AssertionError("predefined split without split_column was accepted")


def test_hf_graphormer_regression_metrics():
    actual = np.asarray([1.0, 2.0, 3.0])
    predicted = np.asarray([1.0, 2.0, 3.0])
    metrics = _metrics(actual, predicted)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert np.isclose(metrics["pearson"], 1.0)
    assert np.isclose(metrics["spearman"], 1.0)


def test_hf_graphormer_masked_loss_balances_tasks():
    logits = torch.tensor([[0.0, 2.0], [2.0, 8.0], [4.0, 0.0]])
    labels = torch.tensor([[0.0, 0.0], [0.0, float("nan")], [0.0, float("nan")]])
    weights = torch.tensor([0.5, 1.5])
    expected = (0.0 * 0.5 + 4.0 * 0.5 + 16.0 * 0.5 + 4.0 * 1.5) / 3.0
    actual = HuggingFaceGraphormerTrainer._masked_multitask_loss(
        logits, labels, weights
    )
    assert np.isclose(float(actual), expected)


def test_hf_graphormer_binary_classification_metrics():
    actual = np.asarray([0.0, 0.0, 1.0, 1.0])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
    metrics = _classification_metrics(actual, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["loss"] > 0.0


def test_hf_graphormer_masked_classification_loss():
    logits = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([[0.0, float("nan")], [1.0, 1.0]])
    weights = torch.tensor([1.0, 2.0])
    actual = HuggingFaceGraphormerTrainer._masked_multitask_loss(
        logits,
        labels,
        weights,
        task="classification",
    )
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(0.0), torch.tensor(0.0)
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(0.0), torch.tensor(1.0)
        )
        + 2.0
        * torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(2.0), torch.tensor(1.0)
        )
    ) / 4.0
    assert np.isclose(float(actual), float(expected))


def test_hf_graphormer_masked_mixed_task_loss():
    logits = torch.tensor([[2.0, 0.0], [4.0, 2.0]])
    labels = torch.tensor([[1.0, 0.0], [3.0, 1.0]])
    weights = torch.ones(2)
    actual = HuggingFaceGraphormerTrainer._masked_multitask_loss(
        logits,
        labels,
        weights,
        task="mixed",
        task_types=["regression", "classification"],
    )
    expected = (
        1.0
        + 1.0
        + float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(0.0), torch.tensor(0.0)))
        + float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(2.0), torch.tensor(1.0)))
    ) / 4.0
    assert np.isclose(float(actual), expected)


def test_hf_graphormer_classification_evaluation_outputs_probabilities_and_classes():
    trainer = object.__new__(HuggingFaceGraphormerTrainer)
    trainer.task = "classification"
    trainer.task_types = ["classification", "classification"]
    trainer.threshold = 0.5
    trainer.device = torch.device("cpu")
    trainer.data_cfg = {"target_column": ["endpoint_a", "endpoint_b"]}

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, **inputs):
            return SimpleNamespace(
                logits=torch.tensor([[-2.0, 2.0], [2.0, -2.0]])
            )

    loader = [
        {
            "names": ["first", "second"],
            "smiles": ["CC", "CCC"],
            "original_indices": [0, 1],
            "raw_targets": [[0.0, 1.0], [1.0, 0.0]],
            "labels": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "input_nodes": torch.ones((2, 1), dtype=torch.long),
        }
    ]
    metrics, frame = trainer._evaluate(
        FakeModel(), loader, np.zeros(2), np.ones(2)
    )
    assert frame["endpoint_a_class"].tolist() == [0, 1]
    assert frame["endpoint_b_class"].tolist() == [1, 0]
    assert np.allclose(
        frame["endpoint_a_prediction"], frame["endpoint_a_probability"]
    )
    assert frame["endpoint_a_prediction"].between(0.0, 1.0).all()
    assert metrics["endpoint_a"]["accuracy"] == 1.0
    assert metrics["endpoint_b"]["roc_auc"] == 1.0


def test_hf_graphormer_mixed_evaluation_uses_task_specific_outputs():
    trainer = object.__new__(HuggingFaceGraphormerTrainer)
    trainer.task = "mixed"
    trainer.task_types = ["regression", "classification"]
    trainer.threshold = 0.5
    trainer.device = torch.device("cpu")
    trainer.data_cfg = {"target_column": ["solubility", "active"]}

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, **inputs):
            return SimpleNamespace(logits=torch.tensor([[1.0, -2.0], [2.0, 2.0]]))

    loader = [{
        "names": ["first", "second"],
        "smiles": ["CC", "CCC"],
        "original_indices": [0, 1],
        "raw_targets": [[12.0, 0.0], [14.0, 1.0]],
        "labels": torch.tensor([[1.0, 0.0], [2.0, 1.0]]),
        "input_nodes": torch.ones((2, 1), dtype=torch.long),
    }]
    metrics, frame = trainer._evaluate(
        FakeModel(), loader, np.asarray([10.0, 0.0]), np.asarray([2.0, 1.0])
    )
    assert frame["solubility_prediction"].tolist() == [12.0, 14.0]
    assert frame["active_class"].tolist() == [0, 1]
    assert "solubility_class" not in frame
    assert metrics["solubility"]["loss"] == 0.0
    assert metrics["active"]["accuracy"] == 1.0
