from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.cli.main import build_parser
from src.deep_learning.chemeleon.pretrained import ensure_pretrained_weights
from src.deep_learning.chemeleon.trainer import _evaluate, load_config


class CheMeleonCliTest(unittest.TestCase):
    def test_train_subcommand_is_registered(self):
        args = build_parser().parse_args(
            ["train", "chemeleon", "config.toml"]
        )
        self.assertEqual(args.model, "chemeleon")
        self.assertEqual(args.func.__name__, "train_chemeleon")


class CheMeleonConfigTest(unittest.TestCase):
    def test_minimal_config_uses_expected_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                """
[BaseConfig]
workdir = "run"
task = "regression"

[DatasetConfig]
dataset_path = "data.csv"
target_column = "activity"
""".strip(),
                encoding="utf-8",
            )
            config = load_config(path)

        self.assertEqual(config.base.task, "regression")
        self.assertEqual(config.dataset.smiles_column, "SMILES")
        self.assertIsNone(config.dataset.test_dataset_path)
        self.assertEqual(config.training.num_epochs, 30)
        self.assertEqual(config.training.num_workers, 0)
        self.assertEqual(config.training.strategy, "auto")
        self.assertEqual(config.training.num_nodes, 1)
        self.assertEqual(config.model.ffn_hidden_dim, 256)

    def test_external_test_dataset_requires_zero_test_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                """
[BaseConfig]
workdir = "run"

[DatasetConfig]
dataset_path = "train.csv"
test_dataset_path = "test.csv"
test_fraction = 0.1
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "test_fraction must be 0"):
                load_config(path)

    def test_mps_rejects_ddp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                """
[BaseConfig]
workdir = "run"

[DatasetConfig]
dataset_path = "data.csv"

[CheMeleonTrainingConfig]
accelerator = "mps"
devices = 2
strategy = "ddp"
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "MPS supports only"):
                load_config(path)

    def test_custom_weight_file_is_checksum_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            content = b"trusted test weights"
            path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            resolved = ensure_pretrained_weights(path, sha256=digest)
            self.assertEqual(resolved, path.resolve())


class CheMeleonEvaluationTest(unittest.TestCase):
    def test_regression_metrics_use_chemflow_names(self):
        metrics = _evaluate(
            "regression",
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.0, 2.0, 3.0]),
        )
        self.assertAlmostEqual(metrics["test_rmse"], 0.0)
        self.assertAlmostEqual(metrics["test_r2"], 1.0)

    def test_classification_metrics_accept_probabilities(self):
        metrics = _evaluate(
            "classification",
            np.asarray([0.1, 0.9, 0.2, 0.8]),
            np.asarray([0.0, 1.0, 0.0, 1.0]),
        )
        self.assertAlmostEqual(metrics["test_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["test_roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
