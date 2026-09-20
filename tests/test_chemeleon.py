from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from chemflow.cli.main import build_parser
from chemflow.deep_learning.chemeleon.pretrained import ensure_pretrained_weights
from chemflow.deep_learning.chemeleon.mixed_predictor import MixedTaskMetric
from chemflow.deep_learning.chemeleon.trainer import (
    _evaluate,
    _load_transfer_encoder,
    _molecules_for_split,
    load_config,
)


class CheMeleonCliTest(unittest.TestCase):
    def test_train_subcommand_is_registered(self):
        args = build_parser().parse_args(
            ["train", "chemeleon", "config.toml"]
        )
        self.assertEqual(args.model, "chemeleon")
        self.assertEqual(args.func.__name__, "train_chemeleon")

    def test_applicability_controls_belong_to_prediction(self):
        args = build_parser().parse_args(
            [
                "predict",
                "chemeleon",
                "--smiles",
                "CCO",
                "--model-checkpoint",
                "best.ckpt",
                "--embedding-dimensions",
                "64",
                "--calibration-confidence",
                "0.95",
                "--similarity-radius",
                "3",
                "--similarity-bits",
                "1024",
                "--mc-dropout-samples",
                "30",
                "--output",
                "predictions.csv",
            ]
        )
        self.assertEqual(args.embedding_dimensions, 64)
        self.assertEqual(args.calibration_confidence, 0.95)
        self.assertEqual(args.similarity_radius, 3)
        self.assertEqual(args.similarity_bits, 1024)
        self.assertEqual(args.mc_dropout_samples, 30)


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
        self.assertTrue(config.training.inspect_task_metrics)
        self.assertEqual(config.model.ffn_hidden_dim, 256)

    def test_mixed_task_types_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                """
[BaseConfig]
workdir = "run"
task = "mixed"

[DatasetConfig]
dataset_path = "data.csv"
target_column = ["solubility", "active"]
task_types = ["regression", "classification"]
""".strip(),
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config.base.task, "mixed")
        self.assertEqual(
            config.dataset.task_types, ["regression", "classification"]
        )

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

    def test_predefined_split_requires_and_accepts_split_column(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_column = root / "missing_column.toml"
            missing_column.write_text(
                """
[BaseConfig]
workdir = "run"

[DatasetConfig]
dataset_path = "data.csv"
split_type = "predefined"
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires DatasetConfig.split_column"):
                load_config(missing_column)

            valid = root / "valid.toml"
            valid.write_text(
                """
[BaseConfig]
workdir = "run"

[DatasetConfig]
dataset_path = "data.csv"
split_type = "predefined"
split_column = "split"
""".strip(),
                encoding="utf-8",
            )
            config = load_config(valid)

        self.assertEqual(config.dataset.split_type, "predefined")
        self.assertEqual(config.dataset.split_column, "split")

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

    def test_transfer_options_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                """
[BaseConfig]
workdir = "run"

[DatasetConfig]
dataset_path = "data.csv"

[CheMeleonConfig]
transfer_checkpoint = "multitask.ckpt"
transfer_encoder_only = true
freeze_epochs = 3
""".strip(),
                encoding="utf-8",
            )
            config = load_config(path)

        self.assertEqual(config.model.transfer_checkpoint, "multitask.ckpt")
        self.assertTrue(config.model.transfer_encoder_only)
        self.assertEqual(config.model.freeze_epochs, 3)

    def test_transfer_loader_ignores_incompatible_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multitask.ckpt"
            source = torch.nn.Linear(2, 3)
            with torch.no_grad():
                source.weight.fill_(2.0)
                source.bias.fill_(3.0)
            torch.save(
                {
                    "state_dict": {
                        "message_passing.weight": source.weight.detach().clone(),
                        "message_passing.bias": source.bias.detach().clone(),
                        "predictor.output.weight": torch.zeros((2, 3)),
                    }
                },
                path,
            )
            destination = torch.nn.Linear(2, 3)
            loaded = _load_transfer_encoder(destination, path)

        self.assertEqual(loaded, 2)
        torch.testing.assert_close(destination.weight, source.weight)
        torch.testing.assert_close(destination.bias, source.bias)


class CheMeleonEvaluationTest(unittest.TestCase):
    def test_mixed_validation_metric_uses_classification_logits(self):
        metric = MixedTaskMetric(["regression", "classification"])
        predictions = torch.tensor([[1.0, 0.0], [3.0, 2.0]])
        targets = torch.tensor([[0.0, 0.0], [2.0, 1.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        metric.update(predictions, targets, mask)
        expected = (
            1.0
            + 1.0
            + float(torch.nn.functional.binary_cross_entropy_with_logits(
                torch.tensor(0.0), torch.tensor(0.0)
            ))
            + float(torch.nn.functional.binary_cross_entropy_with_logits(
                torch.tensor(2.0), torch.tensor(1.0)
            ))
        ) / 4.0
        self.assertAlmostEqual(float(metric.compute()), expected)

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

    def test_mixed_metrics_use_each_tasks_metric_family(self):
        metrics = _evaluate(
            "mixed",
            np.asarray([[1.0, 0.1], [2.0, 0.9], [3.0, 0.8]]),
            np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0]]),
            ["solubility", "active"],
            ["regression", "classification"],
        )
        self.assertAlmostEqual(metrics["test_solubility_rmse"], 0.0)
        self.assertAlmostEqual(metrics["test_active_accuracy"], 1.0)


class CheMeleonSplitTest(unittest.TestCase):
    def test_scaffold_split_removes_stereo_from_copies_only(self):
        molecule = Chem.MolFromSmiles("F/C=C/F")
        records = [{"mol": molecule}]

        split_molecules = _molecules_for_split(records, "scaffold_balanced")

        self.assertIsNot(split_molecules[0], molecule)
        self.assertTrue(
            any(
                bond.GetStereo() != Chem.BondStereo.STEREONONE
                for bond in molecule.GetBonds()
            )
        )
        self.assertTrue(
            all(
                bond.GetStereo() == Chem.BondStereo.STEREONONE
                for bond in split_molecules[0].GetBonds()
            )
        )

    def test_non_scaffold_split_reuses_original_molecules(self):
        molecule = Chem.MolFromSmiles("F/C=C/F")
        records = [{"mol": molecule}]

        split_molecules = _molecules_for_split(records, "random")

        self.assertIs(split_molecules[0], molecule)


if __name__ == "__main__":
    unittest.main()
