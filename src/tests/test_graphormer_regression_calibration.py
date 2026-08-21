from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.cli.main import build_parser
from src.cli.predict import (
    apply_calibration_to_predictions,
    fit_calibration_from_frame,
    parse_calibration_pairs,
    predict_graphormer,
)
from src.deep_learning.graphormer.inference.predictor import GraphormerPredictor


class GraphormerRegressionCalibrationTest(unittest.TestCase):
    def _make_predictor(self) -> GraphormerPredictor:
        predictor = object.__new__(GraphormerPredictor)
        predictor.task = "regression"
        predictor.classification_type = None
        predictor.model_config = type("ModelConfig", (), {})()
        predictor.regression_calibration = None
        return predictor

    def test_fit_regression_calibration_uses_validation_residuals(self):
        predictor = self._make_predictor()

        validation_predictions = np.array([0.0, 2.0, 4.0, 6.0], dtype=float)
        validation_targets = np.array([0.1, 2.3, 3.8, 5.9], dtype=float)

        calibration = predictor.fit_regression_calibration(
            validation_predictions,
            validation_targets,
        )

        self.assertGreater(calibration["scale"], 0.0)
        self.assertGreaterEqual(calibration["offset"], 0.0)
        self.assertEqual(calibration["n_samples"], len(validation_predictions))

    def test_regression_format_predictions_keeps_prediction_column_and_adds_uncertainty(self):
        predictor = self._make_predictor()
        calibration = predictor.fit_regression_calibration(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
            np.array([1.2, 2.0, 3.4, 4.3], dtype=float),
        )
        predictor.regression_calibration = calibration

        frame = predictor._format_predictions(torch.tensor([1.0, 2.5, 4.0], dtype=torch.float32))

        self.assertIsInstance(frame, pd.DataFrame)
        self.assertIn("prediction", frame.columns)
        self.assertIn("prediction_std", frame.columns)
        self.assertTrue((frame["prediction_std"] >= 0.0).all())

    def test_parse_calibration_pairs_accepts_target_prediction_pairs(self):
        self.assertEqual(
            parse_calibration_pairs(["target_0:predict_0", "target_1:predict_1"]),
            [("target_0", "predict_0"), ("target_1", "predict_1")],
        )

    def test_pairwise_calibration_uses_single_file_columns(self):
        frame = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN"],
                "target_0": [1.0, 3.0],
                "predict_0": [0.5, 3.5],
            }
        )

        calibration = fit_calibration_from_frame(frame, ["target_0:predict_0"])

        self.assertIn("target_0:predict_0", calibration)
        self.assertAlmostEqual(calibration["target_0:predict_0"]["offset"], 0.0, places=6)
        self.assertAlmostEqual(calibration["target_0:predict_0"]["scale"], 0.7413, places=3)

        adjusted = apply_calibration_to_predictions(frame.copy(), ["target_0:predict_0"])
        self.assertIn("predict_0", adjusted.columns)
        self.assertTrue((adjusted["predict_0"] >= 0.0).all())



class GraphormerPredictCliTest(unittest.TestCase):
    def test_predict_graphormer_cli_parses_and_executes(self):
        parser = build_parser()
        args = parser.parse_args([
            "predict",
            "graphormer",
            "--smiles",
            "CCO",
            "--task-names",
            "activity",
            "--model-checkpoint",
            "/tmp/model.pt",
            "--output",
            "/tmp/out.csv",
            "--calibration-file",
            "/tmp/calibration.csv",
            "--calibration-pairs",
            "target_0:predict_0",
        ])

        self.assertEqual(args.command, "predict")
        self.assertEqual(args.predict_model, "graphormer")
        self.assertEqual(args.smiles, "CCO")
        self.assertEqual(args.task_names, ["activity"])
        self.assertEqual(args.calibration_pairs, ["target_0:predict_0"])

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            calibration_path = Path(temp_dir) / "calibration.csv"
            calibration_path.write_text("target_0,predict_0\n1.0,0.9\n3.0,2.9\n", encoding="utf-8")
            output_path = Path(temp_dir) / "output.csv"

            fake_predictor = object.__new__(GraphormerPredictor)
            fake_predictor.regression_calibration = None

            with patch("src.cli.predict.GraphormerPredictor") as mock_predictor_cls:
                mock_instance = mock_predictor_cls.return_value
                mock_instance.predict_smiles.return_value = pd.DataFrame({"prediction": [1.23]})
                mock_instance.regression_calibration = None

                args.model_checkpoint = str(model_path)
                args.output = str(output_path)
                args.calibration_file = str(calibration_path)

                predict_graphormer(args)

                mock_predictor_cls.assert_called_once_with(
                    checkpoint_path=str(model_path),
                    device=None,
                    threshold=0.5,
                    validation_predictions=None,
                    validation_targets=None,
                )
                mock_instance.predict_smiles.assert_called_once_with(
                    smiles_list=["CCO"],
                    batch_size=64,
                    num_workers=0,
                )

                self.assertTrue(output_path.exists())
                saved = pd.read_csv(output_path)
                self.assertIn("activity", saved.columns)
                self.assertEqual(len(saved), 1)


if __name__ == "__main__":
    unittest.main()
