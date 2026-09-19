"""Tests for the combined CheMeleon clearance deployment wrapper."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from torch import nn

from chemflow.deep_learning.chemeleon.adme_model import ADMEModel


class _FakePredictor:
    def __init__(self, checkpoint_path, device=None, threshold=0.5, **_kwargs):
        del device, threshold
        self.task = "regression"
        self.model = nn.Identity()
        if "multi" in str(checkpoint_path):
            self.target_names = ["Log_HLM_CLint", "Log_RLM_CLint"]
        else:
            self.target_names = ["Log_MLM_CLint"]

    def predict_smiles(self, smiles, batch_size=64, num_workers=0):
        del batch_size, num_workers
        count = len(smiles)
        if len(self.target_names) == 2:
            return {
                "Log_HLM_CLint": np.full(count, 1.0),
                "Log_RLM_CLint": np.full(count, 2.0),
            }
        return {"Log_MLM_CLint": np.full(count, 3.0)}


class ADMEModelTests(unittest.TestCase):
    @patch(
        "chemflow.deep_learning.chemeleon.adme_model.CheMeleonPredictor",
        _FakePredictor,
    )
    def test_combines_models_and_inverts_log10_predictions(self):
        model = ADMEModel("multi.ckpt", "mlm.ckpt", device="cpu")
        frame = model.predict_frame(["CCO", "CCN"], batch_size=2)

        self.assertEqual(frame["SMILES"].tolist(), ["CCO", "CCN"])
        np.testing.assert_allclose(frame["Log_HLM_CLint_prediction"], 1.0)
        np.testing.assert_allclose(frame["Log_RLM_CLint_prediction"], 2.0)
        np.testing.assert_allclose(frame["Log_MLM_CLint_prediction"], 3.0)
        np.testing.assert_allclose(
            frame["HLM_CLint_prediction (mL/min/kg)"], 10.0
        )
        np.testing.assert_allclose(
            frame["RLM_CLint_prediction (mL/min/kg)"], 100.0
        )
        np.testing.assert_allclose(
            frame["MLM_CLint_prediction (mL/min/kg)"], 1000.0
        )

    @patch(
        "chemflow.deep_learning.chemeleon.adme_model.CheMeleonPredictor",
        _FakePredictor,
    )
    def test_can_deploy_only_the_selected_single_task_model(self):
        model = ADMEModel(None, "mlm.ckpt", device="cpu")
        frame = model.predict_frame(["CCO"], properties=["MLM"])

        self.assertEqual(
            frame.columns.tolist(),
            [
                "SMILES",
                "Log_MLM_CLint_prediction",
                "MLM_CLint_prediction (mL/min/kg)",
            ],
        )
        np.testing.assert_allclose(frame["Log_MLM_CLint_prediction"], 3.0)


if __name__ == "__main__":
    unittest.main()
