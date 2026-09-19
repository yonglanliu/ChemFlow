from __future__ import annotations

import unittest

import numpy as np
import torch

from chemflow.deep_learning.chemeleon.applicability import (
    applicability_diagnostics,
    apply_calibration,
    apply_local_calibration,
    fit_validation_calibration,
    fit_ood_calibration,
    packed_morgan_fingerprints,
)


class CheMeleonApplicabilityTest(unittest.TestCase):
    def test_local_calibration_overrides_global_values_only_when_supported(self):
        output = {
            "clearance": np.asarray([1.0, 2.0], dtype=np.float32),
            "calibrated_clearance": np.asarray([1.1, 2.1], dtype=np.float32),
            "clearance_lower_90": np.asarray([0.5, 1.5], dtype=np.float32),
            "clearance_upper_90": np.asarray([1.7, 2.7], dtype=np.float32),
            "clearance_local_calibration_used": np.asarray([True, False]),
            "clearance_local_calibration_bias": np.asarray([0.2, np.nan]),
            "clearance_local_uncertainty_radius": np.asarray([0.3, np.nan]),
        }
        apply_local_calibration(
            output,
            original_names=["clearance"],
            output_names=["clearance"],
            confidence=0.90,
        )
        np.testing.assert_allclose(output["calibrated_clearance"], [1.2, 2.1])
        np.testing.assert_allclose(output["clearance_lower_90"], [0.9, 1.5])
        np.testing.assert_allclose(output["clearance_upper_90"], [1.5, 2.7])

    def test_ood_calibration_uses_validation_domain_distribution(self):
        fitted = fit_ood_calibration(
            similarities=np.asarray([0.9, 0.8, 0.7, 0.6]),
            embedding_distances=np.asarray([0.1, 0.2, 0.3, 0.4]),
            confidence=0.75,
        )
        self.assertEqual(fitted["n_samples"], 4)
        self.assertAlmostEqual(fitted["fp_similarity_threshold"], 0.6)
        self.assertAlmostEqual(fitted["embedding_distance_threshold"], 0.4)

    def test_regression_calibration_adds_bias_and_conformal_interval(self):
        calibration = fit_validation_calibration(
            "regression",
            targets=np.asarray([[2.0], [4.0], [6.0]]),
            predictions=np.asarray([[1.0], [3.0], [5.0]]),
            target_names=["clearance"],
            confidence=0.9,
        )
        output = {"clearance": np.asarray([7.0, np.nan], dtype=np.float32)}
        apply_calibration(
            output,
            calibration,
            original_names=["clearance"],
            output_names=["clearance"],
            task="regression",
            threshold=0.5,
        )
        self.assertAlmostEqual(output["calibrated_clearance"][0], 8.0)
        self.assertAlmostEqual(output["clearance_lower_90"][0], 8.0)
        self.assertAlmostEqual(output["clearance_upper_90"][0], 8.0)
        self.assertTrue(np.isnan(output["calibrated_clearance"][1]))

    def test_diagnostics_find_exact_training_neighbors(self):
        payload = {
            "projection": {
                "mean": torch.zeros(2),
                "components": torch.eye(2),
            },
            "similarity": {"radius": 2, "bits": 256},
            "training": {
                "smiles": ["CCO", "c1ccccc1"],
                "embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "fingerprints": torch.from_numpy(
                    packed_morgan_fingerprints(
                        ["CCO", "c1ccccc1"], radius=2, bits=256
                    )
                ),
            },
        }
        diagnostics = applicability_diagnostics(
            ["CCO", "not-a-smiles"],
            embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            valid_indices=[0],
            payload=payload,
        )
        self.assertAlmostEqual(diagnostics["train_max_tanimoto"][0], 1.0)
        self.assertEqual(
            diagnostics["train_nearest_tanimoto_smiles"][0], "CCO"
        )
        self.assertAlmostEqual(
            diagnostics["train_embedding_cosine_distance"][0], 0.0
        )
        self.assertTrue(
            np.isnan(diagnostics["train_embedding_cosine_distance"][1])
        )


if __name__ == "__main__":
    unittest.main()
