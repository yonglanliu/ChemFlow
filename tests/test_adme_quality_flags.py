from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from chemflow.desktop.adme_page import (
    add_prediction_quality_flags,
    count_valid_log_predictions,
    export_prediction_csv,
    parse_named_smiles,
    simplify_adme_results,
)


class ADMEQualityFlagsTest(unittest.TestCase):
    def test_counts_valid_predictions_before_display_columns_are_simplified(self):
        raw = pd.DataFrame(
            {
                "Log_HLM_CLint_prediction": [1.0, 2.0, float("nan")],
                "Log_MLM_CLint_prediction": [1.5, float("nan"), 3.0],
            }
        )

        self.assertEqual(count_valid_log_predictions(raw, ("HLM", "MLM")), 1)
        self.assertEqual(count_valid_log_predictions(raw, ("RLM",)), 0)

    def test_named_smiles_are_preserved_as_identifiers(self):
        frame = parse_named_smiles(
            "CCO ethanol\nCC(=O)Oc1ccccc1C(=O)O aspirin tablet\nCCC"
        )

        self.assertEqual(frame.columns.tolist(), ["Molecule Name", "SMILES"])
        self.assertEqual(
            frame["Molecule Name"].tolist(),
            ["ethanol", "aspirin tablet", "row_3"],
        )
        self.assertEqual(frame["SMILES"].tolist()[0], "CCO")

    def test_plain_smiles_input_does_not_add_an_identifier_column(self):
        frame = parse_named_smiles("CCO\nCCC")
        self.assertEqual(frame.columns.tolist(), ["SMILES"])

    def test_exports_simplified_results_with_csv_extension(self):
        frame = pd.DataFrame({"SMILES": ["CC"], "HLM_pred_raw": [12.5]})
        with tempfile.TemporaryDirectory() as directory:
            output = export_prediction_csv(frame, Path(directory) / "predictions")
            loaded = pd.read_csv(output)

        self.assertEqual(output.name, "predictions.csv")
        pd.testing.assert_frame_equal(loaded, frame)

    def test_flags_ood_and_wide_calibrated_interval(self):
        frame = pd.DataFrame(
            {
                "SMILES": ["CC", "CCC"],
                "Log_HLM_CLint_lower_90": [0.0, 0.0],
                "Log_HLM_CLint_upper_90": [0.4, 1.5],
                "HLM_train_max_tanimoto": [0.8, 0.2],
                "HLM_train_embedding_cosine_distance": [0.1, 0.6],
            }
        )

        result = add_prediction_quality_flags(
            frame,
            ("HLM",),
            min_tanimoto_similarity=0.35,
            max_embedding_distance=0.35,
            max_log_interval_width=1.0,
        )

        self.assertEqual(result["quality_flag"].tolist(), ["PASS", "OOD + HIGH UNCERTAINTY"])
        self.assertEqual(result["review_required"].tolist(), [False, True])
        self.assertEqual(result["HLM_uncertainty_log_width"].tolist(), [0.4, 1.5])

    def test_missing_checkpoint_diagnostics_are_not_reported_as_pass(self):
        result = add_prediction_quality_flags(
            pd.DataFrame({"SMILES": ["CC"]}),
            ("MLM",),
            min_tanimoto_similarity=0.35,
            max_embedding_distance=0.35,
            max_log_interval_width=1.0,
        )

        self.assertEqual(result.loc[0, "quality_flag"], "NOT ASSESSED")
        self.assertTrue(result.loc[0, "review_required"])
        self.assertFalse(result.loc[0, "domain_assessed"])
        self.assertFalse(result.loc[0, "uncertainty_assessed"])

    def test_simplifies_columns_and_uses_best_domain_values(self):
        frame = pd.DataFrame(
            {
                "SMILES": ["CC"],
                "HLM_CLint_prediction (mL/min/kg)": [10.0],
                "HLM_CLint_calibrated (mL/min/kg)": [12.0],
                "HLM_CLint_lower_90 (mL/min/kg)": [3.0],
                "HLM_CLint_upper_90 (mL/min/kg)": [48.0],
                "HLM_uncertainty_log_width": [1.2],
                "MLM_CLint_prediction (mL/min/kg)": [7.0],
                "MLM_CLint_calibrated (mL/min/kg)": [8.0],
                "MLM_CLint_lower_90 (mL/min/kg)": [2.0],
                "MLM_CLint_upper_90 (mL/min/kg)": [32.0],
                "MLM_uncertainty_log_width": [1.1],
                "HLM_train_max_tanimoto": [0.7],
                "MLM_train_max_tanimoto": [0.4],
                "HLM_train_embedding_cosine_distance": [0.2],
                "MLM_train_embedding_cosine_distance": [0.5],
                "HLM_ood_score": [0.2],
                "MLM_ood_score": [0.8],
                "quality_flag": ["OOD"],
                "is_ood": [True],
                "high_uncertainty": [False],
                "MLM_train_nearest_embedding_smiles": ["CCC"],
            }
        )

        result = simplify_adme_results(frame, ["SMILES"], ("HLM", "MLM"))

        self.assertEqual(
            result.columns.tolist(),
            [
                "SMILES",
                "HLM_pred_raw",
                "HLM_pred_calibrated",
                "HLM_low",
                "HLM_high",
                "HLM_uncertainty_log",
                "MLM_pred_raw",
                "MLM_pred_calibrated",
                "MLM_low",
                "MLM_high",
                "MLM_uncertainty_log",
                "HLM_FP_sim",
                "HLM_EB_sim",
                "HLM_OOD_score",
                "MLM_FP_sim",
                "MLM_EB_sim",
                "MLM_OOD_score",
                "FP_sim_best",
                "FP_sim_worst",
                "EB_sim_best",
                "EB_sim_worst",
                "OOD_score_best",
                "OOD_score_worst",
                "Flag",
                "OOD",
                "High_uncertainty",
            ],
        )
        self.assertEqual(result.loc[0, "FP_sim_best"], 0.7)
        self.assertEqual(result.loc[0, "FP_sim_worst"], 0.4)
        self.assertEqual(result.loc[0, "HLM_EB_sim"], 0.8)
        self.assertEqual(result.loc[0, "MLM_EB_sim"], 0.5)
        self.assertEqual(result.loc[0, "EB_sim_best"], 0.8)
        self.assertEqual(result.loc[0, "EB_sim_worst"], 0.5)
        self.assertEqual(result.loc[0, "OOD_score_best"], 0.2)
        self.assertEqual(result.loc[0, "OOD_score_worst"], 0.8)
        self.assertNotIn("MLM_train_nearest_embedding_smiles", result)

    def test_combined_domain_flag_uses_best_model(self):
        frame = pd.DataFrame(
            {
                "HLM_train_max_tanimoto": [0.2, 0.2],
                "HLM_train_embedding_cosine_distance": [0.6, 0.6],
                "MLM_train_max_tanimoto": [0.8, 0.3],
                "MLM_train_embedding_cosine_distance": [0.1, 0.5],
            }
        )

        result = add_prediction_quality_flags(
            frame,
            ("HLM", "MLM"),
            min_tanimoto_similarity=0.4,
            max_embedding_distance=0.35,
            max_log_interval_width=1.0,
        )

        self.assertEqual(result["is_ood"].tolist(), [False, True])

    def test_validation_calibrated_ood_score_takes_priority(self):
        frame = pd.DataFrame(
            {
                "HLM_train_max_tanimoto": [0.1, 0.9],
                "HLM_train_embedding_cosine_distance": [0.9, 0.1],
                "HLM_ood_score": [0.40, 0.98],
            }
        )
        result = add_prediction_quality_flags(
            frame,
            ("HLM",),
            min_tanimoto_similarity=0.35,
            max_embedding_distance=0.35,
            max_log_interval_width=1.0,
            ood_score_threshold=0.95,
        )
        self.assertEqual(result["is_ood"].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
