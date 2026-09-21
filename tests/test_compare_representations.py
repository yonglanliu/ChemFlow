from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from example.ml_training.compare_representations import (
    METRICS,
    REPRESENTATIONS,
    add_holm_adjustment,
    align_representations,
    plot_combined_summary,
    plot_pairwise_heatmaps,
    summarize_paired_differences,
)


class RepresentationComparisonTest(unittest.TestCase):
    def test_alignment_does_not_combine_stable_smiles_with_changed_row_ids(self):
        frames = {
            "a": pd.DataFrame(
                {
                    "SMILES": ["CC", "CCC"],
                    "test_row": [1, 2],
                    "true_value": [1.0, 2.0],
                    "predicted_value": [1.1, 2.1],
                }
            ),
            "b": pd.DataFrame(
                {
                    "SMILES": ["CC", "CCC"],
                    "test_row": [101, 102],
                    "true_value": [1.0, 2.0],
                    "predicted_value": [0.9, 1.9],
                }
            ),
        }

        aligned = align_representations(frames)

        self.assertEqual(list(aligned["SMILES"]), ["CC", "CCC"])
        self.assertEqual(len(aligned), 2)

    def test_alignment_rejects_reused_names_with_different_targets(self):
        frames = {
            "a": pd.DataFrame(
                {
                    "Molecule Name": ["0", "1"],
                    "SMILES": ["CC", "CCC"],
                    "true_value": [1.0, 2.0],
                    "predicted_value": [1.1, 2.1],
                }
            ),
            "b": pd.DataFrame(
                {
                    "Molecule Name": ["0", "1"],
                    "SMILES": ["CCC", "CC"],
                    "true_value": [2.0, 1.0],
                    "predicted_value": [1.9, 0.9],
                }
            ),
        }

        aligned = align_representations(frames)

        self.assertEqual(list(aligned["SMILES"]), ["CC", "CCC"])
        self.assertEqual(list(aligned["true__b"]), [1.0, 2.0])

    def test_paired_improvement_respects_metric_direction(self):
        labels = OrderedDict((("a", "A"), ("b", "B")))
        points = {
            "a": {metric: (1.0 if metric == "RMSE" else 0.8) for metric in METRICS},
            "b": {metric: (2.0 if metric == "RMSE" else 0.5) for metric in METRICS},
        }
        bootstraps = {
            "a": {
                metric: ([1.0, 1.1, 0.9] if metric == "RMSE" else [0.8, 0.9, 0.7])
                for metric in METRICS
            },
            "b": {
                metric: ([2.0, 2.1, 1.9] if metric == "RMSE" else [0.5, 0.6, 0.4])
                for metric in METRICS
            },
        }
        result = add_holm_adjustment(
            summarize_paired_differences("HLM", "LightGBM", labels, points, bootstraps)
        ).set_index("Metric")

        self.assertGreater(result.loc["RMSE", "Improvement_A_over_B"], 0)
        self.assertGreater(result.loc["R2", "Improvement_A_over_B"], 0)
        self.assertEqual(result.loc["RMSE", "Probability_A_better"], 1.0)
        self.assertIn("P_value_Holm", result.columns)

    def test_combined_figure_writes_png_and_pdf(self):
        rows = []
        for metric in METRICS:
            for representation, estimate in (("ECFP4", 0.5), ("RDKit2D", 0.6)):
                rows.append(
                    {
                        "Task": "HLM",
                        "Representation": representation,
                        "Metric": metric,
                        "Estimate": estimate,
                        "CI_2.5%": estimate - 0.1,
                        "CI_97.5%": estimate + 0.1,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "combined.png"
            plot_combined_summary(
                pd.DataFrame(rows), ["HLM"], "LightGBM", output
            )
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".pdf").is_file())

    def test_pairwise_heatmap_writes_png_and_pdf(self):
        points = {
            key: {metric: float(index) for metric in METRICS}
            for index, key in enumerate(REPRESENTATIONS)
        }
        bootstraps = {
            key: {
                metric: [float(index), float(index) + 0.1, float(index) - 0.1]
                for metric in METRICS
            }
            for index, key in enumerate(REPRESENTATIONS)
        }
        pairwise = add_holm_adjustment(
            summarize_paired_differences(
                "HLM", "LightGBM", REPRESENTATIONS, points, bootstraps
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pairwise.png"
            plot_pairwise_heatmaps(
                pairwise, ["HLM"], "RMSE", "LightGBM", output
            )
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".pdf").is_file())


if __name__ == "__main__":
    unittest.main()
