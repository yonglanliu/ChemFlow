from __future__ import annotations

import json

import numpy as np
import pandas as pd

from chemflow.deep_learning.test_evaluation import (
    fingerprint_local_predictions,
    save_test_evaluation,
)
from chemflow.deep_learning.uncertainty_report import generate_uncertainty_report


def test_fingerprint_calibration_is_local_and_flags_ood():
    predictions = fingerprint_local_predictions(
        train_smiles=["CC", "CCC", "c1ccccc1"],
        validation_smiles=["CCO", "CCN", "c1ccccc1O"],
        validation_truths=np.asarray([[2.0], [3.0], [4.0]]),
        validation_predictions=np.asarray([[1.5], [2.5], [3.5]]),
        test_smiles=["CCCO", "[Na+].[Cl-]"],
        direct_predictions=np.asarray([[1.0], [2.0]]),
        mc_predictions=np.asarray([[1.2], [2.2]]),
        mc_standard_deviations=np.asarray([[0.1], [0.2]]),
        targets=["CL"],
        confidence=0.90,
    )

    np.testing.assert_allclose(
        predictions["calibrated_CL_prediction"], [1.7, 2.7]
    )
    assert predictions["CL_local_calibration_used"].all()
    assert predictions["CL_local_calibration_count"].min() > 0
    assert predictions["CL_ood_score"].shape == (2,)
    assert predictions["CL_ood_flag"].dtype == np.bool_


def test_standard_test_outputs_include_all_prediction_variants(tmp_path):
    predictions = {
        "direct_CL_prediction": np.asarray([1.0, 2.0, 3.0]),
        "mc_mean_CL_prediction": np.asarray([1.1, 2.1, 3.1]),
        "mc_std_CL_prediction": np.asarray([0.1, 0.2, 0.3]),
        "calibrated_CL_prediction": np.asarray([1.2, 2.2, 3.2]),
        "CL_prediction_lower_90": np.asarray([0.8, 1.8, 2.8]),
        "CL_prediction_upper_90": np.asarray([1.6, 2.6, 3.6]),
        "CL_ood_flag": np.asarray([False, False, True]),
        "CL_ood_score": np.asarray([0.1, 0.2, 0.99]),
        "CL_local_calibration_used": np.asarray([True, True, True]),
        "CL_local_calibration_weak": np.asarray([False, False, True]),
        "CL_prediction_uncertainty_flag": np.asarray([False, True, True]),
    }
    frame, summary = save_test_evaluation(
        workdir=tmp_path,
        smiles=["CC", "CCC", "CCCC"],
        truths=np.asarray([[1.1], [2.0], [3.5]]),
        targets=["CL"],
        task_types=["regression"],
        predictions=predictions,
        summary={"backend": "test"},
        confidence=0.90,
    )

    required = {
        "CL_direct_prediction",
        "CL_mc_prediction",
        "CL_mc_sd",
        "CL_calibrated_prediction",
        "CL_calibrated_uncertainty",
        "CL_ood_flag",
        "CL_uncertainty_flag",
        "CL_interval_covered",
    }
    assert required.issubset(frame.columns)
    assert set(summary["test_metrics"]) == {
        "direct_prediction",
        "mc_dropout_prediction",
        "locally_calibrated_prediction",
    }
    assert summary["test_calibration"]["source"] == "validation_only"
    assert summary["test_calibration"]["global_fallback"] is False
    assert summary["test_uncertainty_diagnostics"]["CL"]["ood_count"] == 1
    toolbox = summary["test_uncertainty_diagnostics"]["CL"][
        "uncertainty_toolbox"
    ]
    assert toolbox["package"] == "uncertainty-toolbox"
    assert toolbox["available"] is True
    assert "rms_cal" in toolbox["average_calibration"]
    assert "crps" in toolbox["proper_scoring_rules"]
    assert pd.read_csv(tmp_path / "test_predictions.csv").shape[0] == 3
    assert json.loads((tmp_path / "metrics.json").read_text())["backend"] == "test"


def test_uncertainty_report_saves_plots_and_csv_tables(tmp_path):
    rng = np.random.default_rng(7)
    size = 12
    truth = np.linspace(-1.0, 1.0, size)
    mc_prediction = truth + rng.normal(0.0, 0.15, size)
    mc_sd = np.linspace(0.1, 0.3, size)
    frame = pd.DataFrame(
        {
            "SMILES": ["C" * (index % 4 + 1) for index in range(size)],
            "CL_true": truth,
            "CL_direct_prediction": mc_prediction - 0.01,
            "CL_mc_prediction": mc_prediction,
            "CL_mc_sd": mc_sd,
            "CL_calibrated_prediction": mc_prediction + 0.02,
            "CL_calibrated_uncertainty": np.maximum(1.645 * mc_sd, 0.25),
            "CL_ood_score": np.linspace(0.0, 1.0, size),
            "CL_ood_flag": np.arange(size) >= 10,
            "CL_uncertainty_flag": np.arange(size) >= 8,
            "CL_interval_covered": np.ones(size),
        }
    )

    manifest = generate_uncertainty_report(
        workdir=tmp_path,
        predictions=frame,
        targets=["CL"],
        task_types=["regression"],
        confidence=0.90,
    )

    output = tmp_path / "uncertainty"
    assert manifest["targets_plotted"] == ["CL"]
    assert (output / "CL/01_confidence_band.png").is_file()
    assert (output / "CL/10_uncertainty_overview.pdf").is_file()
    confidence_data = pd.read_csv(
        output / "CL/confidence_band_sorted_by_prediction.csv"
    )
    ordered_data = pd.read_csv(
        output / "CL/ordered_intervals_sorted_by_observed.csv"
    )
    assert confidence_data["mc_prediction"].is_monotonic_increasing
    assert ordered_data["observed_value"].is_monotonic_increasing
    comparison = pd.read_csv(output / "calibration_metrics_comparison.csv")
    assert comparison["Stage"].tolist() == ["MC dropout", "Locally calibrated"]
    assert {"MACE", "RMSCE", "Miscalibration_area"}.issubset(comparison)
