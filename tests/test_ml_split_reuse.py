from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from chemflow.machine_learning.train.train_runner import (
    load_or_create_split_features,
)
from chemflow.machine_learning.train.utils import (
    build_split_config,
    write_test_predictions,
)


class MLSplitReuseTest(unittest.TestCase):
    @staticmethod
    def _write_source(path: Path, include_split=False) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "Molecule Name": ["Mol1", "Mol2", "Mol3", "Mol4", "Mol5", "Mol6"],
                "SMILES": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCS"],
                "target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "other_target": [10, 20, 30, 40, 50, 60],
            }
        )
        if include_split:
            frame["split"] = ["train", "train", "train", "val", "test", "test"]
        frame.to_csv(path, index=False)
        return frame

    def test_predefined_column_is_used_without_repartitioning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            self._write_source(source, include_split=True)
            result = load_or_create_split_features(
                training_data_file=source,
                smiles_col="SMILES",
                y_col="target",
                feature_types=["descriptor"],
                split_config={
                    "split_method": "predefined",
                    "split_column": "split",
                    "save_split_data": True,
                    "save_dir": str(root / "splits"),
                    "split_name": "provided",
                },
                split_npz_path=None,
                job_dir=root / "job",
                require_train_smiles=True,
            )

            X_train, y_train, X_test, y_test, X_valid, y_valid, source_name, smiles = result
            self.assertEqual(source_name, "pre_split_column")
            self.assertEqual(X_train.shape[0], 3)
            self.assertEqual(X_valid.shape[0], 1)
            self.assertEqual(X_test.shape[0], 2)
            np.testing.assert_allclose(y_train, [0.1, 0.2, 0.3])
            np.testing.assert_allclose(y_valid, [0.4])
            np.testing.assert_allclose(y_test, [0.5, 0.6])
            np.testing.assert_array_equal(smiles, ["CCO", "CCN", "CCC"])
            exported = pd.read_csv(root / "splits" / "provided_split_data.csv")
            self.assertEqual(
                list(exported.columns),
                ["Molecule Name", "SMILES", "target", "split"],
            )
            self.assertNotIn("other_target", exported.columns)
            self.assertEqual(
                exported["split"].tolist(),
                ["train", "train", "train", "validation", "test", "test"],
            )

    def test_predefined_and_legacy_names_build_the_same_config(self):
        for method in ("predefined", "splitted"):
            config = build_split_config(
                {
                    "split_method": method,
                    "split_column": "partition",
                    "save_split_data": False,
                }
            )
            self.assertEqual(config["split_method"], "predefined")
            self.assertEqual(config["split_column"], "partition")
            self.assertFalse(config["save_split_data"])

        inferred = build_split_config({"split_column": "partition"})
        self.assertEqual(inferred["split_method"], "predefined")

    def test_legacy_cache_reuses_indices_and_upgrades_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            self._write_source(source)
            cache = root / "legacy_split.npz"

            legacy_train = np.empty(3, dtype=object)
            legacy_test = np.empty(2, dtype=object)
            legacy_valid = np.empty(1, dtype=object)
            for array in (legacy_train, legacy_test, legacy_valid):
                for index in range(len(array)):
                    array[index] = np.asarray([float(index)])
            np.savez_compressed(
                cache,
                X_train=legacy_train,
                X_test=legacy_test,
                X_valid=legacy_valid,
                y_train=np.asarray([0.1, 0.2, 0.3]),
                y_test=np.asarray([0.5, 0.6]),
                y_valid=np.asarray([0.4]),
                train_indices=np.asarray([0, 1, 2]),
                valid_indices=np.asarray([3]),
                test_indices=np.asarray([4, 5]),
            )

            result = load_or_create_split_features(
                training_data_file=source,
                smiles_col="SMILES",
                y_col="target",
                feature_types=["descriptor"],
                split_config={"split_method": "scaffold", "save_split_data": True},
                split_npz_path=cache,
                job_dir=root / "job",
                require_train_smiles=True,
            )

            self.assertEqual(result[0].shape[0], 3)
            self.assertEqual(result[2].shape[0], 2)
            self.assertEqual(result[4].shape[0], 1)
            np.testing.assert_array_equal(result[7], ["CCO", "CCN", "CCC"])

            with np.load(cache, allow_pickle=False) as upgraded:
                self.assertEqual(upgraded["X_train"].dtype, np.float32)
                self.assertEqual(upgraded["train_smiles"].dtype.kind, "U")
                np.testing.assert_array_equal(upgraded["train_indices"], [0, 1, 2])
                np.testing.assert_array_equal(upgraded["feature_types"], ["descriptor"])

            exported = pd.read_csv(root / "legacy_split_split_data.csv")
            self.assertEqual(
                exported["split"].tolist(),
                ["train", "train", "train", "validation", "test", "test"],
            )

    def test_changed_features_rebuild_cache_without_changing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            self._write_source(source)
            cache = root / "feature_cache.npz"
            np.savez_compressed(
                cache,
                X_train=np.zeros((3, 9), dtype=np.float32),
                X_test=np.zeros((2, 9), dtype=np.float32),
                X_valid=np.zeros((1, 9), dtype=np.float32),
                y_train=np.asarray([0.1, 0.2, 0.3]),
                y_test=np.asarray([0.5, 0.6]),
                y_valid=np.asarray([0.4]),
                train_indices=np.asarray([0, 1, 2]),
                valid_indices=np.asarray([3]),
                test_indices=np.asarray([4, 5]),
                feature_types=np.asarray(["descriptor"]),
            )

            result = load_or_create_split_features(
                training_data_file=source,
                smiles_col="SMILES",
                y_col="target",
                feature_types=["descriptor", "erg"],
                split_config={"split_method": "scaffold", "save_split_data": True},
                split_npz_path=cache,
                job_dir=root / "job",
            )

            self.assertEqual(result[0].shape, (3, 324))
            self.assertEqual(result[2].shape, (2, 324))
            with np.load(cache, allow_pickle=False) as rebuilt:
                np.testing.assert_array_equal(
                    rebuilt["feature_types"], ["descriptor", "erg"]
                )
                np.testing.assert_array_equal(rebuilt["train_indices"], [0, 1, 2])
                np.testing.assert_array_equal(rebuilt["valid_indices"], [3])
                np.testing.assert_array_equal(rebuilt["test_indices"], [4, 5])

    def test_test_predictions_csv_contains_truth_prediction_and_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = write_test_predictions(
                output_dir=directory,
                model_name="LightGBM",
                task_type="regression",
                evaluation_results={
                    "y_test": [1.0, 2.0],
                    "y_pred": [0.75, 2.5],
                },
                test_metadata={
                    "Molecule Name": ["MolA", "MolB"],
                    "SMILES": ["CCO", "CCN"],
                },
            )
            frame = pd.read_csv(output)
            self.assertEqual(
                list(frame.columns),
                [
                    "Molecule Name",
                    "SMILES",
                    "test_row",
                    "true_value",
                    "predicted_value",
                    "residual",
                    "absolute_error",
                ],
            )
            np.testing.assert_allclose(frame["residual"], [0.25, -0.5])
            np.testing.assert_allclose(frame["absolute_error"], [0.25, 0.5])


if __name__ == "__main__":
    unittest.main()
