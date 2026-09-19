import unittest

import numpy as np
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from chemflow.featurization import (
    DESC_NAMES,
    RDKIT2D_DESC_NAMES,
    smiles_to_avalon,
    smiles_to_descriptors,
    smiles_to_erg,
    smiles_to_rdkit2d,
)
from chemflow.machine_learning.data.data_pipeline import (
    CorrelationThreshold,
    featurize_array,
    fitted_feature_dimensions,
    make_scaled_pipeline,
)
from chemflow.machine_learning.train.hyperparameter_tuning import (
    _filter_size_dependent_grid,
    build_cv_splitter,
)
from chemflow.machine_learning.utils.get_model import (
    build_models_config,
    get_model,
)
from chemflow.machine_learning.utils.get_grid_param import (
    get_default_param_grid,
)


class RDKit2DFeaturizationTest(unittest.TestCase):
    def test_expanded_set_starts_with_all_legacy_descriptors(self):
        self.assertEqual(RDKIT2D_DESC_NAMES[: len(DESC_NAMES)], DESC_NAMES)
        self.assertNotIn("Ipc", RDKIT2D_DESC_NAMES)
        self.assertIn("AvgIpc", RDKIT2D_DESC_NAMES)
        self.assertEqual(
            len(RDKIT2D_DESC_NAMES),
            len(Descriptors._descList) - 1,
        )

    def test_legacy_values_are_preserved_in_expanded_vector(self):
        smiles = "CCOc1ccc(C(=O)N)cc1"
        legacy = smiles_to_descriptors(smiles)
        expanded = smiles_to_rdkit2d(smiles)

        self.assertEqual(expanded.shape, (len(RDKIT2D_DESC_NAMES),))
        np.testing.assert_allclose(
            expanded[: len(DESC_NAMES)],
            legacy,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_saved_descriptor_order_can_be_reused_for_prediction(self):
        smiles = np.asarray(["CCO", "c1ccccc1"])
        targets = np.zeros(len(smiles))
        features, _, valid_indices = featurize_array(
            smiles,
            targets,
            feature_types=["rdkit2d"],
            descriptor_names_by_feature={"rdkit2d": list(DESC_NAMES)},
        )

        self.assertEqual(features.shape, (2, len(DESC_NAMES)))
        np.testing.assert_array_equal(valid_indices, np.asarray([0, 1]))
        np.testing.assert_allclose(
            features[0],
            smiles_to_descriptors("CCO"),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_rdkit2d_can_be_combined_with_ecfp4(self):
        features, _, _ = featurize_array(
            np.asarray(["CCO", "CCN"]),
            np.zeros(2),
            feature_types=["rdkit2d", "ecfp4"],
        )
        self.assertEqual(features.shape, (2, len(RDKIT2D_DESC_NAMES) + 2048))

    def test_avalon_and_erg_dimensions(self):
        avalon = smiles_to_avalon("CCOc1ccccc1")
        erg = smiles_to_erg("CCOc1ccccc1")

        self.assertEqual(avalon.shape, (2048,))
        self.assertEqual(erg.shape, (315,))
        self.assertTrue(np.isin(avalon, [0.0, 1.0]).all())
        self.assertTrue(np.isfinite(erg).all())

    def test_all_recommended_representations_can_be_combined(self):
        features, _, _ = featurize_array(
            np.asarray(["CCO", "CCN"]),
            np.zeros(2),
            feature_types=["rdkit2d", "erg", "ecfp4", "avalon"],
        )
        expected = len(RDKIT2D_DESC_NAMES) + 315 + 2048 + 2048
        self.assertEqual(features.shape, (2, expected))

    def test_correlation_filter_removes_later_correlated_column(self):
        values = np.asarray(
            [
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 0.0],
                [3.0, 6.0, 1.0],
                [4.0, 8.0, 0.0],
            ]
        )
        transformer = CorrelationThreshold(threshold=0.95).fit(values)
        np.testing.assert_array_equal(
            transformer.get_support(),
            np.asarray([True, False, True]),
        )
        self.assertEqual(transformer.transform(values).shape, (4, 2))

    def test_model_selection_is_only_added_for_linear_or_svm_models(self):
        tree_pipeline = make_scaled_pipeline(
            RandomForestRegressor(n_estimators=2, random_state=1),
            ["rdkit2d"],
            descriptor_model_selection=True,
            descriptor_selection_estimators=4,
        )
        svm_pipeline = make_scaled_pipeline(
            SVR(),
            ["rdkit2d"],
            descriptor_model_selection=True,
            descriptor_selection_estimators=4,
        )

        tree_steps = tree_pipeline.named_steps[
            "descriptor_preprocessor"
        ].named_steps
        svm_steps = svm_pipeline.named_steps[
            "descriptor_preprocessor"
        ].named_steps
        self.assertNotIn("model_selector", tree_steps)
        self.assertIn("model_selector", svm_steps)

    def test_fitted_pipeline_reports_final_dimensions(self):
        smiles = np.asarray(
            [
                "CCO",
                "CCN",
                "CCC",
                "CCCC",
                "CCCl",
                "CCBr",
                "CC(=O)O",
                "CCOC",
                "CCS",
                "c1ccccc1",
                "c1ccncc1",
                "CC(C)O",
            ]
        )
        targets = np.linspace(0.1, 1.2, len(smiles))
        features, targets, _ = featurize_array(
            smiles,
            targets,
            feature_types=["rdkit2d", "erg", "ecfp4", "avalon"],
        )
        pipeline = make_scaled_pipeline(
            RandomForestRegressor(n_estimators=2, random_state=1),
            ["rdkit2d", "erg", "ecfp4", "avalon"],
        )
        pipeline.fit(features, targets)
        summary = fitted_feature_dimensions(pipeline, features.shape[1])

        self.assertEqual(summary["raw_total"], 4620)
        self.assertEqual(summary["descriptor_raw"], len(RDKIT2D_DESC_NAMES))
        self.assertEqual(summary["fingerprint_raw"], 4411)
        self.assertLess(summary["descriptor_after_correlation"], 209)
        self.assertEqual(
            summary["final_model_input"],
            pipeline.named_steps["model"].n_features_in_,
        )

    def test_scaffold_grouped_cv_never_splits_a_scaffold(self):
        smiles = np.asarray(
            [
                "c1ccccc1",
                "Cc1ccccc1",
                "c1ccncc1",
                "Cc1ccncc1",
                "C1CCCCC1",
                "CC1CCCCC1",
                "c1ccsc1",
                "Cc1ccsc1",
            ]
        )
        splitter, groups, strategy = build_cv_splitter(
            {"cv": 4, "cv_strategy": "scaffold_grouped"},
            "regression",
            train_smiles=smiles,
        )

        self.assertEqual(strategy, "scaffold-grouped")
        self.assertEqual(len(np.unique(groups)), 4)
        for train_indices, validation_indices in splitter.split(
            np.zeros((len(smiles), 1)),
            groups=groups,
        ):
            self.assertTrue(
                set(groups[train_indices]).isdisjoint(groups[validation_indices])
            )

    def test_standard_cv_uses_no_groups(self):
        splitter, groups, strategy = build_cv_splitter(
            {"cv": 3, "cv_strategy": "cv", "cv_random_seed": 7},
            "regression",
        )
        self.assertEqual(strategy, "cv")
        self.assertIsNone(groups)
        self.assertEqual(splitter.n_splits, 3)

    def test_lightgbm_grid_activates_row_subsampling(self):
        grid = get_default_param_grid("LightGBM", "regression")
        self.assertEqual(grid["subsample_freq"], [1])
        self.assertTrue(all(depth > 0 for depth in grid["max_depth"]))
        self.assertTrue(all(value > 0 for value in grid["reg_lambda"]))

    def test_every_default_grid_matches_its_estimator(self):
        models_by_task = {
            "classification": [
                "Random Forest",
                "Extra Trees",
                "Gradient Boosting",
                "XGBoost",
                "LightGBM",
                "SVM_RBF",
                "KNN",
                "MLP",
                "Logistic Regression",
            ],
            "regression": [
                "Random Forest",
                "Extra Trees",
                "Gradient Boosting",
                "XGBoost",
                "LightGBM",
                "SVM_RBF",
                "KNN",
                "MLP",
                "Ridge Regression",
                "Lasso Regression",
                "PLS",
            ],
        }
        for task_type, model_names in models_by_task.items():
            for model_name in model_names:
                with self.subTest(task_type=task_type, model_name=model_name):
                    model = get_model(model_name, task_type, tune_hyperparameter=True)
                    grid = get_default_param_grid(model_name, task_type)
                    self.assertTrue(set(grid).issubset(model.get_params()))

    def test_size_dependent_candidates_are_bounded_by_cv_fold(self):
        knn_grid = _filter_size_dependent_grid(
            {"n_neighbors": [3, 11, 31]},
            "KNN",
            min_fold_train_size=12,
            n_features=100,
        )
        pls_grid = _filter_size_dependent_grid(
            {"n_components": [2, 5, 10, 20]},
            "PLS",
            min_fold_train_size=9,
            n_features=6,
        )
        self.assertEqual(knn_grid["n_neighbors"], [3, 11])
        self.assertEqual(pls_grid["n_components"], [2, 5])

    def test_lightgbm_is_available_in_both_model_catalogs(self):
        for task_type in ("classification", "regression"):
            models, skipped = build_models_config(
                ["LightGBM"],
                task_type=task_type,
                seeds=[42],
                hyperparameter_tuning=True,
            )
            self.assertFalse(skipped)
            self.assertIn("LightGBM", models)
            self.assertEqual(models["LightGBM"]["cv_strategy"], "cv")


if __name__ == "__main__":
    unittest.main()
