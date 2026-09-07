from __future__ import annotations

import pickle

from pathlib import Path

from typing import (
    Any,
    Dict,
    Sequence,
)

import numpy as np
import pandas as pd

from src.chemflow.machine_learning.data.data_pipeline import (
    featurize_array,
)


# ============================================================
# Load model
# ============================================================

def load_pickle_model(
    model_path: str | Path,
) -> Dict[str, Any]:

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: "
            f"{model_path}"
        )

    with open(
        model_path,
        "rb",
    ) as f:

        obj = pickle.load(
            f
        )

    # --------------------------------------------------------
    # ChemFlow model package
    # --------------------------------------------------------

    if (
        isinstance(obj, dict)
        and
        "model" in obj
    ):

        return {

            "model":
                obj.get("model"),

            "model_name":
                obj.get("model_name"),

            "task_type":
                obj.get("task_type"),

            "feature_config":
                obj.get("feature_config"),

            "training_config":
                obj.get("training_config"),

            "metrics":
                obj.get("metrics"),

            "chemflow_package":
                True,
        }

    # --------------------------------------------------------
    # Raw estimator / pipeline
    # --------------------------------------------------------

    return {

        "model":
            obj,

        "model_name":
            None,

        "task_type":
            None,

        "feature_config":
            None,

        "training_config":
            None,

        "metrics":
            None,

        "chemflow_package":
            False,
    }


# ============================================================
# Predictor
# ============================================================

class ChemFlowPredictor:

    def __init__(
        self,
        model_path: str | Path,
    ):

        self.model_path = Path(
            model_path
        )

        self.package = (
            load_pickle_model(
                self.model_path
            )
        )

        self.model = (
            self.package["model"]
        )

        if self.model is None:

            raise ValueError(
                "The loaded model package "
                "contains no model."
            )

        self.model_name = (
            self.package.get(
                "model_name"
            )
        )

        self.task_type = (
            self.package.get(
                "task_type"
            )
        )

        self.feature_config = (
            self.package.get(
                "feature_config"
            )
            or {}
        )


        # ====================================================
        # Feature representation
        # ====================================================

        self.feature_types = (

            self.feature_config.get(
                "feature_types"
            )

            or

            self.feature_config.get(
                "representations"
            )

            or

            self.feature_config.get(
                "features"
            )
        )


        if self.feature_types is None:

            raise ValueError(

                "The model package does not contain "
                "feature configuration.\n\n"

                "Expected one of:\n"
                "  feature_config['feature_types']\n"
                "  feature_config['representations']\n"
                "  feature_config['features']\n\n"

                "Please use the ChemFlow model package "
                "generated during training."
            )


        if isinstance(
            self.feature_types,
            str,
        ):

            self.feature_types = [
                self.feature_types
            ]


    # ========================================================
    # Model information
    # ========================================================

    def get_model_info(
        self,
    ) -> Dict[str, Any]:

        info = {

            "model_path":
                str(self.model_path),

            "model_name":
                self.model_name,

            "task_type":
                self.task_type,

            "model_type":
                type(self.model).__name__,

            "feature_types":
                self.feature_types,

            "n_bits":
                self.feature_config.get(
                    "n_bits"
                ),

            "fp_bits":
                self.feature_config.get(
                    "fp_bits"
                ),

            "desc_names":
                self.feature_config.get(
                    "desc_names"
                ),

            "smiles_col":
                self.feature_config.get(
                    "smiles_col"
                ),

            "target_col":
                self.feature_config.get(
                    "target_col"
                ),

            "feature_array_shapes":
                self.feature_config.get(
                    "feature_array_shapes"
                ),

            "metrics":
                self.package.get(
                    "metrics"
                ),
        }


        # ====================================================
        # Inspect sklearn Pipeline
        # ====================================================

        if hasattr(
            self.model,
            "named_steps",
        ):

            info[
                "pipeline_steps"
            ] = list(
                self.model.named_steps.keys()
            )


            # -----------------------------------------------
            # VarianceThreshold diagnostics
            # -----------------------------------------------

            if (
                "variance_filter"
                in self.model.named_steps
            ):

                variance_filter = (
                    self.model.named_steps[
                        "variance_filter"
                    ]
                )

                if hasattr(
                    variance_filter,
                    "variances_",
                ):

                    mask = (
                        variance_filter
                        .get_support()
                    )

                    info[
                        "features_before_variance_filter"
                    ] = int(
                        len(mask)
                    )

                    info[
                        "features_after_variance_filter"
                    ] = int(
                        mask.sum()
                    )

                    info[
                        "zero_variance_features_removed"
                    ] = int(
                        (~mask).sum()
                    )


        return info


    # ========================================================
    # Featurize
    # ========================================================

    def featurize(
        self,
        smiles_list: Sequence[str],
    ):

        smiles_list = list(
            smiles_list
        )

        dummy_y = np.zeros(
            len(smiles_list),
            dtype=float,
        )


        X, _, valid_indices = (
            featurize_array(

                np.asarray(
                    smiles_list
                ),

                dummy_y,

                self.feature_types,
            )
        )


        if X is None:

            return (
                None,
                None,
            )


        # ====================================================
        # Verify expected raw feature dimension
        # ====================================================

        expected_shapes = (
            self.feature_config.get(
                "feature_array_shapes"
            )
            or {}
        )


        expected_n_features = None


        for key in (
            "X_train",
            "X_valid",
            "X_test",
        ):

            shape = (
                expected_shapes.get(
                    key
                )
            )

            if (
                shape is not None
                and len(shape) >= 2
            ):

                expected_n_features = int(
                    shape[1]
                )

                break


        # ----------------------------------------------------
        # Fallback: inspect saved estimator pipeline
        # ----------------------------------------------------

        if expected_n_features is None:

            if hasattr(
                self.model,
                "n_features_in_",
            ):

                # For a pipeline, n_features_in_
                # represents the original input dimension.
                expected_n_features = int(
                    self.model.n_features_in_
                )


        # ====================================================
        # Validate
        # ====================================================

        if (
            expected_n_features is not None
            and
            X.shape[1]
            != expected_n_features
        ):

            raise ValueError(

                "Feature dimension mismatch.\n\n"

                f"Model expects "
                f"{expected_n_features} raw features.\n"

                f"Prediction featurization produced "
                f"{X.shape[1]} features.\n\n"

                f"Feature types: "
                f"{self.feature_types}\n\n"

                "This usually means the feature "
                "generation configuration used during "
                "prediction does not match training."
            )


        return (
            X,
            valid_indices,
        )


    # ========================================================
    # Predict
    # ========================================================

    def predict(
        self,
        smiles_list: Sequence[str],
    ) -> pd.DataFrame:

        smiles_list = list(
            smiles_list
        )


        X, valid_indices = (
            self.featurize(
                smiles_list
            )
        )


        if (
            X is None
            or
            len(X) == 0
        ):

            raise ValueError(
                "No valid molecules could be featurized."
            )


        # ====================================================
        # IMPORTANT
        #
        # If self.model is:
        #
        # Pipeline(
        #     preprocessor
        #     -> variance_filter
        #     -> model
        # )
        #
        # then all fitted preprocessing is automatically
        # applied here.
        #
        # DO NOT:
        #   - manually scale descriptors
        #   - fit another scaler
        #   - run VarianceThreshold.fit()
        #   - manually remove columns
        # ====================================================

        y_pred = self.model.predict(
            X
        )


        result_df = pd.DataFrame(

            {
                "input_index":
                    valid_indices,

                "smiles":
                    [
                        smiles_list[i]
                        for i in valid_indices
                    ],

                "prediction":
                    np.asarray(
                        y_pred,
                        dtype=float,
                    ),
            }
        )


        # ====================================================
        # Classification probabilities
        # ====================================================

        if hasattr(
            self.model,
            "predict_proba",
        ):

            try:

                y_proba = (
                    self.model.predict_proba(
                        X
                    )
                )


                if y_proba.ndim == 2:

                    for i in range(
                        y_proba.shape[1]
                    ):

                        result_df[
                            f"prob_class_{i}"
                        ] = y_proba[
                            :,
                            i
                        ]


                    result_df[
                        "confidence"
                    ] = np.max(
                        y_proba,
                        axis=1,
                    )


            except Exception:
                pass


        return result_df


    # ========================================================
    # Predict one molecule
    # ========================================================

    def predict_one(
        self,
        smiles: str,
    ) -> Dict[str, Any]:

        result = self.predict(
            [smiles]
        )


        if len(result) == 0:

            raise ValueError(
                "Invalid SMILES or failed "
                f"featurization: {smiles}"
            )


        return (
            result.iloc[0]
            .to_dict()
        )


    # ========================================================
    # Predict DataFrame
    # ========================================================

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str | None = None,
    ) -> pd.DataFrame:

        if smiles_col is None:

            smiles_col = (

                self.feature_config.get(
                    "smiles_col"
                )

                or

                "SMILES"
            )


        if smiles_col not in df.columns:

            raise ValueError(

                f"SMILES column "
                f"'{smiles_col}' "
                "not found in dataframe.\n"

                f"Available columns: "
                f"{df.columns.tolist()}"
            )


        pred_df = self.predict(

            df[
                smiles_col
            ]
            .astype(str)
            .tolist()
        )


        # ----------------------------------------------------
        # Preserve valid input rows
        # ----------------------------------------------------

        output = (

            df.iloc[
                pred_df[
                    "input_index"
                ].to_numpy()
            ]

            .reset_index(
                drop=True
            )
        )


        # ----------------------------------------------------
        # Append predictions
        # ----------------------------------------------------

        pred_columns = (
            pred_df.drop(
                columns=[
                    "input_index",
                    "smiles",
                ]
            )
            .reset_index(
                drop=True
            )
        )


        output = pd.concat(

            [
                output,
                pred_columns,
            ],

            axis=1,
        )


        return output