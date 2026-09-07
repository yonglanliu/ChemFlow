from __future__ import annotations

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.chemflow.featurization import (
    DESC_NAMES,
    DESC_TYPES,
    FP_BITS,
    FP_TYPES,
    MACCS_TYPES,
    smiles_to_descriptors,
    smiles_to_fp,
    smiles_to_maccs,
)


# ============================================================
# Feature-type normalization
# ============================================================

def _normalize_feature_types(
    feature_types,
):
    if feature_types is None:
        return ["ecfp4"]

    if isinstance(feature_types, str):
        feature_types = [feature_types]

    return [
        str(feature).lower()
        for feature in feature_types
    ]


# ============================================================
# Featurize one molecule
# ============================================================

def _featurize_single_smiles(
    smi,
    feature_types,
):
    mol_features = []

    for feature_type in feature_types:

        # ----------------------------------------------------
        # ECFP4
        # ----------------------------------------------------

        if feature_type == "ecfp4":

            x = smiles_to_fp(
                smi,
                radius=2,
                n_bits=FP_BITS["ecfp4"],
                use_features=False,
            )

        # ----------------------------------------------------
        # ECFP6
        # ----------------------------------------------------

        elif feature_type == "ecfp6":

            x = smiles_to_fp(
                smi,
                radius=3,
                n_bits=FP_BITS["ecfp6"],
                use_features=False,
            )

        # ----------------------------------------------------
        # FCFP4
        # ----------------------------------------------------

        elif feature_type == "fcfp4":

            x = smiles_to_fp(
                smi,
                radius=2,
                n_bits=FP_BITS["fcfp4"],
                use_features=True,
            )

        # ----------------------------------------------------
        # FCFP6
        # ----------------------------------------------------

        elif feature_type == "fcfp6":

            x = smiles_to_fp(
                smi,
                radius=3,
                n_bits=FP_BITS["fcfp6"],
                use_features=True,
            )

        # ----------------------------------------------------
        # MACCS
        # ----------------------------------------------------

        elif feature_type in {
            "maccs",
            "macc",
        }:

            x = smiles_to_maccs(
                smi
            )

        # ----------------------------------------------------
        # Descriptors
        # ----------------------------------------------------

        elif feature_type in {
            "descriptor",
            "descriptors",
        }:

            x = smiles_to_descriptors(
                smi
            )

        else:

            raise ValueError(
                f"Unknown feature_type: {feature_type}"
            )

        # ----------------------------------------------------
        # Invalid feature
        # ----------------------------------------------------

        if x is None:
            return None

        x = np.asarray(
            x,
            dtype=np.float32,
        ).ravel()

        # ----------------------------------------------------
        # Remove invalid numerical vectors
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(x)
        ):
            return None

        mol_features.append(
            x
        )

    if not mol_features:
        return None

    # IMPORTANT:
    # Feature order follows feature_types.
    #
    # For:
    #   ["descriptors", "ecfp4"]
    #
    # output is:
    #
    #   9 descriptors
    #   +
    #   2048 ECFP4 bits
    #
    # = 2057 features
    return np.concatenate(
        mol_features,
        axis=0,
    )


# ============================================================
# Featurize DataFrame
# ============================================================

def featurize_dataframe(
    df,
    smiles_col="SMILES",
    feature_types=None,
):
    """
    Featurize molecules from a DataFrame.

    Returns
    -------
    X:
        Feature matrix for valid molecules.

    clean_df:
        DataFrame containing only molecules that were
        successfully featurized.
    """

    feature_types = (
        _normalize_feature_types(
            feature_types
        )
    )

    X_list = []
    valid_indices = []

    for idx, smi in df[
        smiles_col
    ].items():

        x = _featurize_single_smiles(
            smi,
            feature_types,
        )

        if x is not None:

            X_list.append(
                x
            )

            valid_indices.append(
                idx
            )

    if not X_list:

        return None, None

    X = np.vstack(
        X_list
    ).astype(
        np.float32
    )

    clean_df = df.loc[
        valid_indices
    ].copy()

    return X, clean_df


# ============================================================
# Featurize arrays
# ============================================================

def featurize_array(
    X,
    y,
    feature_types=None,
):
    """
    Featurize SMILES stored in an array.

    Parameters
    ----------
    X:
        Array-like collection of SMILES.

    y:
        Target values.

    feature_types:
        Feature representations.

    Returns
    -------
    X_features
        Feature matrix.

    y_clean
        Labels corresponding to valid molecules.

    valid_indices
        Original row indices retained after featurization.
    """

    feature_types = (
        _normalize_feature_types(
            feature_types
        )
    )

    X_features = []
    y_clean = []
    valid_indices = []

    for idx, smi in enumerate(X):

        x = _featurize_single_smiles(
            smi,
            feature_types,
        )

        if x is not None:

            X_features.append(
                x
            )

            y_clean.append(
                y[idx]
            )

            valid_indices.append(
                idx
            )

    if not X_features:

        return (
            None,
            None,
            None,
        )

    X_features = np.vstack(
        X_features
    ).astype(
        np.float32
    )

    y_clean = np.asarray(
        y_clean
    )

    valid_indices = np.asarray(
        valid_indices
    )

    return (
        X_features,
        y_clean,
        valid_indices,
    )


# ============================================================
# Build preprocessing + model pipeline
# ============================================================

def make_scaled_pipeline(
    model,
    feature_types,
):
    """
    Build the preprocessing/model pipeline.

    Rules
    -----
    Fingerprints only:
        VarianceThreshold(0.0)
        -> model

    Descriptors only:
        VarianceThreshold(0.0)
        -> StandardScaler
        -> model

    Mixed descriptors + fingerprints:
        ColumnTransformer:
            descriptors -> StandardScaler
            fingerprints -> passthrough

        then:
            VarianceThreshold(0.0)
            -> model

    IMPORTANT
    ---------
    VarianceThreshold is inside the sklearn Pipeline.
    Therefore, during cross-validation, the variance filter
    is fitted using only the corresponding training fold.

    During prediction, the fitted mask stored in the pipeline
    is automatically reused.
    """

    feature_types = (
        _normalize_feature_types(
            feature_types
        )
    )

    desc_indices = []

    current_start = 0

    has_fp = False
    has_desc = False

    # ========================================================
    # Determine feature positions
    # ========================================================

    for feature in feature_types:

        # ----------------------------------------------------
        # Descriptors
        # ----------------------------------------------------

        if (
            feature in DESC_TYPES
            or feature in {
                "descriptor",
                "descriptors",
            }
        ):

            has_desc = True

            n_desc = len(
                DESC_NAMES
            )

            desc_indices.extend(
                range(
                    current_start,
                    current_start + n_desc,
                )
            )

            current_start += (
                n_desc
            )

        # ----------------------------------------------------
        # Fingerprints
        # ----------------------------------------------------

        elif feature in FP_TYPES:

            has_fp = True

            current_start += int(
                FP_BITS[feature]
            )

        # ----------------------------------------------------
        # MACCS
        # ----------------------------------------------------

        elif (
            feature in MACCS_TYPES
            or feature in {
                "maccs",
                "macc",
            }
        ):

            has_fp = True

            current_start += int(
                FP_BITS["maccs"]
            )

        else:

            raise ValueError(
                f"Unknown feature type: {feature}"
            )

    # ========================================================
    # Fingerprints only
    # ========================================================

    if has_fp and not has_desc:

        return Pipeline(
            [
                (
                    "variance_filter",
                    VarianceThreshold(
                        threshold=0.0
                    ),
                ),

                (
                    "model",
                    model,
                ),
            ]
        )

    # ========================================================
    # Descriptors only
    # ========================================================

    if has_desc and not has_fp:

        return Pipeline(
            [
                (
                    "variance_filter",
                    VarianceThreshold(
                        threshold=0.0
                    ),
                ),

                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "model",
                    model,
                ),
            ]
        )

    # ========================================================
    # Mixed descriptors + fingerprints
    # ========================================================

    preprocessor = ColumnTransformer(

        transformers=[
            (
                "descriptor_scaler",
                StandardScaler(),
                desc_indices,
            ),
        ],

        # Fingerprints and any other non-descriptor columns
        # pass through unchanged.
        remainder="passthrough",
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),

            (
                "variance_filter",
                VarianceThreshold(
                    threshold=0.0
                ),
            ),

            (
                "model",
                model,
            ),
        ]
    )