from __future__ import annotations

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR

from chemflow.featurization import (
    DESC_NAMES,
    DESC_TYPES,
    FP_BITS,
    FP_TYPES,
    MACCS_TYPES,
    RDKIT2D_DESC_NAMES,
    RDKIT2D_DESC_TYPES,
    smiles_to_descriptors,
    smiles_to_avalon,
    smiles_to_erg,
    smiles_to_fp,
    smiles_to_maccs,
    smiles_to_rdkit2d,
)


class CorrelationThreshold(BaseEstimator, TransformerMixin):
    """Remove later columns whose absolute training correlation is too high."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("CorrelationThreshold expects a two-dimensional array.")
        threshold = float(self.threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Correlation threshold must be in the interval (0, 1].")

        n_features = values.shape[1]
        if n_features == 0:
            raise ValueError("CorrelationThreshold received no features.")
        if n_features == 1:
            self.support_mask_ = np.ones(1, dtype=bool)
            self.n_features_in_ = 1
            return self

        correlation = np.corrcoef(values, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0)
        support = np.ones(n_features, dtype=bool)
        for column in range(1, n_features):
            previous = np.flatnonzero(support[:column])
            if previous.size and np.any(
                np.abs(correlation[column, previous]) > threshold
            ):
                support[column] = False

        self.support_mask_ = support
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        if not hasattr(self, "support_mask_"):
            raise RuntimeError("CorrelationThreshold must be fitted before transform.")
        values = np.asarray(X)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Input feature dimensions do not match the fitted data.")
        return values[:, self.support_mask_]

    def get_support(self, indices=False):
        if not hasattr(self, "support_mask_"):
            raise RuntimeError("CorrelationThreshold must be fitted before get_support.")
        if indices:
            return np.flatnonzero(self.support_mask_)
        return self.support_mask_.copy()


_LINEAR_OR_SVM_MODELS = (
    ElasticNet,
    Lasso,
    LinearSVC,
    LinearSVR,
    LogisticRegression,
    Ridge,
    SVC,
    SVR,
)


def _descriptor_pipeline(
    model,
    correlation_threshold,
    model_selection,
    selection_threshold,
    selection_estimators,
):
    steps = [
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        (
            "correlation_filter",
            CorrelationThreshold(threshold=float(correlation_threshold)),
        ),
        ("scaler", StandardScaler()),
    ]

    if bool(model_selection) and isinstance(model, _LINEAR_OR_SVM_MODELS):
        selector_model = (
            ExtraTreesClassifier(
                n_estimators=int(selection_estimators),
                random_state=42,
                n_jobs=1,
                class_weight="balanced",
            )
            if is_classifier(model)
            else ExtraTreesRegressor(
                n_estimators=int(selection_estimators),
                random_state=42,
                n_jobs=1,
            )
        )
        steps.append(
            (
                "model_selector",
                SelectFromModel(
                    selector_model,
                    threshold=selection_threshold,
                ),
            )
        )

    return Pipeline(steps)


def fitted_feature_dimensions(model, raw_feature_count):
    """Summarize the feature counts retained by a fitted ChemFlow pipeline."""
    raw_feature_count = int(raw_feature_count)
    summary = {
        "raw_total": raw_feature_count,
        "descriptor_raw": 0,
        "descriptor_after_zero_variance": 0,
        "descriptor_after_correlation": 0,
        "descriptor_after_model_selection": 0,
        "fingerprint_raw": raw_feature_count,
        "fingerprint_after_zero_variance": raw_feature_count,
        "final_model_input": raw_feature_count,
    }
    if not isinstance(model, Pipeline):
        return summary

    descriptor_pipeline = None
    if "descriptor_preprocessor" in model.named_steps:
        descriptor_pipeline = model.named_steps["descriptor_preprocessor"]
    elif "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        descriptor_pipeline = preprocessor.named_transformers_.get(
            "descriptor_preprocessor"
        )

    descriptor_final = 0
    if isinstance(descriptor_pipeline, Pipeline):
        descriptor_variance = descriptor_pipeline.named_steps["variance_filter"]
        descriptor_raw = int(descriptor_variance.n_features_in_)
        after_variance = int(descriptor_variance.get_support().sum())
        correlation_filter = descriptor_pipeline.named_steps["correlation_filter"]
        after_correlation = int(correlation_filter.get_support().sum())
        selector = descriptor_pipeline.named_steps.get("model_selector")
        descriptor_final = (
            int(selector.get_support().sum())
            if selector is not None
            else after_correlation
        )
        summary.update(
            {
                "descriptor_raw": descriptor_raw,
                "descriptor_after_zero_variance": after_variance,
                "descriptor_after_correlation": after_correlation,
                "descriptor_after_model_selection": descriptor_final,
                "fingerprint_raw": raw_feature_count - descriptor_raw,
            }
        )

    global_variance = model.named_steps.get("variance_filter")
    if global_variance is not None:
        support = global_variance.get_support()
        final_count = int(support.sum())
        fingerprint_start = descriptor_final
        fingerprint_after = int(support[fingerprint_start:].sum())
    else:
        final_count = descriptor_final or raw_feature_count
        fingerprint_after = summary["fingerprint_raw"]

    summary["fingerprint_after_zero_variance"] = fingerprint_after
    summary["final_model_input"] = final_count
    return summary


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
    descriptor_names_by_feature=None,
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
        # Avalon structural fingerprint
        # ----------------------------------------------------

        elif feature_type == "avalon":

            x = smiles_to_avalon(
                smi,
                n_bits=FP_BITS["avalon"],
            )

        # ----------------------------------------------------
        # ErG pharmacophore fingerprint
        # ----------------------------------------------------

        elif feature_type == "erg":

            x = smiles_to_erg(
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

        # ----------------------------------------------------
        # Expanded RDKit 2D descriptors
        # ----------------------------------------------------

        elif feature_type in RDKIT2D_DESC_TYPES:

            saved_names = (descriptor_names_by_feature or {}).get(feature_type)
            x = smiles_to_rdkit2d(
                smi,
                descriptor_names=saved_names,
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
    descriptor_names_by_feature=None,
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
            descriptor_names_by_feature=descriptor_names_by_feature,
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
    descriptor_names_by_feature=None,
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
            descriptor_names_by_feature=descriptor_names_by_feature,
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
    descriptor_correlation_threshold=0.95,
    descriptor_model_selection=False,
    descriptor_selection_threshold="mean",
    descriptor_selection_estimators=128,
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
        -> CorrelationThreshold
        -> StandardScaler
        -> optional model-based selection for linear/SVM models
        -> model

    Mixed descriptors + fingerprints:
        ColumnTransformer:
            descriptors -> variance/correlation filtering -> scaling
                           -> optional linear/SVM model selection
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

        elif feature in RDKIT2D_DESC_TYPES:

            has_desc = True
            n_desc = len(RDKIT2D_DESC_NAMES)
            desc_indices.extend(
                range(current_start, current_start + n_desc)
            )
            current_start += n_desc

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
                    "descriptor_preprocessor",
                    _descriptor_pipeline(
                        model=model,
                        correlation_threshold=descriptor_correlation_threshold,
                        model_selection=descriptor_model_selection,
                        selection_threshold=descriptor_selection_threshold,
                        selection_estimators=descriptor_selection_estimators,
                    ),
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
                "descriptor_preprocessor",
                _descriptor_pipeline(
                    model=model,
                    correlation_threshold=descriptor_correlation_threshold,
                    model_selection=descriptor_model_selection,
                    selection_threshold=descriptor_selection_threshold,
                    selection_estimators=descriptor_selection_estimators,
                ),
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
