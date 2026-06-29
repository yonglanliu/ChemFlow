import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.chemflow.featurization import (
    FP_TYPES,
    MACCS_TYPES,
    DESC_TYPES,
    FP_BITS,
    DESC_NAMES,
    smiles_to_fp,
    smiles_to_descriptors,
    smiles_to_maccs,
)


# ============================================================
# Featurization helpers
# ============================================================
def _normalize_feature_types(feature_types):
    if feature_types is None:
        return ["ecfp4"]

    if isinstance(feature_types, str):
        feature_types = [feature_types]

    return [str(f).lower() for f in feature_types]


def _featurize_single_smiles(smi, feature_types):
    mol_features = []

    for feature_type in feature_types:
        if feature_type == "ecfp4":
            x = smiles_to_fp(
                smi,
                radius=2,
                n_bits=FP_BITS["ecfp4"],
                use_features=False,
            )

        elif feature_type == "ecfp6":
            x = smiles_to_fp(
                smi,
                radius=3,
                n_bits=FP_BITS["ecfp6"],
                use_features=False,
            )

        elif feature_type == "fcfp4":
            x = smiles_to_fp(
                smi,
                radius=2,
                n_bits=FP_BITS["fcfp4"],
                use_features=True,
            )

        elif feature_type == "fcfp6":
            x = smiles_to_fp(
                smi,
                radius=3,
                n_bits=FP_BITS["fcfp6"],
                use_features=True,
            )

        elif feature_type in ["maccs", "macc"]:
            x = smiles_to_maccs(smi)

        elif feature_type in ["descriptor", "descriptors"]:
            x = smiles_to_descriptors(smi)

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        if x is None:
            return None

        x = np.asarray(x, dtype=np.float32).ravel()

        if not np.all(np.isfinite(x)):
            return None

        mol_features.append(x)

    return np.concatenate(mol_features, axis=0)


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

    Supports:
        - ECFP4 / ECFP6
        - FCFP4 / FCFP6
        - MACCS / MACC
        - descriptor / descriptors

    Returns:
        X: np.ndarray, shape = (n_valid_molecules, n_features)
        clean_df: DataFrame containing only valid molecules
    """

    feature_types = _normalize_feature_types(feature_types)

    X_list = []
    valid_indices = []

    for idx, smi in df[smiles_col].items():
        x = _featurize_single_smiles(smi, feature_types)

        if x is not None:
            X_list.append(x)
            valid_indices.append(idx)

    if len(X_list) == 0:
        return None, None

    X = np.vstack(X_list).astype(np.float32)
    clean_df = df.loc[valid_indices].copy()

    return X, clean_df


# ============================================================
# Featurize arrays
# ============================================================
def featurize_array(
    X: np.ndarray,
    y: np.ndarray,
    feature_types=None,
):
    """
    Featurize SMILES stored in an array.

    Args:
        X: array-like SMILES
        y: array-like labels
        feature_types: list of feature types

    Returns:
        X_features: np.ndarray
        y_clean: np.ndarray
        valid_indices: np.ndarray
    """

    feature_types = _normalize_feature_types(feature_types)

    X_features = []
    y_clean = []
    valid_indices = []

    for idx, smi in enumerate(X):
        x = _featurize_single_smiles(smi, feature_types)

        if x is not None:
            X_features.append(x)
            y_clean.append(y[idx])
            valid_indices.append(idx)

    if len(X_features) == 0:
        return None, None, None

    X_features = np.vstack(X_features).astype(np.float32)
    y_clean = np.asarray(y_clean)
    valid_indices = np.asarray(valid_indices)

    return X_features, y_clean, valid_indices


# ============================================================
# Preprocessing
# ============================================================
def make_scaled_pipeline(
    model,
    feature_types,
):
    """
    Scaling rules:
        - fingerprints only: no scaling
        - descriptors only: scale all columns
        - mixed features: scale descriptor columns only

    Feature order follows feature_types.
    """

    feature_types = _normalize_feature_types(feature_types)

    desc_indices = []
    current_start = 0
    has_fp = False
    has_desc = False

    for feature in feature_types:
        if feature in DESC_TYPES or feature in ["descriptor", "descriptors"]:
            has_desc = True
            n_desc = len(DESC_NAMES)

            desc_indices.extend(
                range(current_start, current_start + n_desc)
            )

            current_start += n_desc

        elif feature in FP_TYPES:
            has_fp = True
            current_start += int(FP_BITS[feature])

        elif feature in MACCS_TYPES or feature in ["maccs", "macc"]:
            has_fp = True
            current_start += int(FP_BITS["maccs"])

        else:
            raise ValueError(f"Unknown feature type: {feature}")

    if has_fp and not has_desc:
        return model

    if has_desc and not has_fp:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "descriptor_scaler",
                StandardScaler(),
                desc_indices,
            )
        ],
        remainder="passthrough",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )