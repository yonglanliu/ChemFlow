import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.chemflow.machine_learning.data.data_pipeline import featurize_array


def load_pickle_model(model_path: str | Path) -> Dict[str, Any]:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "model" in obj:
        return {
            "model": obj.get("model"),
            "model_name": obj.get("model_name"),
            "task_type": obj.get("task_type"),
            "feature_config": obj.get("feature_config"),
            "training_config": obj.get("training_config"),
            "metrics": obj.get("metrics"),
            "chemflow_package": True,
        }

    return {
        "model": obj,
        "model_name": None,
        "task_type": None,
        "feature_config": None,
        "training_config": None,
        "metrics": None,
        "chemflow_package": False,
    }


class ChemFlowPredictor:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.package = load_pickle_model(self.model_path)

        self.model = self.package["model"]
        self.model_name = self.package.get("model_name")
        self.task_type = self.package.get("task_type")
        self.feature_config = self.package.get("feature_config") or {}

        self.feature_types = (
            self.feature_config.get("feature_types")
            or self.feature_config.get("representations")
        )

        if self.feature_types is None:
            raise ValueError(
                "This model package does not contain feature_config['feature_types']. "
                "Please use a ChemFlow model_package.pkl instead of raw best_model.pkl."
            )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "task_type": self.task_type,
            "feature_types": self.feature_types,
            "n_bits": self.feature_config.get("n_bits"),
            "desc_names": self.feature_config.get("desc_names"),
            "smiles_col": self.feature_config.get("smiles_col"),
            "target_col": self.feature_config.get("target_col"),
            "feature_array_shapes": self.feature_config.get("feature_array_shapes"),
            "metrics": self.package.get("metrics"),
        }

    def featurize(self, smiles_list: Sequence[str]):
        dummy_y = np.zeros(len(smiles_list), dtype=float)

        X, _, valid_indices = featurize_array(
            np.asarray(smiles_list),
            dummy_y,
            self.feature_types,
        )

        return X, valid_indices

    def predict(self, smiles_list: Sequence[str]) -> pd.DataFrame:
        smiles_list = list(smiles_list)

        X, valid_indices = self.featurize(smiles_list)

        if X is None or len(X) == 0:
            raise ValueError("No valid molecules could be featurized.")

        y_pred = self.model.predict(X)

        result_df = pd.DataFrame(
            {
                "input_index": valid_indices,
                "smiles": [smiles_list[i] for i in valid_indices],
                "prediction": y_pred,
            }
        )

        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.model.predict_proba(X)

                if y_proba.ndim == 2:
                    for i in range(y_proba.shape[1]):
                        result_df[f"prob_class_{i}"] = y_proba[:, i]

                    result_df["confidence"] = np.max(y_proba, axis=1)

            except Exception:
                pass

        return result_df

    def predict_one(self, smiles: str) -> Dict[str, Any]:
        df = self.predict([smiles])

        if len(df) == 0:
            raise ValueError(f"Invalid SMILES or failed featurization: {smiles}")

        return df.iloc[0].to_dict()

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str | None = None,
    ) -> pd.DataFrame:
        if smiles_col is None:
            smiles_col = self.feature_config.get("smiles_col", "SMILES")

        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in dataframe.")

        pred_df = self.predict(df[smiles_col].astype(str).tolist())

        output = df.iloc[pred_df["input_index"].to_numpy()].reset_index(drop=True)
        output = pd.concat(
            [
                output,
                pred_df.drop(columns=["input_index", "smiles"]).reset_index(drop=True),
            ],
            axis=1,
        )

        return output