from pathlib import Path
import pickle


def load_pickle_model(model_path: str | Path):
    """
    Load a trained sklearn/XGBoost model from a pickle file.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model