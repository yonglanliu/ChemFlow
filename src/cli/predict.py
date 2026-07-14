# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.deep_learning.graphormer.inference.predictor import (
    GraphormerPredictor,
)


SUPPORTED_INPUT_SUFFIXES = {
    ".smi",
    ".smiles",
    ".txt",
    ".csv",
    ".parquet",
    ".pq",
}


def read_smi_file(
    path: str | Path,
    structure_column: str = "structure",
) -> pd.DataFrame:
    """
    Read a .smi/.smiles file.

    Supported formats
    -----------------
    One column:
        CCO
        CCN
        c1ccccc1

    Two or more whitespace-separated columns:
        CCO ethanol
        CCN ethylamine

    The first column is interpreted as SMILES. The remaining text is
    interpreted as a molecule name.
    """
    path = Path(path).expanduser().resolve()

    records: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)

            smiles = parts[0].strip()
            name = (
                parts[1].strip()
                if len(parts) > 1
                else f"molecule_{len(records) + 1}"
            )

            records.append(
                {
                    "molecule_id": name,
                    structure_column: smiles,
                }
            )

    if not records:
        raise ValueError(
            f"No molecules were found in SMI file: {path}"
        )

    return pd.DataFrame(records)


def load_inference_input(
    *,
    smiles: str | None,
    input_path: str | Path | None,
    structure_column: str,
) -> pd.DataFrame:
    """
    Load inference structures from a single SMILES or an input file.
    """
    if smiles is not None and input_path is not None:
        raise ValueError(
            "Provide either --smiles or --input, not both."
        )

    if smiles is None and input_path is None:
        raise ValueError(
            "One of --smiles or --input must be provided."
        )

    # ---------------------------------------------------------
    # Single SMILES
    # ---------------------------------------------------------
    if smiles is not None:
        smiles = smiles.strip()

        if not smiles:
            raise ValueError("--smiles cannot be empty.")

        return pd.DataFrame(
            {
                "molecule_id": ["query_1"],
                structure_column: [smiles],
            }
        )

    # ---------------------------------------------------------
    # Input file
    # ---------------------------------------------------------
    path = Path(input_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Inference input file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported input format '{suffix}'. "
            f"Expected one of {sorted(SUPPORTED_INPUT_SUFFIXES)}."
        )

    if suffix in {".smi", ".smiles", ".txt"}:
        frame = read_smi_file(
            path=path,
            structure_column=structure_column,
        )

    elif suffix == ".csv":
        frame = pd.read_csv(path)

    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)

    else:
        raise RuntimeError(
            f"Unhandled input suffix: {suffix}"
        )

    if structure_column not in frame.columns:
        raise KeyError(
            f"Structure column '{structure_column}' was not found in "
            f"{path}. Available columns: {list(frame.columns)}"
        )

    frame = frame.copy()

    frame[structure_column] = (
        frame[structure_column]
        .astype("string")
        .str.strip()
    )

    invalid_mask = (
        frame[structure_column].isna()
        | frame[structure_column].eq("")
    )

    if invalid_mask.any():
        invalid_rows = frame.index[invalid_mask].tolist()

        raise ValueError(
            f"Found {len(invalid_rows)} empty structures in column "
            f"'{structure_column}'. Example row indices: "
            f"{invalid_rows[:10]}"
        )

    return frame.reset_index(drop=True)


def save_prediction_frame(
    frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save predictions as CSV or Parquet.
    """
    output_path = Path(
        output_path
    ).expanduser().resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        frame.to_csv(
            output_path,
            index=False,
        )

    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(
            output_path,
            index=False,
        )

    else:
        raise ValueError(
            "Output file must end with .csv, .parquet, or .pq, "
            f"got: {output_path}"
        )

    return output_path


def predict_graphormer(args) -> None:
    """
    Run Graphormer inference from one SMILES or a molecular structure file.
    """
    if not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError(
            f"--threshold must be between 0 and 1, got {args.threshold}."
        )

    if int(args.batch_size) < 1:
        raise ValueError(
            f"--batch-size must be positive, got {args.batch_size}."
        )

    if int(args.num_workers) < 0:
        raise ValueError(
            f"--num-workers cannot be negative, got {args.num_workers}."
        )

    input_frame = load_inference_input(
        smiles=args.smiles,
        input_path=args.input,
        structure_column=args.structure_column,
    )

    predictor = GraphormerPredictor(
        checkpoint_path=args.model_checkpoint,
        device=args.device,
        threshold=args.threshold,
    )

    structures = (
        input_frame[args.structure_column]
        .astype(str)
        .tolist()
    )

    prediction_frame = predictor.predict_smiles(
        smiles_list=structures,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if len(input_frame) != len(prediction_frame):
        raise RuntimeError(
            "The number of predictions does not match the number "
            f"of input structures: {len(prediction_frame)} versus "
            f"{len(input_frame)}."
        )

    duplicate_columns = set(input_frame.columns).intersection(
        prediction_frame.columns
    )

    if duplicate_columns:
        raise ValueError(
            "Prediction output contains columns already present in the "
            f"input data: {sorted(duplicate_columns)}."
        )

    result_frame = pd.concat(
        [
            input_frame.reset_index(drop=True),
            prediction_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    output_path = save_prediction_frame(
        frame=result_frame,
        output_path=args.output,
    )

    print(f"Input molecules: {len(input_frame):,}")
    print(f"Predicted molecules: {len(prediction_frame):,}")
    print(f"Predictions saved to: {output_path}")


def add_graphormer_predict_parser(subparsers) -> None:
    """
    Add the ``predict graphormer`` CLI command.

    Examples
    --------
    Single SMILES:

        chemflow predict graphormer \
            --smiles "CCO" \
            --model-checkpoint best_model.pt \
            --output predictions.csv

    Molecular file:

        chemflow predict graphormer \
            --input molecules.csv \
            --structure-column SMILES \
            --model-checkpoint best_model.pt \
            --output predictions.csv
    """
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run model inference.",
    )

    model_subparsers = predict_parser.add_subparsers(
        dest="predict_model",
        required=True,
    )

    graphormer_parser = model_subparsers.add_parser(
        "graphormer",
        help=(
            "Run Graphormer inference on one SMILES or a molecular "
            "structure file."
        ),
    )

    # ---------------------------------------------------------
    # Input source
    # ---------------------------------------------------------
    input_group = graphormer_parser.add_mutually_exclusive_group(
        required=True,
    )

    input_group.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="Predict one SMILES string.",
    )

    input_group.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Input .smi, .smiles, .txt, .csv, .parquet, "
            "or .pq file."
        ),
    )

    graphormer_parser.add_argument(
        "--structure-column",
        type=str,
        default="SMILES",
        help=(
            "Column containing SMILES for CSV or Parquet input. "
            "Default: SMILES."
        ),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    graphormer_parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to a full Graphormer checkpoint.",
    )

    graphormer_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold for binary classification. "
            "Default: 0.5."
        ),
    )

    graphormer_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Inference device, such as cpu, cuda, cuda:0, or mps. "
            "Automatically selected when omitted."
        ),
    )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------
    graphormer_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size. Default: 64.",
    )

    graphormer_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Default: 0.",
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    graphormer_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .csv, .parquet, or .pq file.",
    )

    graphormer_parser.set_defaults(
        func=predict_graphormer,
    )