from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.chemflow.machine_learning.predict.predictor import (
    ChemFlowPredictor,
)

from src.deep_learning.graphormer.inference.predictor import (
    GraphormerPredictor,
)


# ============================================================
# Input helpers
# ============================================================

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
    structure_column: str,
) -> pd.DataFrame:

    path = Path(
        path
    ).expanduser().resolve()


    records = []


    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split(
                maxsplit=1
            )

            smiles = parts[0].strip()

            name = (
                parts[1].strip()
                if len(parts) > 1
                else f"molecule_{len(records) + 1}"
            )

            records.append(
                {
                    "molecule_id":
                        name,

                    structure_column:
                        smiles,
                }
            )


    if not records:

        raise ValueError(
            f"No molecules found in: {path}"
        )


    return pd.DataFrame(
        records
    )


def load_inference_input(
    *,
    smiles: str | None,
    input_path: str | Path | None,
    structure_column: str,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # Validate mutually exclusive inputs
    # --------------------------------------------------------

    if (
        smiles is not None
        and
        input_path is not None
    ):

        raise ValueError(
            "Provide either --smiles or --input, "
            "not both."
        )


    if (
        smiles is None
        and
        input_path is None
    ):

        raise ValueError(
            "One of --smiles or --input is required."
        )


    # --------------------------------------------------------
    # Single SMILES
    # --------------------------------------------------------

    if smiles is not None:

        smiles = smiles.strip()

        if not smiles:

            raise ValueError(
                "--smiles cannot be empty."
            )

        return pd.DataFrame(
            {
                "molecule_id":
                    ["query_1"],

                structure_column:
                    [smiles],
            }
        )


    # --------------------------------------------------------
    # File input
    # --------------------------------------------------------

    path = Path(
        input_path
    ).expanduser().resolve()


    if not path.is_file():

        raise FileNotFoundError(
            f"Input file not found: {path}"
        )


    suffix = (
        path.suffix.lower()
    )


    if suffix not in (
        SUPPORTED_INPUT_SUFFIXES
    ):

        raise ValueError(

            f"Unsupported input format: "
            f"{suffix}\n"

            f"Supported formats: "
            f"{sorted(SUPPORTED_INPUT_SUFFIXES)}"
        )


    # --------------------------------------------------------
    # SMILES-like text
    # --------------------------------------------------------

    if suffix in {
        ".smi",
        ".smiles",
        ".txt",
    }:

        frame = read_smi_file(
            path,
            structure_column,
        )


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    elif suffix == ".csv":

        frame = pd.read_csv(
            path
        )


    # --------------------------------------------------------
    # Parquet
    # --------------------------------------------------------

    elif suffix in {
        ".parquet",
        ".pq",
    }:

        frame = pd.read_parquet(
            path
        )


    else:

        raise RuntimeError(
            f"Unhandled input suffix: {suffix}"
        )


    # --------------------------------------------------------
    # Check structure column
    # --------------------------------------------------------

    if structure_column not in frame.columns:

        raise KeyError(

            f"Structure column "
            f"'{structure_column}' "
            "not found.\n"

            f"Available columns: "
            f"{frame.columns.tolist()}"
        )


    frame = frame.copy()


    frame[structure_column] = (
        frame[
            structure_column
        ]
        .astype("string")
        .str.strip()
    )


    invalid = (
        frame[
            structure_column
        ].isna()
        |
        frame[
            structure_column
        ].eq("")
    )


    if invalid.any():

        raise ValueError(

            f"Found "
            f"{int(invalid.sum())} "
            f"empty SMILES in column "
            f"'{structure_column}'."
        )


    return frame.reset_index(
        drop=True
    )


# ============================================================
# Output helper
# ============================================================

def save_prediction_frame(
    frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:

    output_path = Path(
        output_path
    ).expanduser().resolve()


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    suffix = (
        output_path.suffix.lower()
    )


    if suffix == ".csv":

        frame.to_csv(
            output_path,
            index=False,
        )

    elif suffix in {
        ".parquet",
        ".pq",
    }:

        frame.to_parquet(
            output_path,
            index=False,
        )

    else:

        raise ValueError(

            "Output must be .csv, "
            ".parquet, or .pq."
        )


    return output_path


# ============================================================
# Traditional ML prediction
# ============================================================

def predict_ml(
    args,
) -> None:

    print(
        "\n" + "=" * 70
    )

    print(
        "CHEMFLOW TRADITIONAL ML PREDICTION"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_frame = load_inference_input(

        smiles=args.smiles,

        input_path=args.input,

        structure_column=args.structure_column,
    )


    print(
        f"\nInput molecules: "
        f"{len(input_frame):,}"
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    predictor = ChemFlowPredictor(
        model_path=args.model
    )


    # --------------------------------------------------------
    # Print model configuration
    # --------------------------------------------------------

    print(
        "\nModel configuration:"
    )


    info = (
        predictor.get_model_info()
    )


    for key, value in info.items():

        print(
            f"  {key}: {value}"
        )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    result_frame = (
        predictor.predict_from_dataframe(
            df=input_frame,
            smiles_col=args.structure_column,
        )
    )


    # --------------------------------------------------------
    # Rename prediction column
    # --------------------------------------------------------

    prediction_name = (
        args.task_name
    )


    if (
        prediction_name
        and
        "prediction"
        in result_frame.columns
    ):

        result_frame.rename(

            columns={
                "prediction":
                    prediction_name
            },

            inplace=True,
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = save_prediction_frame(
        result_frame,
        args.output,
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "ML PREDICTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Predicted molecules: "
        f"{len(result_frame):,}"
    )

    print(
        f"Output: "
        f"{output_path}"
    )


# ============================================================
# Graphormer prediction
# ============================================================

def predict_graphormer(
    args,
) -> None:

    print(
        "\n" + "=" * 70
    )

    print(
        "CHEMFLOW GRAPHORMER PREDICTION"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_frame = load_inference_input(

        smiles=args.smiles,

        input_path=args.input,

        structure_column=args.structure_column,
    )


    structures = (
        input_frame[
            args.structure_column
        ]
        .astype(str)
        .tolist()
    )


    print(
        f"\nInput molecules: "
        f"{len(structures):,}"
    )


    # --------------------------------------------------------
    # Load Graphormer
    # --------------------------------------------------------

    predictor = GraphormerPredictor(

        checkpoint_path=
            args.model_checkpoint,

        device=
            args.device,

        threshold=
            args.threshold,

        validation_predictions=
            None,

        validation_targets=
            None,
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction_frame = (
        predictor.predict_smiles(

            smiles_list=
                structures,

            batch_size=
                args.batch_size,

            num_workers=
                args.num_workers,
        )
    )


    # --------------------------------------------------------
    # Check prediction count
    # --------------------------------------------------------

    if (
        len(prediction_frame)
        !=
        len(input_frame)
    ):

        raise RuntimeError(

            "Prediction count does not match "
            "input molecule count.\n"

            f"Input: "
            f"{len(input_frame)}\n"

            f"Predictions: "
            f"{len(prediction_frame)}"
        )


    # --------------------------------------------------------
    # Rename prediction columns
    # --------------------------------------------------------

    task_names = [
        task.strip()
        for task in args.task_names
    ]


    if task_names:

        if (
            len(task_names)
            !=
            len(prediction_frame.columns)
        ):

            raise ValueError(

                "The number of --task-names "
                "must match the number of "
                "prediction columns.\n"

                f"Task names: "
                f"{len(task_names)}\n"

                f"Prediction columns: "
                f"{len(prediction_frame.columns)}"
            )


        prediction_frame = (
            prediction_frame.rename(
                columns=dict(
                    zip(
                        prediction_frame.columns,
                        task_names,
                    )
                )
            )
        )


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    result_frame = pd.concat(

        [
            input_frame.reset_index(
                drop=True
            ),

            prediction_frame.reset_index(
                drop=True
            ),
        ],

        axis=1,
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = save_prediction_frame(
        result_frame,
        args.output,
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "GRAPHORMER PREDICTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Predicted molecules: "
        f"{len(result_frame):,}"
    )

    print(
        f"Output: "
        f"{output_path}"
    )


# ============================================================
# ML parser
# ============================================================

def add_ml_predict_parser(
    model_subparsers,
) -> None:

    ml_parser = model_subparsers.add_parser(
        "ml",
        help=(
            "Run traditional ML inference "
            "using a saved .pkl model."
        ),
    )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_group = (
        ml_parser.add_mutually_exclusive_group(
            required=True
        )
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
            "Input .smi, .smiles, .txt, .csv, "
            ".parquet, or .pq file."
        ),
    )


    # --------------------------------------------------------
    # Structure column
    # --------------------------------------------------------

    ml_parser.add_argument(
        "--structure-column",
        type=str,
        default="SMILES",
        help=(
            "Column containing SMILES. "
            "Default: SMILES."
        ),
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    ml_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Path to trained ML .pkl model "
            "or ChemFlow model package."
        ),
    )


    # --------------------------------------------------------
    # Prediction column
    # --------------------------------------------------------

    ml_parser.add_argument(
        "--task-name",
        type=str,
        default="prediction",
        help=(
            "Name of prediction column. "
            "Default: prediction."
        ),
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    ml_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output .csv, .parquet, or .pq file."
        ),
    )


    ml_parser.set_defaults(
        func=predict_ml
    )


# ============================================================
# Graphormer parser
# ============================================================

def add_graphormer_predict_parser(
    model_subparsers,
) -> None:

    graphormer_parser = (
        model_subparsers.add_parser(
            "graphormer",
            help=(
                "Run Graphormer inference "
                "on molecular structures."
            ),
        )
    )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_group = (
        graphormer_parser.add_mutually_exclusive_group(
            required=True
        )
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
            "Input .smi, .smiles, .txt, .csv, "
            ".parquet, or .pq file."
        ),
    )


    # --------------------------------------------------------
    # Structure column
    # --------------------------------------------------------

    graphormer_parser.add_argument(
        "--structure-column",
        type=str,
        default="SMILES",
        help=(
            "Column containing SMILES. "
            "Default: SMILES."
        ),
    )


    # --------------------------------------------------------
    # Task names
    # --------------------------------------------------------

    graphormer_parser.add_argument(
        "--task-names",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Names of prediction tasks."
        ),
    )


    # --------------------------------------------------------
    # Model checkpoint
    # --------------------------------------------------------

    graphormer_parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help=(
            "Path to Graphormer checkpoint."
        ),
    )


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    graphormer_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Binary classification threshold. "
            "Default: 0.5."
        ),
    )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    graphormer_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Inference device, e.g. cpu, cuda, "
            "cuda:0, or mps."
        ),
    )


    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    graphormer_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size. Default: 64.",
    )


    # --------------------------------------------------------
    # Workers
    # --------------------------------------------------------

    graphormer_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers. Default: 0."
        ),
    )


    graphormer_parser.set_defaults(
        func=predict_graphormer
    )


# ============================================================
# Main predict parser
# ============================================================

def add_predict_parser(
    subparsers,
) -> None:
    """
    Register the top-level:

        chemflow predict

    command with:

        chemflow predict ml
        chemflow predict graphormer
    """

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run model inference.",
    )


    model_subparsers = (
        predict_parser.add_subparsers(
            dest="predict_model",
            required=True,
        )
    )


    # Traditional ML
    add_ml_predict_parser(
        model_subparsers
    )


    # Graphormer
    add_graphormer_predict_parser(
        model_subparsers
    )