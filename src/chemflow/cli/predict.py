from __future__ import annotations

import argparse
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd

from chemflow.machine_learning.predict.predictor import (
    ChemFlowPredictor,
)

# ============================================================
# Supported input formats
# ============================================================

SUPPORTED_INPUT_SUFFIXES = {
    ".smi",
    ".smiles",
    ".txt",
    ".csv",
    ".parquet",
    ".pq",
}


# ============================================================
# Read SMILES file
# ============================================================

def read_smi_file(
    path: str | Path,
    structure_column: str,
) -> pd.DataFrame:
    """
    Read a .smi, .smiles, or .txt file.

    Supported examples
    ------------------
    One column:

        CCO
        CCN
        c1ccccc1

    Two columns:

        CCO ethanol
        CCN ethylamine
    """

    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

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

            smiles = (
                parts[0]
                .strip()
            )

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


# ============================================================
# Generic inference input loader
# ============================================================

def load_inference_input(
    *,
    smiles: str | None,
    input_path: str | Path | None,
    structure_column: str,
) -> pd.DataFrame:
    """
    Load molecular structures from either:

        --smiles

    or:

        --input

    Supported files:

        .smi
        .smiles
        .txt
        .csv
        .parquet
        .pq
    """

    # --------------------------------------------------------
    # Validate input source
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


    # ========================================================
    # Single SMILES
    # ========================================================

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


    # ========================================================
    # File input
    # ========================================================

    path = (
        Path(input_path)
        .expanduser()
        .resolve()
    )

    if not path.is_file():

        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    suffix = (
        path.suffix.lower()
    )

    if suffix not in SUPPORTED_INPUT_SUFFIXES:

        raise ValueError(
            f"Unsupported input format: {suffix}\n"
            f"Supported formats: "
            f"{sorted(SUPPORTED_INPUT_SUFFIXES)}"
        )


    # --------------------------------------------------------
    # SMILES text file
    # --------------------------------------------------------

    if suffix in {
        ".smi",
        ".smiles",
        ".txt",
    }:

        frame = read_smi_file(
            path=path,
            structure_column=structure_column,
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


    # ========================================================
    # Validate structure column
    # ========================================================

    if structure_column not in frame.columns:

        raise KeyError(
            f"Structure column "
            f"'{structure_column}' "
            f"not found.\n"
            f"Available columns: "
            f"{frame.columns.tolist()}"
        )

    frame = frame.copy()

    frame[structure_column] = (
        frame[structure_column]
        .astype("string")
        .str.strip()
    )


    # --------------------------------------------------------
    # Check empty structures
    # --------------------------------------------------------

    invalid = (
        frame[structure_column].isna()
        |
        frame[structure_column].eq("")
    )

    if invalid.any():

        invalid_indices = (
            frame.index[
                invalid
            ]
            .tolist()
        )

        raise ValueError(
            f"Found {int(invalid.sum())} "
            f"empty structures in column "
            f"'{structure_column}'.\n"
            f"Example row indices: "
            f"{invalid_indices[:10]}"
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
    """
    Save prediction results.

    Supported output formats:

        .csv
        .parquet
        .pq
    """

    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

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
            "Output must end with "
            ".csv, .parquet, or .pq.\n"
            f"Received: {output_path}"
        )

    return output_path


# ============================================================
# Traditional ML prediction
# ============================================================

def predict_ml(
    args,
) -> None:
    """
    Run inference using a saved ChemFlow traditional
    machine-learning model package.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "CHEMFLOW TRADITIONAL ML PREDICTION"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # Input
    # ========================================================

    input_frame = load_inference_input(
        smiles=args.smiles,
        input_path=args.input,
        structure_column=args.structure_column,
    )

    print(
        f"\nInput molecules: "
        f"{len(input_frame):,}"
    )


    # ========================================================
    # Load trained model
    # ========================================================

    predictor = ChemFlowPredictor(
        model_path=args.model
    )


    # ========================================================
    # Display model information
    # ========================================================

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


    # ========================================================
    # Predict
    # ========================================================

    result_frame = (
        predictor.predict_from_dataframe(
            df=input_frame,
            smiles_col=args.structure_column,
        )
    )


    # ========================================================
    # Rename prediction column
    # ========================================================

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


    # ========================================================
    # Save
    # ========================================================

    output_path = save_prediction_frame(
        frame=result_frame,
        output_path=args.output,
    )


    # ========================================================
    # Summary
    # ========================================================

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
        f"Input molecules: "
        f"{len(input_frame):,}"
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
# CheMeleon prediction
# ============================================================

def predict_chemeleon(args) -> None:
    from chemflow.deep_learning.chemeleon.predictor import CheMeleonPredictor

    input_frame = load_inference_input(
        smiles=args.smiles,
        input_path=args.input,
        structure_column=args.structure_column,
    )
    predictor = CheMeleonPredictor(
        checkpoint_path=args.model_checkpoint,
        device=args.device,
        threshold=args.threshold,
        applicability_domain=args.applicability_domain,
        embedding_dimensions=args.embedding_dimensions,
        calibration_confidence=args.calibration_confidence,
        similarity_radius=args.similarity_radius,
        similarity_bits=args.similarity_bits,
        mc_dropout_samples=args.mc_dropout_samples,
    )
    task_names = (
        [name.strip() for name in args.task_names]
        if args.task_names
        else predictor.target_names
    )
    prediction_data = predictor.predict_smiles(
        input_frame[args.structure_column].astype(str).tolist(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task_names=task_names,
    )
    prediction_frame = pd.DataFrame(prediction_data)
    duplicate_columns = set(input_frame.columns) & set(prediction_frame.columns)
    if duplicate_columns:
        raise ValueError(
            "Prediction output columns already exist in the input: "
            f"{sorted(duplicate_columns)}"
        )
    result_frame = pd.concat(
        [input_frame.reset_index(drop=True), prediction_frame], axis=1
    )
    output_path = save_prediction_frame(result_frame, args.output)
    print("\n" + "=" * 70)
    print("CHEMELEON PREDICTION COMPLETE")
    print("=" * 70)
    print(f"Input molecules: {len(input_frame):,}")
    print(f"Invalid molecules: {len(predictor.invalid_indices):,}")
    print(f"Prediction columns: {prediction_frame.columns.tolist()}")
    print(f"Output: {output_path}")


def predict_hf_graphormer(args) -> None:
    from chemflow.deep_learning.hf_graphormer.predictor import (
        HuggingFaceGraphormerPredictor,
    )

    input_frame = load_inference_input(
        smiles=args.smiles,
        input_path=args.input,
        structure_column=args.structure_column,
    )
    predictor = HuggingFaceGraphormerPredictor(
        args.model_directory,
        device=args.device,
        threshold=args.threshold,
        applicability_domain=args.applicability_domain,
        embedding_dimensions=args.embedding_dimensions,
        calibration_confidence=args.calibration_confidence,
        similarity_radius=args.similarity_radius,
        similarity_bits=args.similarity_bits,
        mc_dropout_samples=args.mc_dropout_samples,
    )
    task_names = args.task_names or predictor.target_names
    values = predictor.predict_smiles(
        input_frame[args.structure_column].astype(str).tolist(),
        batch_size=args.batch_size,
        task_names=task_names,
    )
    prediction_frame = pd.DataFrame(values)
    duplicate_columns = set(input_frame.columns) & set(prediction_frame.columns)
    if duplicate_columns:
        raise ValueError(
            "Prediction output columns already exist in the input: "
            f"{sorted(duplicate_columns)}. Use --task-names to rename outputs."
        )
    result = pd.concat(
        [input_frame.reset_index(drop=True), prediction_frame], axis=1
    )
    output_path = save_prediction_frame(result, args.output)
    print("\n" + "=" * 70)
    print("HF GRAPHORMER PREDICTION COMPLETE")
    print("=" * 70)
    print(f"Input molecules: {len(input_frame):,}")
    print(f"Invalid/oversized molecules: {len(predictor.invalid_indices):,}")
    print(f"MC-dropout passes: {args.mc_dropout_samples}")
    print(f"Prediction columns: {prediction_frame.columns.tolist()}")
    print(f"Output: {output_path}")


def predict_chemberta(args) -> None:
    from chemflow.deep_learning.chemberta.predictor import ChemBERTaPredictor

    input_frame = load_inference_input(
        smiles=args.smiles, input_path=args.input,
        structure_column=args.structure_column,
    )
    predictor = ChemBERTaPredictor(
        args.model_directory, device=args.device, threshold=args.threshold,
        mc_dropout_samples=args.mc_dropout_samples,
    )
    values = predictor.predict_smiles(
        input_frame[args.structure_column].astype(str).tolist(),
        batch_size=args.batch_size, max_length=args.max_length,
    )
    result = pd.concat([input_frame.reset_index(drop=True), pd.DataFrame(values)], axis=1)
    output_path = save_prediction_frame(result, args.output)
    print(f"ChemBERTa predictions saved to {output_path}")


def predict_kermt(args) -> None:
    """Run inference with one or more trained KERMT checkpoints."""
    from chemflow.deep_learning.kermt.vendor.task import predict as kermt_predict

    input_frame = load_inference_input(
        smiles=args.smiles,
        input_path=args.input,
        structure_column=args.structure_column,
    )

    if args.checkpoint_path:
        checkpoint_paths = [
            str(Path(args.checkpoint_path).expanduser().resolve())
        ]
        if not Path(checkpoint_paths[0]).is_file():
            raise FileNotFoundError(
                f"KERMT checkpoint not found: {checkpoint_paths[0]}"
            )
    else:
        checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"KERMT checkpoint directory not found: {checkpoint_dir}"
            )
        checkpoint_paths = sorted(
            str(path)
            for path in checkpoint_dir.rglob("*.pt")
            if path.name != "last_checkpoint.pt"
        )
        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No KERMT model checkpoints found in: {checkpoint_dir}"
            )

    if args.device == "cuda" and not kermt_predict.torch.cuda.is_available():
        raise RuntimeError("KERMT requested CUDA, but CUDA is unavailable.")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and kermt_predict.torch.cuda.is_available()
    )

    vendor_args = argparse.Namespace(
        checkpoint_paths=checkpoint_paths,
        checkpoint_path=None,
        checkpoint_dir=None,
        cuda=use_cuda,
        gpu=0 if use_cuda else None,
        batch_size=args.batch_size,
        data_path=None,
        use_compound_names=False,
        fingerprint=False,
    )
    predictions, _ = kermt_predict.make_predictions(
        vendor_args,
        smiles=input_frame[args.structure_column].astype(str).tolist(),
    )
    checkpoint_task_names = list(vendor_args.task_names)
    uncertainty = getattr(vendor_args, "prediction_uncertainty", None)
    if uncertainty is None:
        prediction_frame = pd.DataFrame(
            predictions,
            columns=checkpoint_task_names,
        )
    else:
        direct = np.asarray(uncertainty["direct"], dtype=float)
        mc_mean = np.asarray(uncertainty["mc_mean"], dtype=float)
        mc_std = np.asarray(uncertainty["mc_std"], dtype=float)
        prediction_frame = pd.DataFrame(
            {
                **{
                    f"direct_{name}": direct[:, index]
                    for index, name in enumerate(checkpoint_task_names)
                },
                **{
                    f"mc_mean_{name}": mc_mean[:, index]
                    for index, name in enumerate(checkpoint_task_names)
                },
                **{
                    f"mc_std_{name}": mc_std[:, index]
                    for index, name in enumerate(checkpoint_task_names)
                },
                **{
                    name: mc_mean[:, index]
                    for index, name in enumerate(checkpoint_task_names)
                },
            }
        )
        coverage_label = int(round(args.calibration_confidence * 100))
        z_value = NormalDist().inv_cdf(
            0.5 + args.calibration_confidence / 2.0
        )
        for index, name in enumerate(checkpoint_task_names):
            prediction_frame[f"{name}_mc_lower_{coverage_label}"] = (
                mc_mean[:, index] - z_value * mc_std[:, index]
            )
            prediction_frame[f"{name}_mc_upper_{coverage_label}"] = (
                mc_mean[:, index] + z_value * mc_std[:, index]
            )

    if args.calibration_input:
        calibration_frame = load_inference_input(
            smiles=None,
            input_path=args.calibration_input,
            structure_column=args.structure_column,
        )
        missing_targets = [
            name for name in checkpoint_task_names
            if name not in calibration_frame.columns
        ]
        if missing_targets:
            raise KeyError(
                "Calibration input is missing KERMT target columns: "
                f"{missing_targets}"
            )
        calibration_predictions, _ = kermt_predict.make_predictions(
            vendor_args,
            smiles=calibration_frame[args.structure_column].astype(str).tolist(),
        )
        calibration_uncertainty = getattr(
            vendor_args, "prediction_uncertainty", None
        )
        calibration_values = np.asarray(
            calibration_uncertainty["mc_mean"]
            if calibration_uncertainty is not None
            else calibration_predictions,
            dtype=float,
        )
        prediction_values = np.asarray(
            uncertainty["mc_mean"] if uncertainty is not None else predictions,
            dtype=float,
        )
        for index, name in enumerate(checkpoint_task_names):
            targets = pd.to_numeric(
                calibration_frame[name], errors="coerce"
            ).to_numpy(dtype=float)
            values = calibration_values[:, index]
            valid = np.isfinite(targets) & np.isfinite(values)
            if not valid.any():
                raise ValueError(
                    f"Calibration target '{name}' has no finite labeled rows."
                )
            bias = float(np.median(targets[valid] - values[valid]))
            residuals = np.abs(targets[valid] - values[valid] - bias)
            radius = float(
                np.quantile(
                    residuals,
                    min(
                        1.0,
                        np.ceil((len(residuals) + 1) * args.calibration_confidence)
                        / len(residuals),
                    ),
                    method="higher",
                )
            )
            calibrated = prediction_values[:, index] + bias
            prediction_frame[f"calibrated_{name}"] = calibrated
            prediction_frame[f"{name}_lower_{coverage_label}"] = calibrated - radius
            prediction_frame[f"{name}_upper_{coverage_label}"] = calibrated + radius

    if args.task_names:
        if len(args.task_names) != len(checkpoint_task_names):
            raise ValueError(
                "--task-names must contain one name per KERMT target "
                f"({len(checkpoint_task_names)} expected)."
            )
        rename_map = dict(zip(checkpoint_task_names, args.task_names))
        prediction_frame.rename(columns=rename_map, inplace=True)

    duplicate_columns = set(input_frame.columns) & set(prediction_frame.columns)
    if duplicate_columns:
        raise ValueError(
            "Prediction output columns already exist in the input: "
            f"{sorted(duplicate_columns)}. Use --task-names to rename outputs."
        )
    result_frame = pd.concat(
        [input_frame.reset_index(drop=True), prediction_frame], axis=1
    )
    output_path = save_prediction_frame(result_frame, args.output)
    print("\n" + "=" * 70)
    print("KERMT PREDICTION COMPLETE")
    print("=" * 70)
    print(f"Input molecules: {len(input_frame):,}")
    print(f"Checkpoint models: {len(checkpoint_paths):,}")
    print(f"Prediction columns: {prediction_frame.columns.tolist()}")
    print(f"Output: {output_path}")


# ============================================================
# Traditional ML parser
# ============================================================

def add_ml_predict_parser(
    model_subparsers,
) -> None:
    """
    Register:

        chemflow predict ml
    """

    ml_parser = (
        model_subparsers.add_parser(
            "ml",
            help=(
                "Run traditional ML inference "
                "using a saved ChemFlow .pkl model."
            ),
        )
    )


    # ========================================================
    # Input
    # ========================================================

    input_group = (
        ml_parser.add_mutually_exclusive_group(
            required=True
        )
    )

    input_group.add_argument(
        "--smiles",
        type=str,
        default=None,
        help=(
            "Predict one SMILES string."
        ),
    )

    input_group.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Input .smi, .smiles, .txt, "
            ".csv, .parquet, or .pq file."
        ),
    )


    # ========================================================
    # Structure column
    # ========================================================

    ml_parser.add_argument(
        "--structure-column",
        type=str,
        default="SMILES",
        help=(
            "Column containing SMILES. "
            "Default: SMILES."
        ),
    )


    # ========================================================
    # Model
    # ========================================================

    ml_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Path to trained ML .pkl model "
            "or ChemFlow model package."
        ),
    )


    # ========================================================
    # Prediction name
    # ========================================================

    ml_parser.add_argument(
        "--task-name",
        type=str,
        default="prediction",
        help=(
            "Name of output prediction column. "
            "Default: prediction."
        ),
    )


    # ========================================================
    # Output
    # ========================================================

    ml_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output .csv, .parquet, or .pq file."
        ),
    )


    # ========================================================
    # Dispatcher
    # ========================================================

    ml_parser.set_defaults(
        func=predict_ml
    )


# ============================================================
# CheMeleon parser
# ============================================================

def add_chemeleon_predict_parser(model_subparsers) -> None:
    parser = model_subparsers.add_parser(
        "chemeleon",
        help="Run single-task or multitask CheMeleon inference.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", type=str, default=None)
    input_group.add_argument("--input", type=str, default=None)
    parser.add_argument("--structure-column", type=str, default="SMILES")
    parser.add_argument(
        "--task-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional output task names; defaults to checkpoint target columns.",
    )
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--applicability-domain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calculate checkpoint-contained calibration and domain diagnostics.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=None,
        help="Number of stored PCA embedding dimensions to use (default: all).",
    )
    parser.add_argument("--calibration-confidence", type=float, default=0.90)
    parser.add_argument("--similarity-radius", type=int, default=2)
    parser.add_argument("--similarity-bits", type=int, default=2048)
    parser.add_argument(
        "--mc-dropout-samples",
        type=int,
        default=0,
        help=(
            "Stochastic inference passes used to estimate epistemic uncertainty; "
            "use 20-50 with a checkpoint trained with nonzero dropout (default: 0)."
        ),
    )
    parser.add_argument("--output", type=str, required=True)
    parser.set_defaults(func=predict_chemeleon)


def add_hf_graphormer_predict_parser(model_subparsers) -> None:
    parser = model_subparsers.add_parser(
        "hf-graphormer",
        help="Run calibrated Graphormer inference with optional MC dropout.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", type=str, default=None)
    input_group.add_argument("--input", type=str, default=None)
    parser.add_argument("--structure-column", default="SMILES")
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--model-directory", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--applicability-domain",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--embedding-dimensions", type=int, default=None)
    parser.add_argument("--calibration-confidence", type=float, default=0.90)
    parser.add_argument("--similarity-radius", type=int, default=2)
    parser.add_argument("--similarity-bits", type=int, default=2048)
    parser.add_argument(
        "--mc-dropout-samples",
        type=int,
        default=0,
        help="Use 20-50 stochastic passes with a checkpoint containing dropout.",
    )
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=predict_hf_graphormer)


def add_chemberta_predict_parser(model_subparsers) -> None:
    parser = model_subparsers.add_parser(
        "chemberta", help="Run inference with a trained ChemFlow ChemBERTa model."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", type=str, default=None)
    input_group.add_argument("--input", type=str, default=None)
    parser.add_argument("--structure-column", default="SMILES")
    parser.add_argument("--model-directory", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mc-dropout-samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=predict_chemberta)


def add_kermt_predict_parser(model_subparsers) -> None:
    parser = model_subparsers.add_parser(
        "kermt", help="Run inference with trained KERMT checkpoints."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", type=str, default=None)
    input_group.add_argument("--input", type=str, default=None)
    parser.add_argument("--structure-column", default="SMILES")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--model-checkpoint")
    checkpoint_group.add_argument("--checkpoint-dir")
    checkpoint_group.add_argument("--checkpoint-path")
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=None,
        help="Optional output names; defaults to names stored in the checkpoint.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--mc-dropout-samples",
        type=int,
        default=0,
        help="Use at least 2 stochastic passes to emit MC mean/std intervals.",
    )
    parser.add_argument(
        "--calibration-input",
        default=None,
        help="Labeled CSV/parquet used for validation residual calibration.",
    )
    parser.add_argument(
        "--calibration-confidence",
        type=float,
        default=0.90,
        help="Prediction interval coverage, between 0 and 1 (default: 0.90).",
    )
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=predict_kermt)


# ============================================================
# Main predict parser
# ============================================================

def add_predict_parser(
    subparsers,
) -> None:
    """
    Register:

        chemflow predict

    with:

        chemflow predict ml
        chemflow predict chemeleon
    """

    predict_parser = (
        subparsers.add_parser(
            "predict",
            help=(
                "Run model inference."
            ),
        )
    )

    model_subparsers = (
        predict_parser.add_subparsers(
            dest="predict_model",
            required=True,
        )
    )


    # --------------------------------------------------------
    # Traditional ML
    # --------------------------------------------------------

    add_ml_predict_parser(
        model_subparsers
    )


    # --------------------------------------------------------
    # CheMeleon
    # --------------------------------------------------------

    add_chemeleon_predict_parser(
        model_subparsers
    )
    add_hf_graphormer_predict_parser(model_subparsers)
    add_chemberta_predict_parser(model_subparsers)
    add_kermt_predict_parser(model_subparsers)
