from __future__ import annotations

from functools import partial
from pathlib import Path
import logging

import pandas as pd

from src.chemflow.chemistry.similarity_search import SimilarityCalculator
from src.chemflow.featurization.create_features import (
    FP_TYPES,
    MACCS_TYPES,
    DESC_TYPES,
    FP_BITS,
)
from src.chemflow.data.config import DataPipelineConfig
from src.chemflow.data.partition import DatasetPartitioner
from src.chemflow.data.execution import WorkerExecutor
from src.chemflow.data.collection import ResultCollector
from src.chemflow.utils.logger import setup_logger


SUPPORTED_INPUT_SUFFIXES = {
    ".smi",
    ".smiles",
    ".txt",
    ".csv",
    ".parquet",
    ".pq",
}


# ============================================================
# Input utilities
# ============================================================

def read_smi_file(path: str | Path,structure_column: str = "SMILES") -> pd.DataFrame:
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
    """

    path = Path(path).expanduser().resolve()

    records: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8") as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)

            smiles = parts[0].strip()

            name = (parts[1].strip() if len(parts) > 1 else f"query_{len(records) + 1}")

            records.append(
                {
                    "molecule_id": name,
                    structure_column: smiles,
                }
            )

    if not records:
        raise ValueError(f"No molecules were found in SMI file: {path}")

    return pd.DataFrame(records)


def load_query_input(
    *,
    smiles: str | None,
    input_path: str | Path | None,
    structure_column: str,
) -> pd.DataFrame:
    """
    Load query structures from either:

    - a single SMILES string
    - a .smi/.smiles/.txt file
    - a CSV file
    - a Parquet file
    """

    if smiles is not None and input_path is not None:
        raise ValueError("Provide either --smiles or --query, not both.")

    if smiles is None and input_path is None:
        raise ValueError("One of --smiles or --query must be provided.")

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
    # Query file
    # ---------------------------------------------------------

    path = (Path(input_path).expanduser().resolve())

    if not path.is_file():
        raise FileNotFoundError(
            f"Query input file does not exist: {path}"
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
            f"Structure column '{structure_column}' "
            f"was not found in {path}. "
            f"Available columns: {list(frame.columns)}"
        )

    frame = frame.copy()

    # ---------------------------------------------------------
    # Add query IDs if necessary
    # ---------------------------------------------------------

    if "molecule_id" not in frame.columns:
        frame["molecule_id"] = [f"query_{i + 1}" for i in range(len(frame))]

    frame[structure_column] = (frame[structure_column].astype("string").str.strip())

    invalid_mask = (frame[structure_column].isna() | frame[structure_column].eq(""))

    if invalid_mask.any():
        invalid_rows = (frame.index[invalid_mask].tolist())

        raise ValueError(
            f"Found {len(invalid_rows)} empty structures "
            f"in column '{structure_column}'. "
            f"Example row indices: {invalid_rows[:10]}"
        )

    return frame.reset_index(drop=True)


# ============================================================
# Database shard utilities
# ============================================================

def load_database_shard(
    path: Path,
    structure_column: str,
) -> pd.DataFrame:
    """Load one database shard generated by DatasetPartitioner."""

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(path)

    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)

    elif suffix in {".smi", ".smiles"}:
        frame = read_smi_file(path, structure_column=structure_column)

    else:
        raise ValueError(f"Unsupported database shard format: {suffix}")

    if structure_column not in frame.columns:
        raise KeyError(
            f"Structure column '{structure_column}' "
            f"not found in {path}. "
            f"Available columns: {list(frame.columns)}"
        )

    return frame


# ============================================================
# Similarity representation
# ============================================================

def build_similarity_calculator(rep_type: str, metric: str, logger: logging.Logger) -> SimilarityCalculator:
    """
    Build a SimilarityCalculator from a ChemFlow representation type.
    """

    rep_type = (str(rep_type).strip().lower())

    metric = (str(metric).strip().lower())

    use_features = (rep_type.startswith("fcfp"))

    radius = 2
    n_bits = 2048

    # ---------------------------------------------------------
    # ECFP / FCFP
    # ---------------------------------------------------------

    if rep_type in FP_TYPES:

        mode = "2d_fingerprint"
        n_bits = FP_BITS[rep_type]

        if rep_type.endswith("4"):
            radius = 2

        elif rep_type.endswith("6"):
            radius = 3

    # ---------------------------------------------------------
    # MACCS
    # ---------------------------------------------------------

    elif rep_type in MACCS_TYPES:

        mode = "2d_fingerprint"
        n_bits = FP_BITS[rep_type]

    # ---------------------------------------------------------
    # Molecular descriptors
    # ---------------------------------------------------------

    elif rep_type in DESC_TYPES:

        mode = "2d_descriptor"

    else:
        supported = (FP_TYPES | MACCS_TYPES | DESC_TYPES)

        logger.error(
            "Unsupported representation type '%s'. "
            "Expected one of %s.",
            rep_type,
            sorted(supported),
        )

        raise ValueError(
            f"Unsupported representation type '{rep_type}'. "
            f"Expected one of {sorted(supported)}."
        )

    return SimilarityCalculator(
        mode=mode,
        metric=metric,
        radius=radius,
        n_bits=n_bits,
        use_features=use_features,
    )


# ============================================================
# Worker job
# ============================================================

def similarity_search_job(
    shard_path: Path,
    output_path: Path,
    *,
    query: str,
    query_id: str,
    structure_column: str,
    rep_type: str,
    metric: str,
    logger: logging.Logger,
) -> Path:
    """
    Run similarity search against one database shard
    and write the worker result to a CSV file.

    Parameters
    ----------
    shard_path
        Input database shard.

    output_path
        Output CSV path assigned by WorkerExecutor.

    query
        Query SMILES.

    query_id
        Query molecule identifier.

    structure_column
        Column containing molecular structures.

    rep_type
        Molecular representation type.

    metric
        Similarity metric.

    Returns
    -------
    Path
        Path to the worker output CSV file.
    """

    database_frame = load_database_shard(
        shard_path,
        structure_column=structure_column,
    )

    calculator = build_similarity_calculator(
        rep_type=rep_type,
        metric=metric,
        logger=logger,
    )

    result = calculator.search_dataframe(
        query=query,
        df=database_frame,
        smiles_col=structure_column,
    )

    result = result.copy()

    result.insert(
        0,
        "query_id",
        query_id,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)

    return output_path


# ============================================================
# Database partitioning
# ============================================================

def partition_database(
    *,
    data_path: Path,
    output_dir: Path,
    output_file_prefix: str,
    num_shards: int,
    logger: logging.Logger,
) -> list[Path]:
    """
    Partition the database once and return shard paths.
    """

    shard_dir = (output_dir / "shards")

    logger.info(
        "Partitioning database '%s' into %d shards.",
        data_path,
        num_shards,
    )

    partitioner = DatasetPartitioner(
        input_path=data_path,
        output_dir=shard_dir,
        num_shards=num_shards,
        output_file_prefix=output_file_prefix,
        logger=logger,
    )

    shard_paths = partitioner.split()

    if not shard_paths:
        raise RuntimeError(
            f"No shards were generated from database: {data_path}"
        )

    logger.info("Generated %d database shards.", len(shard_paths))

    return shard_paths


# ============================================================
# Run one query against all shards
# ============================================================

def run_query_against_shards(
    *,
    query: str,
    query_id: str,
    shard_paths: list[Path],
    structure_column: str,
    rep_type: str,
    metric: str,
    num_workers: int,
    sort_by: str,
    descending: bool,
    top_k_ratio: float | None,
    output_dir: Path, 
    fail_fast: bool = True,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Run one query against all pre-generated database shards.
    """

    logger.info("Running similarity search for query '%s'.", query_id)

    # ---------------------------------------------------------
    # Bind all static arguments.
    #
    # Executor only needs:
    #
    #     job(shard_path)
    #
    # ---------------------------------------------------------

    job = partial(
        similarity_search_job,
        query=query,
        query_id=query_id,
        structure_column=structure_column,
        rep_type=rep_type,
        metric=metric,
        logger=logger
    )

    # ---------------------------------------------------------
    # Execute
    # ---------------------------------------------------------

    executor = WorkerExecutor(num_workers=num_workers, fail_fast=fail_fast)

    worker_results = executor.run(worker_fn=job, inputs=shard_paths, outdir=output_dir, query_name=query_id)

    # ---------------------------------------------------------
    # Collect
    # ---------------------------------------------------------

    collector = ResultCollector(sort_by=sort_by, descending=descending, top_k_ratio=top_k_ratio)

    merged = collector.collect(worker_results)

    logger.info(
        "Similarity search for query '%s' returned %d records.",
        query_id,
        len(merged),
    )

    return merged


# ============================================================
# Similarity search pipeline
# ============================================================

def run_similarity_search_pipeline(args):
    """
    CLI entry point for:

        chemflow search similarity

    Workflow
    --------
    1. Load query molecule(s).
    2. Partition the database once.
    3. Reuse those shards for every query.
    4. Execute similarity search in parallel.
    5. Collect and rank results.
    6. Write one result file per query.
    """

    # ---------------------------------------------------------
    # Resolve paths and configuration
    # ---------------------------------------------------------

    data_path = Path(args.database).expanduser().resolve()
    job_name = str(args.job_name).strip()

    output_dir = Path(job_name)
    output_dir.mkdir(parents=True,exist_ok=True)

    logger = setup_logger(log_dir=output_dir, log_name=f"{args.job_name}.log")

    output_path = Path(output_dir / f"{job_name}-out.csv")

    if not data_path.is_file():
        raise FileNotFoundError(f"Database file does not exist: {data_path}")

    structure_column = getattr(args, "structure_column", "SMILES")

    num_workers = int(args.num_workers) if args.num_workers is not None else 1

    num_shards = int(args.num_shards) if args.num_shards is not None else num_workers

    if num_workers <= 0:
        raise ValueError("num_workers must be greater than 0.")

    if num_shards <= 0:
        raise ValueError("num_shards must be greater than 0.")

    # ---------------------------------------------------------
    # 1. Load query molecules
    # ---------------------------------------------------------

    query_frame = load_query_input(
        smiles=getattr(args, "query_smiles", None),
        input_path=getattr(args, "query_file", None),
        structure_column=structure_column,
    )

    logger.info("Loaded %d query molecule(s).", len(query_frame))

    # ---------------------------------------------------------
    # 2. Partition database ONCE
    # ---------------------------------------------------------

    shard_paths = partition_database(
        data_path=data_path,
        output_dir=output_dir,
        output_file_prefix=f"{job_name}_database",
        num_shards=num_shards,
        logger=logger,
    )

    # ---------------------------------------------------------
    # 3. Process all query molecules
    # ---------------------------------------------------------

    output_paths: list[Path] = []

    for _, query_row in query_frame.iterrows():

        query = str(query_row[structure_column])

        query_id = str(query_row["molecule_id"])

        merged = run_query_against_shards(
            query=query,
            query_id=query_id,
            shard_paths=shard_paths,
            structure_column=structure_column,
            rep_type=args.rep_type,
            metric=args.metric,
            num_workers=num_workers,
            sort_by="similarity_score",
            descending=True,
            top_k_ratio=args.top_k_ratio,
            output_dir=output_dir,
            fail_fast=True,
            logger=logger,
        )

        # -----------------------------------------------------
        # One query -> one output file
        # -----------------------------------------------------

        if len(query_frame) == 1:
            query_output_path = output_path

        else:
            query_output_path = (output_dir / f"{job_name}_{query_id}-out.csv")

        merged.to_csv(query_output_path, index=False)

        output_paths.append(query_output_path)

        logger.info("Wrote similarity results for query '%s' to '%s'.", query_id, query_output_path)

    logger.info("Similarity-search pipeline completed successfully.")

    return output_paths


# ============================================================
# CLI parser
# ============================================================

def add_search_parser(subparsers):
    search_parser = subparsers.add_parser(
        "search",
        help="Molecular search and feature calculation",
    )

    search_subparsers = (
        search_parser.add_subparsers(
            dest="search_type",
            required=True,
        )
    )

    # ========================================================
    # Similarity search
    # ========================================================

    similarity_parser = (
        search_subparsers.add_parser(
            "similarity",
            help="Perform molecular similarity search",
        )
    )

    # --------------------------------------------------------
    # Query input
    # --------------------------------------------------------

    query_group = (
        similarity_parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    query_group.add_argument(
        "-qf",
        "--query_file",
        type=Path,
        default=None,
        help=(
            "Query molecule file "
            "(.smi, .smiles, .csv, .parquet)"
        ),
    )

    query_group.add_argument(
        "-qs",
        "--query_smiles",
        type=str,
        default=None,
        help="Single query molecule as a SMILES string",
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    similarity_parser.add_argument(
        "-d",
        "--database",
        type=Path,
        required=True,
        help="Molecular database to search",
    )

    # --------------------------------------------------------
    # Dataset columns
    # --------------------------------------------------------

    similarity_parser.add_argument(
        "--structure-column",
        type=str,
        default="SMILES",
        help="Column containing molecular SMILES",
    )

    similarity_parser.add_argument(
        "--ligand_id_column",
        type=str,
        default=None,
        help="Column containing database molecule identifiers",
    )

    # --------------------------------------------------------
    # Representation
    # --------------------------------------------------------

    similarity_parser.add_argument(
        "--rep_type",
        type=str,
        choices=[
            "ecfp4",
            "ecfp6",
            "fcfp4",
            "fcfp6",
            "maccs",
            "2d_descriptor",
        ],
        default="ecfp4",
        help=(
            "Molecular representation used "
            "for similarity calculation"
        ),
    )

    # --------------------------------------------------------
    # Metric
    # --------------------------------------------------------

    similarity_parser.add_argument(
        "--metric",
        type=str,
        choices=[
            "tanimoto",
            "dice",
            "cosine",
            "euclidean",
            "manhattan",
            "mcconnaughey",
        ],
        default="tanimoto",
        help="Similarity or distance metric",
    )

    # --------------------------------------------------------
    # Parallel resources
    # --------------------------------------------------------

    similarity_parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes",
    )

    similarity_parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help=(
            "Number of database shards. "
            "Defaults to --num-workers."
        ),
    )

    # --------------------------------------------------------
    # Result collection
    # --------------------------------------------------------

    similarity_parser.add_argument(
        "--top_k_ratio",
        type=float,
        default=None,
        help=(
            "Fraction of highest-ranked results to retain. "
            "For example, 0.1 keeps the top 10%%."
        ),
    )

    # --------------------------------------------------------
    # Job metadata
    # --------------------------------------------------------

    similarity_parser.add_argument(
        "--job_name",
        type=str,
        default="similarity",
        help=(
            "Prefix used for intermediate "
            "and output files"
        ),
    )

    # --------------------------------------------------------
    # Connect CLI -> pipeline
    # --------------------------------------------------------

    similarity_parser.set_defaults(
        func=run_similarity_search_pipeline
    )