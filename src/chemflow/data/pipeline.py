from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Any, Callable

from src.chemflow.data.partition import DatasetPartitioner
from src.chemflow.data.execution import WorkerExecutor
from src.chemflow.data.collection import ResultCollector
from src.chemflow.data.config import DataPipelineConfig

import logging
logger = logging.getLogger(__name__)


def pipeline(config: DataPipelineConfig, job_fn: Callable[[Path, Path], Any]):
    """
    Run the parallel data-processing pipeline.

    Workflow
    --------
    1. Split the input dataset into shard files.
    2. Process shards in parallel using multiple workers.
    3. Collect and merge worker results.
    4. Optionally sort results and return top-k predictions.
    """

    data_path = Path(config.input_path)
    output_dir = Path(config.output_dir)
    output_file_prefix = str(config.output_file_prefix)

    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {data_path}")

    if config.num_workers <= 0:
        raise ValueError("num_workers must be greater than 0.")

    num_shards = (config.num_shards if config.num_shards is not None else config.num_workers)

    if num_shards <= 0:
        raise ValueError("num_shards must be greater than 0.")

    shard_dir = output_dir / "shards"
    result_dir = output_dir / "results"

    shard_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting data processing pipeline.")

    # ============================================================
    # 1. Split dataset
    # ============================================================
    logger.info("Step 1: Splitting dataset into %d shards.", num_shards)

    partitioner = DatasetPartitioner(
        input_path=data_path, 
        output_dir=shard_dir,
        num_shards=num_shards,
        output_file_prefix=output_file_prefix,
        )

    shard_paths = partitioner.split()

    if not shard_paths:
        raise RuntimeError(f"No dataset shards were generated from: {data_path}")
    
    # ============================================================
    # 2. Execute workers
    # ============================================================
    logger.info("Step 2: Launching %d workers for %d shards.", config.num_workers, num_shards)
    executor = WorkerExecutor(
        num_workers=config.num_workers,
        output_dir=result_dir,
        fail_fast=getattr(config, "fail_fast", True)
        )

    worker_results = executor.run(worker_fn=job_fn, inputs=shard_paths)

    # ============================================================
    # 3. Collect results
    # ============================================================
    logger.info("Step 3: Collecting prediction results from %d shards.", num_shards)

    collector = ResultCollector(
        sort_by=getattr(config, "sort_by", None),
        descending=getattr(config, "descending", True),
        top_k_ratio=getattr(config, "top_k_ratio", 0.1)
        )

    merged = collector.collect(worker_results)

    output_path = _save_collection_to_csv(merged, output_dir / f"{output_file_prefix}-out.csv")
    logger.info("Step 4: Saved collected data to '%s'.", output_path)


def _save_collection_to_csv(merged: pd.DataFrame, output_path: Path | str) -> Path:
    """
    Write the merged result to a CSV file.

    Parameters
    ----------
    merged
        Merged prediction results.

    output_path
        Output CSV file.

    Returns
    -------
    Path
        Path to the written CSV file.
    """

    if not isinstance(merged, pd.DataFrame):
        logger.error("Expected a pandas DataFrame, got %s.", type(merged).__name__)
        raise TypeError("merged must be a pandas DataFrame.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing %d records to '%s'.", len(merged), output_path)

    merged.to_csv(output_path, index=False)

    return output_path