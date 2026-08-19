from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.chemflow.uncertainty.metrics import METRICS
from src.chemflow.uncertainty.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    rmse_score,
    pearson_score,
    spearman_score,
    kendall_score,
)
from src.chemflow.uncertainty.bootstraping import (
    bootstrap_metric,
    BootstrapResult,
    evaluate_multitask_bootstrap,
)
from src.chemflow.uncertainty.plot import (
    plot_multitask_metrics,
    plot_bootstrap_distribution,
)


# ============================================================
# Parse --task CLI specifications
# ============================================================

def parse_task_columns(
    task_specs: list[str],
) -> dict[
    str,
    tuple[str, str],
]:

    task_columns: dict[
        str,
        tuple[str, str],
    ] = {}

    for spec in task_specs:
        parts = [
            value.strip()
            for value in spec.split(":")
        ]

        if len(parts) != 3:
            raise ValueError(
                f"Invalid --task specification '{spec}'. "
                "Expected format: "
                "task_name:target_column:prediction_column"
            )

        (
            task_name,
            target_column,
            prediction_column,
        ) = parts

        if not task_name:
            raise ValueError(
                "Task name cannot be empty."
            )

        if not target_column:
            raise ValueError(
                f"Target column cannot be empty "
                f"for task '{task_name}'."
            )

        if not prediction_column:
            raise ValueError(
                f"Prediction column cannot be empty "
                f"for task '{task_name}'."
            )

        if task_name in task_columns:
            raise ValueError(
                f"Duplicate task name: '{task_name}'."
            )

        task_columns[
            task_name
        ] = (
            target_column,
            prediction_column,
        )

    return task_columns


# ============================================================
# Load prediction file
# ============================================================

def load_prediction_file(
    input_path: Path | str,
) -> pd.DataFrame:

    input_path = (
        Path(
            input_path
        )
        .expanduser()
        .resolve()
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {input_path}"
        )

    suffix = (
        input_path
        .suffix
        .lower()
    )

    if suffix == ".csv":
        return pd.read_csv(
            input_path
        )

    if suffix in {
        ".parquet",
        ".pq",
    }:
        return pd.read_parquet(
            input_path
        )

    raise ValueError(
        f"Unsupported prediction file format: '{suffix}'."
    )




# ============================================================
# CLI runner
# ============================================================

def run_multitask_bootstrap_evaluation(
    args,
) -> pd.DataFrame:

    output_dir = (
        Path(
            args.output_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load predictions
    # --------------------------------------------------------

    frame = load_prediction_file(
        args.input
    )

    # --------------------------------------------------------
    # 2. Parse task definitions
    # --------------------------------------------------------

    task_columns = parse_task_columns(
        args.task
    )

    # --------------------------------------------------------
    # 3. Bootstrap evaluation
    # --------------------------------------------------------

    (
        metrics_df,
        distributions,
    ) = evaluate_multitask_bootstrap(
        frame=frame,
        task_columns=task_columns,
        metrics=args.metrics,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )

    if metrics_df.empty:
        raise RuntimeError(
            "No valid multi-task evaluation "
            "results were generated."
        )

    # --------------------------------------------------------
    # 4. Save metric table
    # --------------------------------------------------------

    summary_path = (
        output_dir
        / "multitask_bootstrap_metrics.csv"
    )

    metrics_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # 5. ONE figure containing all tasks × all metrics
    # --------------------------------------------------------

    plot_multitask_metrics(
        metrics_df=metrics_df,
        confidence_level=args.confidence_level,
        metric_order=args.metrics,
        output_path=(
            output_dir
            / "multitask_metrics.png"
        ),
    )

    # --------------------------------------------------------
    # 6. Optional individual distributions
    # --------------------------------------------------------

    if args.plot_distributions:
        distribution_dir = (
            output_dir
            / "distributions"
        )

        distribution_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for _, row in metrics_df.iterrows():
            task = str(
                row[
                    "task"
                ]
            )

            metric = str(
                row[
                    "metric"
                ]
            )

            plot_bootstrap_distribution(
                distributions[
                    task
                ][
                    metric
                ],
                task=task,
                metric=metric,
                observed=float(
                    row[
                        "observed"
                    ]
                ),
                ci_lower=float(
                    row[
                        "ci_lower"
                    ]
                ),
                ci_upper=float(
                    row[
                        "ci_upper"
                    ]
                ),
                output_path=(
                    distribution_dir
                    / f"{task}_{metric}.png"
                ),
            )

    return metrics_df


# ============================================================
# CLI
# ============================================================

def add_uncertainty_parser(
    subparsers,
) -> None:

    uncertainty_parser = (
        subparsers.add_parser(
            "uncertainty",
            help=(
                "Model and metric "
                "uncertainty analysis"
            ),
        )
    )

    uncertainty_subparsers = (
        uncertainty_parser
        .add_subparsers(
            dest="uncertainty_type",
            required=True,
        )
    )

    # ========================================================
    # Multi-task bootstrap
    # ========================================================

    bootstrap_parser = (
        uncertainty_subparsers
        .add_parser(
            "bootstrap",
            help=(
                "Bootstrap uncertainty "
                "evaluation for multi-task predictions"
            ),
        )
    )

    bootstrap_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help=(
            "Prediction file "
            "(.csv, .parquet, or .pq)"
        ),
    )

    bootstrap_parser.add_argument(
        "--task",
        action="append",
        required=True,
        help=(
            "Task specification: "
            "task_name:target_column:prediction_column. "
            "Provide --task multiple times."
        ),
    )

    bootstrap_parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[
            "r2",
            "mae",
            "rmse",
            "pearson",
            "spearman",
            "kendall",
        ],
        default=[
            "mae",
            "rmse",
            "r2",
            "pearson",
            "spearman",
            "kendall",
        ],
        help=(
            "Metrics to evaluate."
        ),
    )

    bootstrap_parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help=(
            "Number of bootstrap resamples "
            "(default: 1000)"
        ),
    )

    bootstrap_parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help=(
            "Bootstrap confidence level "
            "(default: 0.95)"
        ),
    )

    bootstrap_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed "
            "(default: 42)"
        ),
    )

    bootstrap_parser.add_argument(
        "--plot-distributions",
        action="store_true",
        help=(
            "Also generate an individual "
            "bootstrap distribution for each "
            "task/metric combination."
        ),
    )

    bootstrap_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output directory."
        ),
    )

    bootstrap_parser.set_defaults(
        func=run_multitask_bootstrap_evaluation
    )