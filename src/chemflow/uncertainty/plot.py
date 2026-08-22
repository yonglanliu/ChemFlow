from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# One figure containing all tasks × all metrics
# ============================================================

def plot_multitask_metrics(
    metrics_df: pd.DataFrame,
    output_path: Path | str,
    *,
    confidence_level: float = 0.95,
    metric_order: list[str] | None = None,
) -> Path:
    """
    Plot all tasks and metrics in one figure.

    Each task is shown as a separate point series.
    Error bars represent bootstrap confidence intervals.
    """

    required_columns = {
        "task",
        "metric",
        "observed",
        "ci_lower",
        "ci_upper",
    }

    missing = (
        required_columns
        - set(
            metrics_df.columns
        )
    )

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = (
        metrics_df
        .copy()
    )

    frame[
        "metric"
    ] = (
        frame[
            "metric"
        ]
        .astype(str)
        .str.lower()
    )

    # --------------------------------------------------------
    # Metric ordering
    # --------------------------------------------------------

    if metric_order is None:
        default_order = [
            "mae",
            "rmse",
            "r2",
            "pearson",
            "spearman",
            "kendall",
        ]

        available_metrics = set(
            frame[
                "metric"
            ]
        )

        metrics = [
            metric
            for metric in default_order
            if metric in available_metrics
        ]

        # Append custom metrics not present in default_order.
        metrics.extend(
            metric
            for metric in frame[
                "metric"
            ].drop_duplicates()
            if metric not in metrics
        )

    else:
        metrics = [
            metric.lower()
            for metric in metric_order
            if metric.lower()
            in set(
                frame[
                    "metric"
                ]
            )
        ]

    if not metrics:
        raise ValueError(
            "No metrics available to plot."
        )

    tasks = (
        frame[
            "task"
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if not tasks:
        raise ValueError(
            "No tasks available to plot."
        )

    x = np.arange(
        len(
            metrics
        )
    )

    # --------------------------------------------------------
    # Horizontal separation between tasks
    # --------------------------------------------------------

    total_width = 0.65

    offset_step = (
        total_width
        / max(
            len(tasks),
            1,
        )
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for task_index, task in enumerate(
        tasks
    ):
        task_frame = (
            frame[
                frame[
                    "task"
                ].astype(str).eq(
                    task
                )
            ]
            .set_index(
                "metric"
            )
            .reindex(
                metrics
            )
        )

        values = task_frame[
            "observed"
        ].to_numpy(
            dtype=float
        )

        lower = task_frame[
            "ci_lower"
        ].to_numpy(
            dtype=float
        )

        upper = task_frame[
            "ci_upper"
        ].to_numpy(
            dtype=float
        )

        valid = (
            np.isfinite(
                values
            )
            & np.isfinite(
                lower
            )
            & np.isfinite(
                upper
            )
        )

        lower_error = (
            values
            - lower
        )

        upper_error = (
            upper
            - values
        )

        offset = (
            task_index
            - (
                len(tasks) - 1
            ) / 2
        ) * offset_step

        task_x = (
            x
            + offset
        )

        if valid.any():
            ax.errorbar(
                task_x[
                    valid
                ],
                values[
                    valid
                ],
                yerr=np.vstack(
                    [
                        lower_error[
                            valid
                        ],
                        upper_error[
                            valid
                        ],
                    ]
                ),
                fmt="o",
                capsize=5,
                markersize=7,
                linewidth=1.5,
                label=task,
            )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(
        x,
    )
    ax.tick_params(
        axis='x',
        labelsize=16,
    )
    ax.tick_params(
        axis='y',
        labelsize=16,
    )

    ax.set_xticklabels(
        [
            metric.upper()
            for metric in metrics
        ]
    )

    # ax.set_xlabel(
    #     "Evaluation metric",
    # )

    ax.set_ylabel(
        "Metric value",
        fontsize=20,
    )
    ax.set_ylim(0.0, 1.0)

    ci_percentage = int(
        round(
            confidence_level
            * 100
        )
    )

    ax.set_title(
        "Multi-task model performance "
        f"with {ci_percentage}% bootstrap confidence intervals",
        fontsize=20,
        pad=18,
    )

    # --------------------------------------------------------
    # Separate prediction-error metrics from correlation metrics
    # --------------------------------------------------------

    error_metrics = {
        "mae",
        "rmse",
    }

    error_positions = [
        index
        for index, metric in enumerate(
            metrics
        )
        if metric in error_metrics
    ]

    correlation_positions = [
        index
        for index, metric in enumerate(
            metrics
        )
        if metric not in error_metrics
    ]

    if (
        error_positions
        and correlation_positions
        and max(
            error_positions
        )
        < min(
            correlation_positions
        )
    ):
        boundary = (
            max(
                error_positions
            )
            + min(
                correlation_positions
            )
        ) / 2

        ax.axvline(
            boundary,
            linestyle="--",
            alpha=0.4,
        )

        error_center = np.mean(
            error_positions
        )

        correlation_center = np.mean(
            correlation_positions
        )

        ymin, ymax = ax.get_ylim()

        text_y = (
            ymax
            + 0.02
            * (
                ymax - ymin
            )
        )

        # ax.text(
        #     error_center,
        #     text_y,
        #     "Error metrics",
        #     ha="center",
        #     va="bottom",
        # )

        # ax.text(
        #     correlation_center,
        #     text_y,
        #     "Correlation / ranking metrics",
        #     ha="center",
        #     va="bottom",
        # )

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        title="Task",
        frameon=True,
        fontsize=16,
        title_fontsize=16,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    fig.subplots_adjust(right=0.80)

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# Individual bootstrap distribution plot
# ============================================================

def plot_bootstrap_distribution(
    samples: np.ndarray,
    *,
    metric: str,
    task: str,
    observed: float,
    ci_lower: float,
    ci_upper: float,
    output_path: Path | str,
) -> Path:

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mean = np.mean(samples)
    median = np.median(samples)

    q1 = np.percentile(samples, 25)
    q3 = np.percentile(samples, 75)

    fig = plt.figure(
        figsize=(8,6)
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1,4],
        hspace=0.05,
    )

    ax_box = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(
        gs[1],
        sharex=ax_box,
    )

    # =====================================================
    # Boxplot
    # =====================================================

    ax_box.boxplot(
        samples,
        vert=False,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )

    ax_box.set_yticks([])
    ax_box.set_xlabel("")

    # =====================================================
    # Histogram
    # =====================================================

    ax_hist.hist(
        samples,
        bins=40,
        alpha=0.85,
    )

    # observed
    ax_hist.axvline(
        observed,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Observed = {observed:.3f}",
    )

    # mean
    ax_hist.axvline(
        mean,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean:.3f}",
    )

    # median
    ax_hist.axvline(
        median,
        color="orange",
        linewidth=2,
        label=f"Median = {median:.3f}",
    )

    # confidence interval
    ax_hist.axvline(
        ci_lower,
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"95% CI",
    )

    ax_hist.axvline(
        ci_upper,
        color="green",
        linestyle=":",
        linewidth=2,
    )

    ax_hist.set_xlabel(metric.upper(), fontsize=18)
    ax_hist.set_ylabel("Bootstrap frequency", fontsize=18)
    ax_hist.tick_params(axis='x', labelsize=16)
    ax_hist.tick_params(axis='y', labelsize=16)

    # ax_hist.set_title(
    #     f"{task}: Bootstrap distribution of {metric.upper()}"
    # )

    ax_hist.grid(
        alpha=0.3,
    )

    ax_hist.legend(
        frameon=False,
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path