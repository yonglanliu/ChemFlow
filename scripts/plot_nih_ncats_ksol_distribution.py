#!/usr/bin/env python3
"""Plot NIH and NCATS solubility distributions used in the combined endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


SOURCES = {
    "NIH_PubChem_AID_1996": ("NIH AID 1996", "#2563EB"),
    "NCATS_AID_1645848": ("NCATS AID 1645848", "#EA580C"),
}


def _exact_values(frame: pd.DataFrame, source: str) -> np.ndarray:
    mask = frame["source"].eq(source) & frame["KSOL_pH7_4_qualifier"].eq("=")
    values = pd.to_numeric(
        frame.loc[mask, "Log_KSOL_pH7_4_mol_L"], errors="coerce"
    ).to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    cumulative = np.arange(1, len(ordered) + 1, dtype=float) / len(ordered)
    return ordered, cumulative


def plot(source_path: Path, output_path: Path, pdf_path: Path | None) -> None:
    frame = pd.read_csv(source_path, low_memory=False)
    distributions = {
        source: _exact_values(frame, source)
        for source in SOURCES
    }
    if any(len(values) == 0 for values in distributions.values()):
        empty = [source for source, values in distributions.items() if len(values) == 0]
        raise ValueError(f"No exact solubility measurements found for: {empty}")

    pooled = np.concatenate(list(distributions.values()))
    lower = np.floor(pooled.min() * 2) / 2
    upper = np.ceil(pooled.max() * 2) / 2
    bins = np.linspace(lower, upper, 54)
    grid = np.linspace(lower, upper, 600)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#243447",
            "axes.edgecolor": "#94A3B8",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    fig, (density_ax, ecdf_ax) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.4),
        gridspec_kw={"width_ratios": [1.25, 1]},
    )
    fig.patch.set_facecolor("#F8FAFC")

    for axis in (density_ax, ecdf_ax):
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)

    summary_rows = []
    for source, values in distributions.items():
        label, color = SOURCES[source]
        median = float(np.median(values))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        summary_rows.append(
            {
                "source": label,
                "n": len(values),
                "median_log10_mol_L": median,
                "q1_log10_mol_L": float(q1),
                "q3_log10_mol_L": float(q3),
                "minimum_log10_mol_L": float(values.min()),
                "maximum_log10_mol_L": float(values.max()),
            }
        )

        density_ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.20,
            color=color,
            edgecolor="none",
            label=f"{label} (n={len(values):,})",
            zorder=2,
        )
        kde = gaussian_kde(values)
        density_ax.plot(grid, kde(grid), color=color, linewidth=2.4, zorder=3)
        density_ax.axvline(
            median,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            zorder=3,
        )

        x_values, y_values = _ecdf(values)
        ecdf_ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=2.2,
            label=f"{label} (median {median:.2f})",
            zorder=3,
        )

    density_ax.set_title("Normalized density")
    density_ax.set_xlabel(r"Kinetic solubility, $\log_{10}$(mol/L)")
    density_ax.set_ylabel("Probability density")
    density_ax.set_xlim(lower, upper)
    density_ax.legend(frameon=False, loc="upper left")

    ecdf_ax.set_title("Empirical cumulative distribution")
    ecdf_ax.set_xlabel(r"Kinetic solubility, $\log_{10}$(mol/L)")
    ecdf_ax.set_ylabel("Cumulative fraction")
    ecdf_ax.set_xlim(lower, upper)
    ecdf_ax.set_ylim(0, 1.01)
    ecdf_ax.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "NIH and NCATS contributions to the combined pH 7.4 solubility endpoint",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Exact, uncensored source measurements. Dashed lines indicate source medians; "
        "13 cross-source duplicate structures are averaged only in the final consolidated table.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#64748B",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.93))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved plot: {output_path}")
    if pdf_path is not None:
        print(f"Saved vector plot: {pdf_path}")
    print(f"Saved summary: {summary_path}")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repository / "dataset/ADMET/grouped/physchem/physchem_multitask.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "dataset/curated/NIH_NCATS_KSOL_distribution.png",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=repository / "dataset/curated/NIH_NCATS_KSOL_distribution.pdf",
    )
    args = parser.parse_args()
    plot(args.source, args.output, args.pdf)


if __name__ == "__main__":
    main()
