"""Analysis of GROMACS ABFE complex and solvent legs."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

from .config import ABFEConfig


def read_restraint_correction(path: Path) -> float:
    """Read PMX's analytical Boresch restraint contribution in kJ/mol."""
    if not path.is_file():
        raise ValueError(f"PMX restraint summary does not exist: {path}")
    match = re.search(r"dG\s+Restraints\s*=\s*([-+0-9.eE]+)\s+kJ/mol", path.read_text())
    if match is None:
        raise ValueError(f"Could not read the restraint correction from {path}")
    return float(match.group(1))


def _analyze_leg(config: ABFEConfig, leg: str, replicate: int) -> tuple[float, float]:
    try:
        from alchemlyb.workflows import ABFE
    except ImportError as error:
        raise ValueError("alchemlyb is required for ABFE analysis") from error

    directory = config.project_dir / leg / f"rep_{replicate}" / "lambda_*"
    files = sorted(directory.parent.glob("lambda_*/prod.xvg"))
    if len(files) != config.lambda_windows:
        raise ValueError(
            f"Expected {config.lambda_windows} production XVG files for {leg} "
            f"replicate {replicate}, found {len(files)}"
        )
    output = config.project_dir / "analysis" / f"{leg}_rep_{replicate}"
    output.mkdir(parents=True, exist_ok=True)
    workflow = ABFE(
        config.temperature_k,
        units="kJ/mol",
        software="GROMACS",
        dir=str(directory),
        prefix="prod",
        suffix="xvg",
        outdirectory=str(output),
    )
    workflow.run(estimators=("MBAR",), breakdown=True)
    row = workflow.summary.loc[("Stages", "TOTAL")]
    workflow.summary.to_csv(output / "summary.csv")
    return float(row["MBAR"]), float(row["MBAR_Error"])


def analyze(config: ABFEConfig) -> Path:
    """Analyze every replicate and write an ABFE summary in kJ/mol."""
    correction = read_restraint_correction(config.project_dir / "setup/restraints.info")
    rows: list[dict[str, float | int]] = []
    for replicate in range(1, config.replicas + 1):
        complex_dg, complex_error = _analyze_leg(config, "complex", replicate)
        solvent_dg, solvent_error = _analyze_leg(config, "solvent", replicate)
        binding_dg = solvent_dg - complex_dg + correction
        propagated_error = math.sqrt(complex_error**2 + solvent_error**2)
        rows.append(
            {
                "replicate": replicate,
                "complex_kj_mol": complex_dg,
                "complex_error_kj_mol": complex_error,
                "solvent_kj_mol": solvent_dg,
                "solvent_error_kj_mol": solvent_error,
                "restraint_correction_kj_mol": correction,
                "binding_kj_mol": binding_dg,
                "propagated_error_kj_mol": propagated_error,
            }
        )

    analysis_dir = config.project_dir / "analysis"
    csv_path = analysis_dir / "abfe_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    binding_values = [float(row["binding_kj_mol"]) for row in rows]
    summary = {
        "method": "MBAR",
        "temperature_k": config.temperature_k,
        "replicates": config.replicas,
        "mean_binding_kj_mol": statistics.mean(binding_values),
        "replicate_standard_error_kj_mol": (
            statistics.stdev(binding_values) / math.sqrt(len(binding_values))
            if len(binding_values) > 1
            else None
        ),
        "restraint_correction_kj_mol": correction,
        "formula": "solvent - complex + restraint_correction",
    }
    summary_path = analysis_dir / "abfe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path
