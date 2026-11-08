"""ChemFlow predictor as a REINVENT4 ``ExternalProcess`` scoring component.

REINVENT4 launches this module as a subprocess, writes one SMILES per line to
stdin and expects JSON on stdout::

    {"version": 1, "payload": {"score": [0.4, NaN, 0.9]}}

Molecules ChemFlow cannot featurise get ``NaN`` (REINVENT treats them as
invalid; ``null`` would give it an unusable object array). Everything except the JSON goes to stderr so library
chatter cannot corrupt the protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from typing import Sequence, TextIO


def score_smiles(predictor, smiles: Sequence[str], column: str = "prediction") -> list[float | None]:
    """Score ``smiles`` with a ChemFlow predictor; ``None`` for failed molecules."""
    smiles = list(smiles)
    scores: list[float | None] = [None] * len(smiles)
    if not smiles:
        return scores
    try:
        frame = predictor.predict(smiles)
    except ValueError:  # nothing featurisable
        return scores
    if column not in frame.columns:
        raise ValueError(
            f"column '{column}' not in predictor output {list(frame.columns)}"
        )
    for index, value in zip(frame["input_index"], frame[column]):
        value = float(value)
        scores[int(index)] = None if math.isnan(value) else value
    return scores


def format_payload(scores: Sequence[float | None]) -> str:
    values = [float("nan") if s is None else float(s) for s in scores]
    return json.dumps({"version": 1, "payload": {"score": values}})


def main(argv: Sequence[str] | None = None, stdin: TextIO | None = None,
         stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChemFlow scorer for REINVENT4 ExternalProcess.")
    parser.add_argument("--model", required=True, help="ChemFlow model pickle")
    parser.add_argument("--column", default="prediction",
                        help="Predictor output column to use as the score (default: prediction)")
    args = parser.parse_args(argv)

    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    smiles = [line.strip() for line in stdin if line.strip()]

    # Library output must not reach the JSON channel.
    with contextlib.redirect_stdout(sys.stderr):
        from chemflow.machine_learning.predict.predictor import ChemFlowPredictor

        predictor = ChemFlowPredictor(args.model)
        scores = score_smiles(predictor, smiles, args.column)

    stdout.write(format_payload(scores) + "\n")
    stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
