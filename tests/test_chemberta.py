from __future__ import annotations

import numpy as np
import torch

from chemflow.cli.main import build_parser
from chemflow.deep_learning.chemberta.trainer import ChemBERTaTrainer


def test_chemberta_cli_is_registered():
    args = build_parser().parse_args(["train", "chemberta", "config.toml"])
    assert args.model == "chemberta"
    assert args.func.__name__ == "train_chemberta"
    predict_args = build_parser().parse_args([
        "predict", "chemberta", "--smiles", "CCO", "--model-directory",
        "best_model", "--output", "predictions.csv",
    ])
    assert predict_args.predict_model == "chemberta"
    assert predict_args.func.__name__ == "predict_chemberta"


def test_chemberta_masked_mixed_loss():
    logits = torch.tensor([[1.0, 0.0], [3.0, 2.0]])
    labels = torch.tensor([[0.0, 0.0], [2.0, float("nan")]])
    loss = ChemBERTaTrainer._loss(
        logits, labels, torch.ones(2), ["regression", "classification"]
    )
    expected = (
        1.0
        + 1.0
        + float(torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(0.0), torch.tensor(0.0)
        ))
    ) / 3.0
    assert np.isclose(float(loss), expected)
