from __future__ import annotations

import numpy as np
import torch

from chemflow.cli.main import build_parser
from chemflow.deep_learning.chemberta.trainer import (
    ChemBERTaTrainer,
    _cpu_rng_state,
    _move_optimizer_state_to_device,
    _validate_split_config,
)


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


def test_chemberta_predefined_split_config():
    config = {"split_type": "predefined", "split_column": "split"}
    _validate_split_config(config)
    assert config["split_type"] == "predefined"

    inferred = {"split_column": "partition"}
    _validate_split_config(inferred)
    assert inferred["split_type"] == "predefined"

    try:
        _validate_split_config({"split_type": "predefined"})
    except ValueError as error:
        assert "requires DatasetConfig.split_column" in str(error)
    else:
        raise AssertionError("predefined split without split_column was accepted")


def test_chemberta_resume_state_is_device_portable():
    original_rng = torch.get_rng_state()
    try:
        # Old checkpoints may be mapped to another device or represented with
        # a non-byte dtype. Resume normalises them before restoring CPU RNG.
        restored_rng = _cpu_rng_state(original_rng.to(dtype=torch.int16))
        assert restored_rng.device.type == "cpu"
        assert restored_rng.dtype == torch.uint8
        torch.set_rng_state(restored_rng)
    finally:
        torch.set_rng_state(original_rng)

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    _move_optimizer_state_to_device(optimizer, torch.device("cpu"))
    assert all(
        not torch.is_tensor(value) or value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
    )
