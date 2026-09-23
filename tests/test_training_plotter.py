from pathlib import Path

import pandas as pd

from chemflow.deep_learning.plotter.training_plotter import plot_training_history


def test_plot_training_history_recognizes_lightning_epoch_loss(tmp_path: Path):
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss_epoch": [0.8, 0.5, 0.3],
            "val_loss": [0.9, 0.6, 0.55],
        }
    )

    result = plot_training_history(
        history,
        tmp_path / "training_history.png",
        title="Test training",
    )

    assert result is not None
    assert result["train_column"] == "train_loss_epoch"
    assert result["validation_column"] == "val_loss"
    assert (tmp_path / "training_history.png").is_file()
    assert (tmp_path / "training_history.pdf").is_file()


def test_plot_training_history_averages_replicates(tmp_path: Path):
    history = pd.DataFrame(
        {
            "model": [0, 1, 0, 1],
            "epoch": [1, 1, 2, 2],
            "train_loss": [0.8, 0.9, 0.5, 0.6],
            "loss_val": [1.0, 1.1, 0.7, 0.8],
        }
    )

    result = plot_training_history(history, tmp_path / "ensemble_curve")

    assert result is not None
    assert result["train_column"] == "train_loss"
    assert result["validation_column"] == "loss_val"
    assert (tmp_path / "ensemble_curve.png").is_file()
    assert (tmp_path / "ensemble_curve.pdf").is_file()
