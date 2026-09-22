from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from chemflow.cli.main import build_parser
from chemflow.deep_learning.chemprop.trainer import ChempropTrainer


TARGETS = ["HLM", "RLM", "MLM"]


def _write_config(
    path: Path,
    dataset: Path,
    test_dataset: Path,
    *,
    split_column: str = "split",
) -> None:
    path.write_text(
        f"""
[BaseConfig]
workdir = "run"
task = "regression"

[DatasetConfig]
dataset_path = "{dataset}"
test_dataset_path = "{test_dataset}"
smiles_column = "SMILES"
target_column = ["HLM", "RLM", "MLM"]
split_type = "predefined"
split_column = "{split_column}"
val_fraction = 0.1
test_fraction = 0.0

[ChempropConfig]
message_hidden_dim = 128
depth = 4

[TrainingConfig]
metrics = ["rmse", "mae", "r2"]
dry_run = true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _datasets(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "trainval.csv"
    pd.DataFrame(
        {
            "SMILES": ["C", "CC", "CCC", "CCCC"],
            "HLM": [1.0, 2.0, 3.0, 4.0],
            "RLM": [1.5, None, 3.5, None],
            "MLM": [None, 2.5, None, 4.5],
            "split": ["train", "train", "val", "val"],
        }
    ).to_csv(dataset, index=False)
    test_dataset = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN"],
            "HLM": [2.0, 3.0],
            "RLM": [2.1, 3.1],
            "MLM": [2.2, 3.2],
        }
    ).to_csv(test_dataset, index=False)
    return dataset, test_dataset


def test_chemprop_cli_is_registered():
    args = build_parser().parse_args(["train", "chemprop", "config.toml"])
    assert args.model == "chemprop"


def test_predefined_split_prepares_three_files_and_command(tmp_path: Path):
    dataset, test_dataset = _datasets(tmp_path)
    config = tmp_path / "conf.toml"
    _write_config(config, dataset, test_dataset)
    trainer = ChempropTrainer(config)

    paths, audit = trainer.prepare_data()
    command = trainer.build_command(paths)

    assert [path.name for path in paths] == ["data_with_splits.csv"]
    prepared = pd.read_csv(paths[0])
    assert prepared["split"].value_counts().to_dict() == {
        "train": 2,
        "val": 2,
        "test": 2,
    }
    assert audit["target_label_counts"] == {"HLM": 4, "RLM": 2, "MLM": 2}
    assert "--target-columns" in command
    assert "--split-type" not in command
    assert command[command.index("--splits-column") + 1] == "split"
    assert command[command.index("--message-hidden-dim") + 1] == "128"
    assert command[command.index("--depth") + 1] == "4"


def test_split_export_flag_tracks_chemprop_version(tmp_path: Path, monkeypatch):
    dataset, test_dataset = _datasets(tmp_path)
    config = tmp_path / "conf.toml"
    _write_config(config, dataset, test_dataset)
    trainer = ChempropTrainer(config)
    paths, _ = trainer.prepare_data()

    monkeypatch.setattr(trainer, "_chemprop_version", lambda: "2.2.1")
    assert "--save-smiles-splits" in trainer.build_command(paths)

    monkeypatch.setattr(trainer, "_chemprop_version", lambda: "2.3.1")
    assert "--save-data-splits" in trainer.build_command(paths)


def test_dry_run_writes_reproducibility_manifest(tmp_path: Path):
    dataset, test_dataset = _datasets(tmp_path)
    config = tmp_path / "conf.toml"
    _write_config(config, dataset, test_dataset)

    trainer = ChempropTrainer(config)
    trainer.train()

    manifest = json.loads(
        (tmp_path / "run" / "chemprop_launch_manifest.json").read_text()
    )
    assert manifest["backend"] == "Chemprop v2"
    assert manifest["data_audit"]["rows"] == 4
    assert Path(manifest["command"][0]).name == "chemprop"
    assert manifest["command"][1] == "train"


def test_predefined_split_rejects_unknown_labels(tmp_path: Path):
    dataset, test_dataset = _datasets(tmp_path)
    frame = pd.read_csv(dataset)
    frame.loc[0, "split"] = "development"
    frame.to_csv(dataset, index=False)
    config = tmp_path / "conf.toml"
    _write_config(config, dataset, test_dataset)

    with pytest.raises(ValueError, match="Unsupported predefined split labels"):
        ChempropTrainer(config).prepare_data()


def test_generated_split_with_external_test_uses_one_cli_file(tmp_path: Path):
    dataset = tmp_path / "training.csv"
    pd.DataFrame(
        {
            "SMILES": ["C" * size for size in range(1, 21)],
            "HLM": list(range(20)),
        }
    ).to_csv(dataset, index=False)
    test_dataset = tmp_path / "test.csv"
    pd.DataFrame({"SMILES": ["CO", "CN"], "HLM": [1.0, 2.0]}).to_csv(
        test_dataset, index=False
    )
    config = tmp_path / "conf.toml"
    config.write_text(
        f"""
[BaseConfig]
workdir = "run"
task = "regression"
[DatasetConfig]
dataset_path = "{dataset}"
test_dataset_path = "{test_dataset}"
smiles_column = "SMILES"
target_column = "HLM"
split_type = "random"
val_fraction = 0.1
test_fraction = 0.0
[TrainingConfig]
dry_run = true
""".strip()
        + "\n"
    )

    trainer = ChempropTrainer(config)
    paths, _ = trainer.prepare_data()
    prepared = pd.read_csv(paths[0])
    command = trainer.build_command(paths)

    assert len(paths) == 1
    assert prepared["split"].value_counts().to_dict() == {
        "train": 18,
        "val": 2,
        "test": 2,
    }
    assert "--splits-column" in command
    assert "--split-type" not in command


def test_resume_discovers_last_checkpoint_and_records_audit(tmp_path: Path):
    dataset, test_dataset = _datasets(tmp_path)
    config = tmp_path / "conf.toml"
    _write_config(config, dataset, test_dataset)
    config.write_text(
        config.read_text().replace(
            "[TrainingConfig]\n",
            "[TrainingConfig]\nresume = true\n",
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "run/model_0/checkpoints/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"chemprop checkpoint")

    trainer = ChempropTrainer(config)
    checkpoints = trainer._resolve_resume_checkpoints()
    paths, _ = trainer.prepare_data()
    command = trainer.build_command(paths, checkpoints)

    assert checkpoints == [checkpoint]
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)

    trainer.train()
    manifest = json.loads(
        (tmp_path / "run/chemprop_launch_manifest.json").read_text()
    )
    assert manifest["resume"]["enabled"] is True
    assert manifest["resume"]["mode"] == "weights_only"
    assert manifest["resume"]["checkpoints"][0]["path"] == str(checkpoint)
    assert len(manifest["resume"]["checkpoints"][0]["sha256"]) == 64
    assert manifest["resume"]["optimizer_scheduler_epoch_restored"] is False


def test_resume_requires_existing_checkpoint(tmp_path: Path):
    dataset, test_dataset = _datasets(tmp_path)
    config = tmp_path / "conf.toml"
    _write_config(config, dataset, test_dataset)
    config.write_text(
        config.read_text().replace(
            "[TrainingConfig]\n",
            "[TrainingConfig]\nresume = true\n",
        ),
        encoding="utf-8",
    )

    trainer = ChempropTrainer(config)
    with pytest.raises(FileNotFoundError, match="resume checkpoint"):
        trainer._resolve_resume_checkpoints()
