#!/usr/bin/env python3
"""Create one clearance-array configuration from the shared model templates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import toml
import yaml


TARGETS = [
    "Log10_HLM_CLint_mL_min_kg",
    "Log10_RLM_CLint_mL_min_kg",
    "Log10_MLM_CLint_mL_min_kg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=("graphormer", "chemprop", "chemeleon", "kermt", "ml"),
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--split-column", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target")
    return parser.parse_args()


def validate_data(args: argparse.Namespace) -> None:
    train_columns = set(pd.read_csv(args.dataset, nrows=0).columns)
    test_columns = set(pd.read_csv(args.test_dataset, nrows=0).columns)
    targets = [args.target] if args.target else TARGETS
    train_required = {"SMILES", args.split_column, *targets}
    test_required = {"SMILES", *targets}
    train_missing = sorted(train_required - train_columns)
    test_missing = sorted(test_required - test_columns)
    if train_missing:
        raise ValueError(f"{args.dataset} is missing columns: {train_missing}")
    if test_missing:
        raise ValueError(f"{args.test_dataset} is missing columns: {test_missing}")


def common_toml(args: argparse.Namespace) -> dict:
    config = toml.load(args.base_config)
    base = config.setdefault("BaseConfig", {})
    base["workdir"] = str(args.workdir)
    base["task"] = "regression"
    dataset = config.setdefault("DatasetConfig", {})
    dataset.update(
        {
            "dataset_path": str(args.dataset),
            "test_dataset_path": str(args.test_dataset),
            "smiles_column": "SMILES",
            "target_column": TARGETS,
            "split_type": "predefined",
            "split_column": args.split_column,
            "val_fraction": 0.1,
            "test_fraction": 0.0,
        }
    )
    return config


def graphormer_config(args: argparse.Namespace) -> dict:
    config = common_toml(args)
    model = config.setdefault("ModelConfig", {})
    model.update(
        {
            "checkpoint_path": "/vf/users/liuy48/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt",
            "local_files_only": True,
            "freeze_encoder": False,
        }
    )
    model.pop("transfer_checkpoint", None)
    training = config.setdefault("TrainingConfig", {})
    training.update(
        {
            "device": "cuda",
            "devices": 1,
            "strategy": "auto",
            "batch_size": 16,
            "num_workers": 8,
            "num_epochs": 100,
            "early_stopping_patience": 15,
            "tensorboard": True,
            "tensorboard_dir": "tensorboard",
            "resume": False,
            "applicability_only": False,
        }
    )
    training.pop("resume_checkpoint", None)
    return config


def chemprop_config(args: argparse.Namespace) -> dict:
    config = common_toml(args)
    model = config.setdefault("ChempropConfig", {})
    model.update(
        {
            "message_hidden_dim": 2048,
            "depth": 6,
            "dropout": 0.1,
            "ffn_hidden_dim": 512,
            "ffn_num_layers": 2,
        }
    )
    training = config.setdefault("TrainingConfig", {})
    training.update(
        {
            "accelerator": "gpu",
            "devices": "1",
            "batch_size": 256,
            "num_workers": 8,
            "num_epochs": 100,
            "early_stopping_patience": 10,
            "warmup_epochs": 5,
            "init_lr": 1e-6,
            "max_lr": 1e-3,
            "final_lr": 1e-4,
            "gradient_clip_val": 1.0,
            "extra_args": ["-v"],
            "resume": False,
            "dry_run": False,
        }
    )
    training.pop("resume_checkpoint", None)
    return config


def chemeleon_config(args: argparse.Namespace) -> dict:
    config = common_toml(args)
    config.setdefault("BaseConfig", {})["seed"] = 42
    training = config.setdefault("CheMeleonTrainingConfig", {})
    training.update(
        {
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
            "num_nodes": 1,
            "batch_size": 64,
            "num_workers": 8,
            "num_epochs": 100,
            "early_stopping": True,
            "early_stopping_patience": 15,
            "evaluate_test": True,
            "tensorboard": True,
            "tensorboard_dir": "tensorboard",
            "resume": False,
        }
    )
    training.pop("resume_checkpoint", None)
    model = config.setdefault("CheMeleonConfig", {})
    model["freeze_epochs"] = 3
    model.pop("transfer_checkpoint", None)
    return config


def kermt_config(args: argparse.Namespace) -> dict:
    config = common_toml(args)
    model = config.setdefault("KERMTConfig", {})
    model.update(
        {
            "checkpoint_path": "/vf/users/liuy48/.cache/chemflow/kermt/NV-KERMT-70M-v2/kermt_contrastive_v2.0.pt",
            "local_files_only": True,
            "use_cuikmolmaker_featurization": True,
            "features_generator": "rdkit_2d_normalized_cuik_molmaker",
            "freeze_encoder": False,
        }
    )
    training = config.setdefault("TrainingConfig", {})
    training.update(
        {
            "batch_size": 32,
            "num_epochs": 100,
            "ensemble_size": 1,
            "num_folds": 1,
            "resume": False,
            "dry_run": False,
        }
    )
    training.pop("resume_checkpoint", None)
    return config


def ml_config(args: argparse.Namespace) -> dict:
    if args.target not in TARGETS:
        raise ValueError("--target must select one clearance endpoint for ML.")
    with args.base_config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Base configuration must be a mapping: {args.base_config}")

    columns = ["SMILES", args.target]
    training = pd.read_csv(args.dataset)
    labels = training[args.split_column].astype(str).str.strip().str.lower()
    if not set(labels.unique()) <= {"train", "val"}:
        raise ValueError(f"Unexpected labels in {args.split_column}: {sorted(labels.unique())}")
    training = training[columns].copy()
    training["split"] = labels
    testing = pd.read_csv(args.test_dataset)[columns].copy()
    testing["split"] = "test"
    prepared_dir = args.workdir / "prepared_data"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    combined_path = prepared_dir / "data_with_splits.csv"
    pd.concat([training, testing], ignore_index=True).to_csv(combined_path, index=False)

    config["workdir"] = str(args.workdir)
    config["task_type"] = "regression"
    config["resume"] = False
    data = config.setdefault("data", {})
    data.update({"data_file": str(combined_path), "X_col": "SMILES", "y_col": args.target})
    split = config.setdefault("data_split", {})
    split.update(
        {
            "split_method": "predefined",
            "split_column": "split",
            "test_fraction": 0.0,
            "valid_fraction": 0.0,
            "save_split_data": True,
            "save_dir": str(args.workdir / "split_data"),
            "split_name": "clearance_predefined_split",
        }
    )
    config["cv_strategy"] = "scaffold-grouped"
    config["n_jobs"] = 8
    config.setdefault("featurization", {})["features"] = ["ecfp4"]
    return config


def main() -> None:
    args = parse_args()
    for path in (args.base_config, args.dataset, args.test_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_data(args)
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    builders = {
        "graphormer": graphormer_config,
        "chemprop": chemprop_config,
        "chemeleon": chemeleon_config,
        "kermt": kermt_config,
        "ml": ml_config,
    }
    config = builders[args.model](args)
    if args.model == "ml":
        with args.output.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    else:
        with args.output.open("w", encoding="utf-8") as stream:
            toml.dump(config, stream)
    print(f"Generated {args.model} configuration: {args.output}", flush=True)


if __name__ == "__main__":
    main()
