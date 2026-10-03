"""Shared sequential multiseed orchestration for configuration-driven trainers."""

from __future__ import annotations

import copy
import json
import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import toml


def run_multiseed(
    config_path: str | Path,
    trainer_factory: Callable[[Path], Any],
    *,
    backend: str,
) -> Any:
    """Run one trainer normally or one isolated run for every configured seed.

    ``TrainingConfig.seeds`` activates orchestration. Each derived TOML removes
    that list, sets ``TrainingConfig.seed``, and writes to
    ``<BaseConfig.workdir>/seed_<seed>``. The individual backend therefore
    remains responsible for all training, checkpointing, and evaluation.
    """
    source = Path(config_path).expanduser().resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    training = raw.get("TrainingConfig", {})
    configured = training.get("seeds")
    if configured is None:
        return trainer_factory(source).train()
    if not isinstance(configured, list) or not configured:
        raise ValueError("TrainingConfig.seeds must be a nonempty integer list.")
    seeds = [int(seed) for seed in configured]
    if len(set(seeds)) != len(seeds):
        raise ValueError("TrainingConfig.seeds must not contain duplicates.")
    if bool(training.get("resume", False)):
        raise ValueError(
            "resume=true is not supported by multiseed orchestration; resume "
            "an individual seed configuration instead."
        )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(
            "TrainingConfig.seeds cannot be orchestrated inside torchrun/DDP. "
            "Launch each generated seed configuration as a separate DDP job."
        )

    base = raw.setdefault("BaseConfig", {})
    parent = Path(base.get("workdir", f"./{backend}_run")).expanduser().resolve()
    config_dir = parent / "multiseed_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    summary_path = parent / "multiseed_summary.json"

    def save_summary(status: str) -> None:
        summary_path.write_text(
            json.dumps(
                {
                    "backend": backend,
                    "source_config": str(source),
                    "seeds": seeds,
                    "status": status,
                    "runs": runs,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for seed in seeds:
        derived = copy.deepcopy(raw)
        derived_training = derived.setdefault("TrainingConfig", {})
        derived_training.pop("seeds", None)
        derived_training["seed"] = int(seed)
        seed_workdir = parent / f"seed_{seed}"
        derived.setdefault("BaseConfig", {})["workdir"] = str(seed_workdir)
        seed_config = config_dir / f"seed_{seed}.toml"
        seed_config.write_text(toml.dumps(derived), encoding="utf-8")
        record = {
            "seed": int(seed),
            "workdir": str(seed_workdir),
            "config": str(seed_config),
            "status": "running",
        }
        runs.append(record)
        save_summary("running")
        print(
            f"{backend} multiseed run: seed={seed}, workdir={seed_workdir}",
            flush=True,
        )
        try:
            trainer_factory(seed_config).train()
        except Exception:
            record["status"] = "failed"
            save_summary("failed")
            raise
        artifact_names = (
            "metrics.json",
            "test_metrics.json",
            "run_summary.json",
            "training_history.csv",
            "training_task_metrics.csv",
        )
        record["artifacts"] = {
            name: str(seed_workdir / name)
            for name in artifact_names
            if (seed_workdir / name).is_file()
        }
        record["status"] = "completed"
        save_summary("running")

    save_summary("completed")
    return None
