# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import torch.distributed as dist
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_distributed()) or dist.get_rank() == 0


def _barrier() -> None:
    if _is_distributed():
        dist.barrier()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


class CrossValidation:
    """
    Generic K-fold splitter.

    The class knows nothing about Graphormer, DataLoaders, collators,
    optimizers, or DDP. It only creates deterministic train/validation
    index splits for a dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_splits: int = 5,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        if int(n_splits) < 2:
            raise ValueError("n_splits must be >= 2.")

        if len(dataset) < int(n_splits):
            raise ValueError(
                f"Dataset size ({len(dataset)}) must be >= n_splits ({n_splits})."
            )

        self.dataset = dataset
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

    def split_indices(
        self,
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        indices = np.arange(len(self.dataset))

        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.seed if self.shuffle else None,
        )

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(indices),
            start=1,
        ):
            yield fold, train_idx, val_idx


class DeepLearningGridSearchCV:
    """
    Grid-search controller for deep-learning models.

    Every hyperparameter configuration is evaluated using the SAME K-fold
    splits. Training is delegated to ``run_single_fold`` so this class stays
    independent of model architecture and training framework details.

    The callback must return a dictionary containing ``monitor_metric``.

    Expected callback signature
    ---------------------------
    run_single_fold(
        fold=...,
        train_idx=...,
        val_idx=...,
        params=...,
        config_index=...,
        config_output_dir=...,
    ) -> dict
    """

    def __init__(
        self,
        param_grid: dict[str, list[Any]],
        n_splits: int = 5,
        cv_seed: int = 42,
        monitor_metric: str = "val_rmse",
        monitor_mode: str = "min",
        output_dir: str | Path = "grid_search_results",
    ) -> None:
        if not isinstance(param_grid, dict) or not param_grid:
            raise ValueError("param_grid must be a non-empty dictionary.")

        for name, values in param_grid.items():
            if not isinstance(values, list) or not values:
                raise TypeError(
                    f"param_grid['{name}'] must be a non-empty list."
                )

        self.param_grid = copy.deepcopy(param_grid)
        self.n_splits = int(n_splits)
        self.cv_seed = int(cv_seed)
        self.monitor_metric = str(monitor_metric)
        self.monitor_mode = str(monitor_mode).lower()

        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'.")

        self.output_dir = Path(output_dir)

        if _is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        _barrier()

        self.results_: list[dict[str, Any]] = []
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.best_result_: dict[str, Any] | None = None

    def generate_parameter_combinations(
        self,
    ) -> list[dict[str, Any]]:
        keys = list(self.param_grid)
        values = [self.param_grid[key] for key in keys]

        return [
            dict(zip(keys, combination))
            for combination in itertools.product(*values)
        ]

    def _is_better(
        self,
        score: float,
        best_score: float | None,
    ) -> bool:
        if best_score is None:
            return True

        if self.monitor_mode == "min":
            return score < best_score

        return score > best_score

    def fit(
        self,
        *,
        cv_dataset: Dataset,
        cv_class: type[CrossValidation],
        run_single_fold: Callable[..., dict[str, Any]],
    ) -> "DeepLearningGridSearchCV":
        parameter_combinations = self.generate_parameter_combinations()

        if _is_main_process():
            print(
                f"Grid search: {len(parameter_combinations)} configurations | "
                f"{self.n_splits} folds/config | "
                f"{len(parameter_combinations) * self.n_splits} total trainings",
                flush=True,
            )

        # All configurations use the same splitter seed and therefore
        # exactly the same folds.
        fixed_splits = list(
            cv_class(
                dataset=cv_dataset,
                n_splits=self.n_splits,
                seed=self.cv_seed,
            ).split_indices()
        )

        for config_index, params in enumerate(
            parameter_combinations,
            start=1,
        ):
            config_output_dir = (
                self.output_dir / f"config_{config_index:03d}"
            )

            if _is_main_process():
                config_output_dir.mkdir(parents=True, exist_ok=True)
                print(
                    "\n"
                    + "=" * 80
                    + f"\nConfiguration {config_index}/{len(parameter_combinations)}"
                    + f"\nParameters: {params}"
                    + "\n"
                    + "=" * 80,
                    flush=True,
                )

            _barrier()

            fold_results: list[dict[str, Any]] = []

            for fold, train_idx, val_idx in fixed_splits:
                if _is_main_process():
                    print(
                        f"\nConfig {config_index} | Fold {fold}/{self.n_splits}",
                        flush=True,
                    )

                result = run_single_fold(
                    fold=fold,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    params=params,
                    config_index=config_index,
                    config_output_dir=config_output_dir,
                )

                if result is None:
                    raise RuntimeError(
                        "run_single_fold returned None. For DDP, return the same "
                        "metric dictionary on every rank and restrict only file "
                        "writing/printing to rank 0."
                    )

                if self.monitor_metric not in result:
                    raise KeyError(
                        f"Metric '{self.monitor_metric}' missing from fold result. "
                        f"Available keys: {sorted(result)}"
                    )

                fold_results.append(result)

            scores = np.asarray(
                [
                    float(result[self.monitor_metric])
                    for result in fold_results
                ],
                dtype=float,
            )

            if not np.all(np.isfinite(scores)):
                raise FloatingPointError(
                    f"Non-finite CV score for config {config_index}: {scores}"
                )

            mean_score = float(np.mean(scores))
            std_score = float(
                np.std(scores, ddof=1)
                if scores.size > 1
                else 0.0
            )

            result = {
                "config_index": int(config_index),
                "params": copy.deepcopy(params),
                "monitor_metric": self.monitor_metric,
                "mean_score": mean_score,
                "std_score": std_score,
                "fold_scores": scores.tolist(),
                "best_epochs": [
                    int(item.get("best_epoch", 0))
                    for item in fold_results
                ],
                "fold_results": fold_results,
            }

            self.results_.append(result)

            if self._is_better(mean_score, self.best_score_):
                self.best_score_ = mean_score
                self.best_params_ = copy.deepcopy(params)
                self.best_result_ = copy.deepcopy(result)

            if _is_main_process():
                with (config_output_dir / "cv_result.json").open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        _json_safe(result),
                        file,
                        indent=4,
                    )

                print(
                    f"{self.monitor_metric}: "
                    f"{mean_score:.6f} ± {std_score:.6f}",
                    flush=True,
                )

            _barrier()

        self._save_summary()
        _barrier()
        return self

    def _save_summary(self) -> None:
        if not _is_main_process():
            return

        rows: list[dict[str, Any]] = []

        for result in self.results_:
            row = {
                "config_index": result["config_index"],
                "mean_score": result["mean_score"],
                "std_score": result["std_score"],
            }
            row.update(result["params"])
            rows.append(row)

        dataframe = pd.DataFrame(rows)

        if not dataframe.empty:
            dataframe = dataframe.sort_values(
                "mean_score",
                ascending=(self.monitor_mode == "min"),
            )

        dataframe.to_csv(
            self.output_dir / "grid_search_summary.csv",
            index=False,
        )

        summary = {
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "n_splits": self.n_splits,
            "cv_seed": self.cv_seed,
            "best_score": self.best_score_,
            "best_params": self.best_params_,
            "best_result": self.best_result_,
        }

        with (self.output_dir / "best_config.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                _json_safe(summary),
                file,
                indent=4,
            )
