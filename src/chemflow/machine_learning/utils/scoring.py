# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from typing import Dict, Any, Optional
import yaml
from pathlib import Path
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    make_scorer,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from src.config import PROJECT_ROOT 
CONFIG_PATH = PROJECT_ROOT / "src"/"config" / "grid_search_conf.yaml"


def load_training_config(config_path: str | Path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_scoring_config(task_type: str):
    config = load_training_config(CONFIG_PATH)
    task = task_type.lower()
    if task not in config["scoring"]:
        raise ValueError(
            f"Invalid task_type: {task_type}. "
            "Expected classification or regression."
        )
    return config["scoring"][task]


def get_scoring(config: Dict[str, Any], n_classes: Optional[int] = None):
    task = config.get("task", config.get("task_type", "regression"))

    if task == "classification":
        assert n_classes is not None
        print(f"Number of Classes: {n_classes}")
        if n_classes <= 2:
            scorers = {
                "roc_auc": "roc_auc",
                "average_precision": "average_precision",
                "balanced_accuracy": "balanced_accuracy",
                "f1": "f1",
                "precision": "precision",
                "recall": "recall",
            }
        else:
            scorers = {
                "roc_auc": "roc_auc_ovr_weighted",
                "balanced_accuracy": "balanced_accuracy",
                "f1": make_scorer(
                    f1_score,
                    average="macro",
                    zero_division=0,
                ),
                "precision": make_scorer(
                    precision_score,
                    average="macro",
                    zero_division=0,
                ),
                "recall": make_scorer(
                    recall_score,
                    average="macro",
                    zero_division=0,
                ),
            }

    else:
        scorers = {
            "r2": "r2",
            "root_mean_squared_error": make_scorer(
                root_mean_squared_error,
                greater_is_better=False,
            ),
            "mean_absolute_error": make_scorer(
                mean_absolute_error,
                greater_is_better=False,
            ),
        }

    scoring_metrics = config["scoring_metrics"]

    for metric in scoring_metrics:
        if metric not in scorers:
            raise ValueError(
                f"Unknown scoring metric '{metric}' for task='{task}', "
                f"n_classes={n_classes}. Available: {list(scorers.keys())}"
            )

    return {
        metric: scorers[metric]
        for metric in scoring_metrics
    }

def get_refit_metrics(config: Dict[str, Any]) -> str:
    """
    Validate and return the refit metric.
    """

    refit_metric = config.get("refit_metric")

    if refit_metric is None:
        raise ValueError(
            "Missing required config key: 'refit_metric'"
        )

    scoring_metrics = config.get("scoring_metrics")

    if scoring_metrics is None:
        raise ValueError(
            "Missing required config key: 'scoring_metrics'"
        )

    if refit_metric not in scoring_metrics:
        raise ValueError(
            f"refit_metric '{refit_metric}' "
            f"must be one of {scoring_metrics}"
        )

    return refit_metric