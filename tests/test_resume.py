from __future__ import annotations

import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from chemflow.deep_learning.gpt.trainer import GPTDDPTrainer
from chemflow.deep_learning.hf_graphormer.trainer import (
    HuggingFaceGraphormerTrainer,
)
from chemflow.deep_learning.lstm.train_utils import (
    get_resume_path,
    load_checkpoint_for_resume,
    save_checkpoint,
)
from chemflow.machine_learning.train.hyperparameter_tuning import (
    tune_parameters_multiple_model,
)
from chemflow.machine_learning.train.regular_training import regular_training


class ResumePortabilityTest(unittest.TestCase):
    def _model_and_optimizer(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.ones((1, 2))).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return model, optimizer

    def test_lstm_documented_resume_checkpoint_alias(self):
        checkpoint_dir = Path("run/checkpoints")
        explicit = "/tmp/explicit-last-model.pt"
        config = SimpleNamespace(resume=False, resume_checkpoint=explicit)
        self.assertEqual(
            get_resume_path(config, checkpoint_dir),
            Path(explicit).resolve(),
        )

    def test_lstm_checkpoint_round_trip_on_cpu(self):
        model, optimizer = self._model_and_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_model.pt"
            save_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                epoch=3,
                train_loss=1.0,
                val_loss=2.0,
                val_perplexity=3.0,
                best_val_loss=2.0,
                best_val_perplexity=3.0,
                best_epoch=3,
                patience_counter=0,
            )
            state = load_checkpoint_for_resume(
                path,
                model,
                optimizer,
                device=torch.device("cpu"),
            )
        self.assertEqual(state["start_epoch"], 4)

    def test_gpt_checkpoint_round_trip_on_cpu(self):
        trainer = object.__new__(GPTDDPTrainer)
        model, optimizer = self._model_and_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_model.pt"
            trainer.save_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                epoch=4,
                train_loss=1.0,
                val_loss=2.0,
                val_perplexity=3.0,
                best_val_loss=2.0,
                best_val_perplexity=3.0,
                best_epoch=4,
                patience_counter=0,
            )
            state = trainer.load_checkpoint_for_resume(
                path,
                model,
                optimizer,
                device=torch.device("cpu"),
            )
        self.assertEqual(state["start_epoch"], 5)

    def test_graphormer_checkpoint_round_trip_on_cpu(self):
        trainer = object.__new__(HuggingFaceGraphormerTrainer)
        trainer.task = "regression"
        trainer.device = torch.device("cpu")
        trainer.is_main_process = False
        model, optimizer = self._model_and_optimizer()
        means = np.asarray([1.0])
        stds = np.asarray([2.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "epoch": 5,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "task": "regression",
                    "best_metric": 0.5,
                    "bad_epochs": 1,
                    "history": [],
                    "target_means": means.tolist(),
                    "target_stds": stds.tolist(),
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                },
                path,
            )
            state = trainer._restore_resume_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                expected_means=means,
                expected_stds=stds,
            )
        self.assertEqual(state[0], 6)

    def test_traditional_ml_resume_skips_completed_tuning_model(self):
        features = np.zeros((4, 2), dtype=float)
        targets = np.zeros(4, dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = root / "random_forest"
            completed.mkdir()
            result = {"model_name": "Random Forest", "status": "complete"}
            (completed / "random_forest_summary.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            with (completed / "random_forest_model_package.pkl").open("wb") as stream:
                pickle.dump({"model": "placeholder"}, stream)
            loaded = tune_parameters_multiple_model(
                features,
                targets,
                features,
                targets,
                [{"model_name": "Random Forest"}],
                {"resume": True, "task_type": "regression"},
                output_dir=root,
            )
        self.assertEqual(loaded, [result])

    def test_traditional_ml_resume_skips_completed_seed(self):
        features = np.zeros((4, 2), dtype=float)
        targets = np.zeros(4, dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = root / "random_forest" / "seed_42"
            completed.mkdir(parents=True)
            result = {"model_name": "Random Forest", "seed": 42}
            (completed / "random_forest_summary.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            with (completed / "random_forest_model_package.pkl").open("wb") as stream:
                pickle.dump({"model": "placeholder"}, stream)
            loaded = regular_training(
                features,
                targets,
                features,
                targets,
                {
                    "model_name": "Random Forest",
                    "task_type": "regression",
                    "seeds": [42],
                    "resume": True,
                },
                output_dir=root,
            )
        self.assertEqual(loaded, [result])


if __name__ == "__main__":
    unittest.main()
