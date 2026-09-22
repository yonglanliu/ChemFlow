from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch

from chemflow.cli.main import build_parser
from chemflow.deep_learning.kermt.pretrained import (
    KERMT_ARTIFACTS,
    KERMT_CHECKPOINT,
    ensure_pretrained_artifacts,
)
from chemflow.deep_learning.kermt.trainer import KERMTTrainer


def test_kermt_cli_is_registered():
    args = build_parser().parse_args(["train", "kermt", "config.toml"])
    assert args.model == "kermt"
    assert args.func.__name__ == "train_kermt"


def test_kermt_rdkit_featurization_without_cuik():
    pytest.importorskip("descriptastorus")
    from chemflow.deep_learning.kermt.vendor.kermt.data.molgraph import MolGraph

    graph = MolGraph(
        "CCO",
        args=Namespace(
            use_cuikmolmaker_featurization=False,
            bond_drop_rate=0.0,
        ),
    )
    assert graph.n_atoms == 3
    assert graph.n_bonds == 4
    assert len(graph.f_atoms[0]) == 151


def test_kermt_cpu_training_does_not_select_cuda(monkeypatch):
    """A numeric --gpu default must not trigger CUDA on CPU-only PyTorch."""
    from chemflow.deep_learning.kermt.vendor.task import train as train_module

    class ReachedDataLoading(Exception):
        pass

    def stop_at_data_loading(*args, **kwargs):
        raise ReachedDataLoading

    def fail_if_cuda_is_selected(*args, **kwargs):
        raise AssertionError("torch.cuda.set_device was called for a CPU run")

    monkeypatch.setattr(train_module, "load_data", stop_at_data_loading)
    monkeypatch.setattr(train_module.torch.cuda, "set_device", fail_if_cuda_is_selected)

    with pytest.raises(ReachedDataLoading):
        train_module.run_training(Namespace(cuda=False, gpu=0))


def test_kermt_model_audit_reports_loaded_checkpoint_and_frozen_encoder():
    from chemflow.deep_learning.kermt.vendor.task.train import _log_model_audit

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kermt = torch.nn.Linear(3, 2)
            self.head = torch.nn.Linear(2, 1)

    model = TinyModel()
    for parameter in model.kermt.parameters():
        parameter.requires_grad = False

    messages = []
    _log_model_audit(
        model,
        Namespace(fine_tune_coff=0.0),
        "/cache/kermt.pt",
        messages.append,
    )
    output = "\n".join(messages)
    assert "Checkpoint loaded successfully: yes" in output
    assert "Checkpoint: /cache/kermt.pt" in output
    assert "Total parameters:     11" in output
    assert "Trainable parameters: 3 (27.2727%)" in output
    assert "Frozen parameters:    8" in output
    assert "Encoder trainable:    0/8 (frozen: yes)" in output


def test_kermt_resume_checkpoint_resolves_current_fold_and_model(tmp_path):
    from chemflow.deep_learning.kermt.vendor.task.train import (
        _resolve_resume_checkpoint,
    )

    checkpoint = tmp_path / "fold_1" / "model_2" / "last_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    args = Namespace(
        resume_checkpoint=str(tmp_path),
        fold_num=1,
        num_folds=2,
        ensemble_size=3,
    )

    assert _resolve_resume_checkpoint(args, 2, "/unused") == str(checkpoint)


def test_kermt_resume_audit_reports_epoch_and_parameter_counts():
    from chemflow.deep_learning.kermt.vendor.task.train import _log_resume_audit

    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad = False
    messages = []
    _log_resume_audit(model, "/run/last_checkpoint.pt", 12, messages.append)
    output = "\n".join(messages)

    assert "Resume successful: yes" in output
    assert "Resume checkpoint: /run/last_checkpoint.pt" in output
    assert "Next epoch: 12" in output
    assert "Trainable parameters after resume: 6/8 (75.0000%)" in output
    assert "Frozen parameters after resume: 2" in output


def test_kermt_pretrained_artifacts_download_to_cache(tmp_path, monkeypatch):
    calls = []

    def fake_download(*, repo_id, filename, revision, local_dir, local_files_only):
        calls.append((repo_id, filename, revision, local_files_only))
        destination = Path(local_dir) / filename
        destination.write_bytes(f"artifact:{filename}".encode())
        return str(destination)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    messages = []
    resolved = ensure_pretrained_artifacts(
        cache_dir=tmp_path / "cache",
        log=messages.append,
    )

    assert set(resolved) == set(KERMT_ARTIFACTS)
    assert resolved[KERMT_CHECKPOINT].parent == (tmp_path / "cache").resolve()
    assert len(calls) == len(KERMT_ARTIFACTS)
    assert any("Downloading KERMT artifact" in message for message in messages)
    assert messages[-1] == "KERMT pretrained artifacts loaded successfully."


def test_kermt_prepares_external_test_and_command(tmp_path):
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"test checkpoint")
    training = tmp_path / "training.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "SMILES": ["CC", "CCC", "CCCC", "CCO", "CCN"],
            "HLM": [1.0, 2.0, 3.0, 4.0, None],
            "split": ["train", "train", "val", "val", "train"],
        }
    ).to_csv(training, index=False)
    pd.DataFrame({"SMILES": ["CCN"], "HLM": [5.0]}).to_csv(test, index=False)
    config = tmp_path / "config.toml"
    config.write_text(
        f'''[BaseConfig]
workdir = "{tmp_path / 'run'}"
task = "regression"
[DatasetConfig]
dataset_path = "{training}"
test_dataset_path = "{test}"
smiles_column = "SMILES"
target_column = "HLM"
split_type = "predefined"
split_column = "split"
val_fraction = 0.1
test_fraction = 0.0
[KERMTConfig]
checkpoint_path = "{checkpoint}"
features_generator = ""
rdkit2d_normalization_type = ""
freeze_encoder = true
[TrainingConfig]
seed = 7
num_epochs = 3
dry_run = true
''',
        encoding="utf-8",
    )

    trainer = KERMTTrainer(config)
    trainer.train()

    prepared = pd.read_csv(tmp_path / "run" / "prepared_data" / "train.csv")
    assert list(prepared.columns) == ["smiles", "HLM"]
    assert len(prepared) == 2
    assert prepared["HLM"].notna().all()
    assert len(pd.read_csv(tmp_path / "run" / "prepared_data" / "test.csv")) == 1
    manifest = json.loads(
        (tmp_path / "run" / "kermt_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["targets"] == ["HLM"]
    assert "finetune" in manifest["command"]
    assert "chemflow.deep_learning.kermt.vendor.main" in manifest["command"]
    coefficient_index = manifest["command"].index("--fine_tune_coff")
    assert manifest["command"][coefficient_index + 1] == "0.0"
    assert manifest["backend"] == "vendored NVIDIA-BioNeMo/KERMT"
    assert manifest["checkpoint_sha256"]
