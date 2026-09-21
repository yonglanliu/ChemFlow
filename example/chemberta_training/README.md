# ChemBERTa property prediction

This example fine-tunes `DeepChem/ChemBERTa-77M-MLM` on canonical SMILES with
a new ChemFlow property head. It supports single-task or partially labeled
multitask regression, binary classification, and mixed regression/classification.

Install the optional dependencies:

```bash
pip install -e '.[chemberta]'
```

Train:

```bash
chemflow train chemberta example/chemberta_training/conf.toml
```

The run writes `data_splits.csv`, `rejected_rows.csv`,
`training_history.csv`, `test_predictions.csv`, and a reloadable
`best_model/` directory. It also writes `training_task_metrics.csv`,
`metrics.json`, `best_model/applicability.pt`, and
`best_model/calibration.json`. Regression targets are standardized using only the
training split. Missing multitask labels are masked, and optional task weights
are calculated from training-label counts.

Set `split_column = "split"` to reuse existing train/val/test assignments.
Otherwise, `split_type = "scaffold_balanced"` keeps scaffolds separated across
the outer splits. Use a local downloaded model directory as `model_name` and
set `local_files_only = true` on an offline HPC node.

`DeepChem/ChemBERTa-77M-MLM` is a RoBERTa checkpoint, so the example uses
`architecture = "roberta"`. Use `architecture = "auto"` only for another
checkpoint with complete Hugging Face AutoConfig metadata.

If Python reports `CERTIFICATE_VERIFY_FAILED` behind an institutional proxy,
do not disable TLS verification. On Python 3.10 or newer, first try using the
certificates trusted by the operating system, then restart the shell:

```bash
python -m pip install pip-system-certs
```

Alternatively, obtain your institutional CA bundle from IT and configure it
before training:

```bash
export REQUESTS_CA_BUNDLE=/path/to/institution-ca-bundle.pem
```

For an offline compute node, download the model once on a machine with Hub
access:

```bash
hf download DeepChem/ChemBERTa-77M-MLM \
  --local-dir /path/to/models/ChemBERTa-77M-MLM
```

Then set `model_name` to that directory and `local_files_only = true`.

Predict with the saved model package:

```bash
chemflow predict chemberta \
  --input molecules.csv \
  --structure-column SMILES \
  --model-directory example/chemberta_training/physchem_multitask/best_model \
  --mc-dropout-samples 30 \
  --output predictions.csv
```

`--mc-dropout-samples 30` is optional and reports the standard deviation of
stochastic property-head passes alongside each prediction. Inference also
applies global or nearest-neighbor local validation calibration and reports
fingerprint/embedding domain diagnostics when applicability artifacts exist.

Resume an interrupted run with `resume = true`; the last complete epoch is
stored at `checkpoints/last.pt`. Checkpoints deserialize on CPU before model
and optimizer state move to the selected CPU, CUDA, or MPS device. Rebuild
applicability/calibration artifacts without optimizing weights by setting
`applicability_only = true`.

For single-node CUDA DDP, set `device = "cuda"`, `devices` to the GPU count,
and `strategy = "ddp"`, then launch, for example:

```bash
torchrun --standalone --nproc_per_node=4 "$(command -v chemflow)" \
  train chemberta example/chemberta_training/conf.toml
```

The configured batch size is per GPU. Rank 0 performs validation,
checkpointing, test evaluation, and applicability packaging.
