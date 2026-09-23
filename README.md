# ChemFlow

<p align="center">
An extensible AI-powered platform for molecular design, property prediction,
virtual screening, and cheminformatics.
</p>

---

## Overview

ChemFlow is an open-source Python framework for modern AI-driven drug discovery and molecular machine learning.

It combines cheminformatics, graph-based deep learning, multitask prediction, generation, and uncertainty-aware evaluation in a modular, experiment-friendly workflow.

ChemFlow is designed to support:

- training graph and sequence models
- multitask property prediction
- uncertainty and bootstrap analysis
- dataset preparation and split-aware loading
- molecular generation and screening
- reproducible experimental pipelines for drug discovery

---

## Recent major updates

### Hugging Face Graphormer

ChemFlow uses the Hugging Face Graphormer implementation and the pretrained
PCQM4Mv1 checkpoint for single-task and masked multitask regression. It
supports target scaling, weighted multitask loss, checkpoint resume, DDP,
per-task metrics, and applicability-domain calibration.

### ChemBERTa

ChemFlow can fine-tune `DeepChem/ChemBERTa-77M-MLM` directly from canonical
SMILES for regression, binary classification, homogeneous multitask, or mixed
multitask property prediction. Missing labels are masked and regression target
scaling is fitted only on the training split. It supports the same operational
workflow as HF Graphormer: DDP, checkpoint resume, per-task metrics, weighted
losses, external or predefined test splits, applicability-only packaging,
validation calibration, local neighbor uncertainty, OOD diagnostics, and
MC-dropout inference.

### Chemprop v2

ChemFlow exposes the official Chemprop v2 CLI as a reproducible, randomly
initialized D-MPNN baseline. The adapter validates single- or multitask CSV
inputs, preserves independent test sets and predefined splits, records the
resolved command, and keeps Chemprop baselines distinct from pretrained
CheMeleon experiments.

### Standard deep-learning test evaluation

Graphormer, CheMeleon, ChemBERTa, Chemprop, and KERMT can run the same final
uncertainty evaluation when `test_mc_dropout_samples` is at least 2. Their
`metrics.json` reports independent-test metrics for deterministic, MC-dropout
mean, and locally calibrated predictions. Their `test_predictions.csv` stores
all three predictions, MC standard deviation, calibrated interval and
uncertainty, OOD flags, weak-local-support flags, and interval coverage.

Calibration is fitted exclusively from the validation partition. Each query
uses neighboring validation compounds; there is no global residual fallback.
`test_uncertainty_diagnostics` compares observed with nominal coverage and
labels the resulting intervals as overconfident, underconfident, or
approximately calibrated. MC dropout requires a nonzero dropout rate in the
trained model. Gaussian MC-dropout diagnostics are calculated with
Uncertainty Toolbox and include RMS/mean calibration error, miscalibration
area, sharpness, NLL, CRPS, check score, and interval score.

When a labeled test set is available, the same post-training step creates an
`uncertainty/` directory for every predictive deep-learning backend. Each
regression endpoint receives confidence-band, prediction-interval, ordered-
interval, calibration-before/after, residual-versus-uncertainty, sharpness,
adversarial-group-calibration, and combined overview plots in PNG and PDF.
Calibration curves, adversarial-group values, per-compound uncertainty data,
and the before/after calibration table are also exported as CSV files.

### Split-aware dataset loading

The model dataset loaders support molecule-aware global splits, predefined
split labels, and completely independent test files. Invalid molecules and the
resolved assignments are written to `rejected_rows.csv` and `data_splits.csv`.

```toml
[DatasetConfig]
split_type = "scaffold_balanced"
val_fraction = 0.1
test_fraction = 0.1
```

Supported generated splits are `random`, `random_with_repeated_smiles`,
`scaffold_balanced`, `kennard_stone`, and `kmeans`.

For predefined splits:

```toml
[DatasetConfig]
split_column = "split"
val_fraction = 0.1
test_fraction = 0.1
```

When `split_column` is provided, the loader respects the existing split rather than randomly splitting the data.

For an independent test file, set `test_dataset_path` and disable the internal
test split:

```toml
[DatasetConfig]
dataset_path = "./data/train.csv"
test_dataset_path = "./data/external_test.csv"
test_fraction = 0.0
```

For Graphormer configuration and HPC examples, see
[`example/hf_graphormer_training`](example/hf_graphormer_training/README.md).

### Uncertainty evaluation and bootstrap metrics

ChemFlow includes an uncertainty bootstrap workflow for multitask model evaluation.

Example:

```bash
CHEMFLOW_BIN="${CHEMFLOW_BIN:-/Users/yonglanliu/Desktop/ChemFlow/.venv/bin/chemflow}"

"$CHEMFLOW_BIN" uncertainty bootstrap \
    --input /path/to/predictions.csv \
    --task "TaskA:target_A:pred_A" \
    --task "TaskB:target_B:pred_B" \
    --metrics r2 mae rmse pearson spearman kendall \
    --n-bootstrap 2000 \
    --confidence-level 0.95 \
    --seed 42 \
    --plot-distributions \
    --output-dir /path/to/bootstrap_results
```

This produces bootstrap confidence intervals and distribution plots for each task and metric.

---

## Core features

### Cheminformatics

- public dataset curation
- molecular similarity search
- descriptors and fingerprints
- visualization
- dataset preprocessing
- structure conversion utilities

### Molecular AI

- Hugging Face Graphormer
- Chemprop v2
- ChemBERTa
- LSTM
- GPT-style generation
- single-task and multitask learning
- LoRA fine-tuning

### Drug discovery workflows

- property prediction
- lead optimization workflows
- dataset benchmarking
- uncertainty-aware model evaluation
- molecular generation
- virtual screening

### Infrastructure

- PyTorch-based training stack
- TOML/YAML configuration
- checkpointing
- CLI-based workflows
- Streamlit UI
- optional native Qt desktop workspace
- multi-GPU and single-GPU support

### Native desktop workspace

Install the optional desktop dependencies and launch ChemFlow Studio:

```bash
mamba activate chemflow
python -m pip install -e ".[desktop]"
chemflow-install-ketcher
chemflow-desktop
```

To launch the focused HLM/RLM/MLM clearance deployment interface instead:

```bash
admet-desktop
```

Ketcher is installed as an offline chemical structure editor for the desktop;
drawn molecules stay on the local computer.

The desktop combines local and Slurm/HPC training, a persistent job monitor,
prediction, similarity search, molecular generation, and uncertainty
evaluation. The training center supports conventional ML, Graphormer,
Chemprop, CheMeleon, ChemBERTa, KERMT, GPT, and LSTM. See the
[desktop guide](src/chemflow/desktop/README.md) for details.

---

## Source layout

All importable runtime code lives under the `chemflow` package:

```text
src/chemflow/
├── cli/
├── config/
├── deep_learning/
├── desktop/
├── machine_learning/
├── streamlit/
├── featurization/
└── utils/
```

Tests live in `tests/`, while one-off dataset and maintenance programs live in
`scripts/`. New code should import modules through `chemflow.*`.

---

## Installation

ChemFlow requires Python 3.11. The recommended installation uses
[Miniforge](https://github.com/conda-forge/miniforge) and Mamba. Two environment
specifications are provided:

- `environment.yaml` for local development, CPU systems, and Apple Silicon.
- `environment-hpc.yaml` for Linux HPC nodes with NVIDIA GPUs. It uses
  `requirements-hpc-torch.txt` to install the CUDA 12.6 PyTorch build.

Conda installs the compiled scientific packages first; pip then installs the
remaining Python packages and ChemFlow itself in editable mode.

### 1. Clone ChemFlow

```bash
git clone https://github.com/yonglanliu/ChemFlow.git
cd ChemFlow
```

### 2A. Local, CPU, or Apple Silicon installation

Run this command from the repository root:

```bash
mamba env create -f environment.yaml
mamba activate chemflow
```

If the `chemflow` environment already exists, update it instead:

```bash
mamba env update --name chemflow --file environment.yaml --prune
mamba activate chemflow
```

### 2B. NVIDIA GPU installation on an HPC cluster

Use this installation only on a Linux cluster with NVIDIA GPUs and a driver
compatible with CUDA 12.6. The NVIDIA driver is normally maintained by the HPC
administrators; do not install or replace the system driver inside the Conda
environment.

If your cluster uses environment modules, inspect the available CUDA versions
and load the module recommended by the cluster documentation:

```bash
module avail cuda
module load cuda/12.6  # module name may differ on your cluster
```

Create the HPC environment from the repository root:

```bash
mamba env create -f environment-hpc.yaml
mamba activate chemflow
```

To update an existing HPC environment:

```bash
mamba env update --name chemflow --file environment-hpc.yaml --prune
mamba activate chemflow
```

`environment-hpc.yaml` installs the packages pinned in
`requirements-hpc-torch.txt`, including the CUDA 12.6 builds of PyTorch and
Torchvision. A complete system CUDA Toolkit is generally unnecessary unless
you need to compile custom CUDA extensions.

Confirm that the active interpreter is Python 3.11:

```bash
python --version
which python
```

On Windows, use `where python` instead of `which python`.

### 3. Verify the installation

```bash
chemflow --help

python -c "import torch, chemprop, lightgbm; print(torch.__version__)"
```

On HPC, perform the GPU check from an allocated GPU compute node rather than a
login node:

```bash
nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

For an HPC installation, `CUDA available` should report `True`. A `False`
result on a login node does not necessarily indicate a broken installation,
because many clusters expose GPUs only inside scheduled jobs.

If you modify `requirements.txt`, synchronize the active environment with:

```bash
mamba env update --name chemflow --file environment.yaml --prune
```

If you modify `requirements-hpc-torch.txt`, update the HPC environment instead:

```bash
mamba env update --name chemflow --file environment-hpc.yaml --prune
```

### Alternative: pip virtual environment

Use this only when Conda/Mamba is unavailable. Compiled dependencies such as
RDKit, PyTorch, and LightGBM are generally easier to install through Conda.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Pretrained model weights

CheMeleon downloads its official pretrained weights automatically on the first
run and reuses the cached file afterward:

```text
~/.cache/chemflow/chemeleon/chemeleon_mp.pt
```

Set `CHEMFLOW_CACHE_DIR` before CheMeleon training to use another cache
location:

```bash
export CHEMFLOW_CACHE_DIR=/path/to/model_cache
```

Hugging Face Graphormer accepts either a Hub model through `model_name` or a
local PCQM4Mv1 checkpoint through `checkpoint_path` in `[ModelConfig]`. A local
checkpoint is recommended on compute nodes without internet access. The
included examples use:

```text
~/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt
```

ChemBERTa accepts a Hugging Face model ID or a local downloaded model directory
through `ModelConfig.model_name`. Install its optional dependency with
`pip install -e '.[chemberta]'`; use `local_files_only = true` on offline HPC
compute nodes. The included DeepChem checkpoint uses
`ModelConfig.architecture = "roberta"`. If an institutional proxy causes
`CERTIFICATE_VERIFY_FAILED`, configure `REQUESTS_CA_BUNDLE` with the trusted
institution CA or install `pip-system-certs`; do not disable TLS verification.

### Hardware notes

- NVIDIA multi-GPU Graphormer training is launched with `torchrun`; CheMeleon
  uses Lightning's `strategy = "ddp"` configuration.
- Apple Silicon uses the MPS GPU backend for single-device training. MPS does
  not support multi-device DDP.
- Keep `num_workers = 0` on macOS if multiprocessing or shared-memory errors
  occur.

To check accelerator availability:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

### Troubleshooting imports

If Python reports a missing package such as `lightgbm` or `chemprop`,
confirm that the `chemflow` environment is active and reinstall from the
environment specification:

```bash
mamba activate chemflow
mamba env update --name chemflow --file environment.yaml --prune
python -m pip check
```

---

## Quick start

### Train Hugging Face Graphormer

```bash
chemflow train hf-graphormer example/hf_graphormer_training/conf.toml
```

HF Graphormer and CheMeleon support single-task, homogeneous multitask, and
mixed regression/binary-classification training. Set `BaseConfig.task = "mixed"`
and provide `DatasetConfig.task_types` in target-column order; see
[the HF Graphormer tutorial](example/hf_graphormer_training/README.md).

### Train ChemBERTa

```bash
pip install -e '.[chemberta]'
chemflow train chemberta example/chemberta_training/conf.toml
```

See [the ChemBERTa tutorial](example/chemberta_training/README.md).

### Train a Chemprop v2 baseline

```bash
chemflow train chemprop example/chemprop_training/conf.toml
```

The example uses a predefined cluster-disjoint CL-3 validation split and the
independent CL-Test dataset. See the
[Chemprop baseline guide](example/chemprop_training/README.md).

### Fine-tune KERMT

ChemFlow vendors NVIDIA/Merck's KERMT architecture and fine-tuning runtime.
By default, the pinned official checkpoint and vocabularies are downloaded
from `nvidia/NV-KERMT-70M-v2` into the ChemFlow cache on first use:

```bash
chemflow train kermt example/kermt_training/conf.toml
```

ChemFlow prepares separate train/validation/test CSV files and records the
exact vendored command and pretrained-weight SHA256. NVIDIA HPC can use
`cuik_molmaker`; macOS uses the included RDKit fallback. See
[the KERMT guide](example/kermt_training/README.md).

### Fine-tune the pretrained CheMeleon model

Copy and edit `example/chemeleon_training/conf.toml`, then run:

```bash
chemflow train chemeleon example/chemeleon_training/conf.toml
```

For multitask training, set `target_column` to a TOML list in the same file.
The comments in each model's single `conf.toml` explain how to select
single-task, multitask, classification, mixed-task, external-test, and DDP
operation.

Missing multitask labels are masked automatically. Run prediction with the
task names stored in the checkpoint:

```bash
chemflow predict chemeleon \
  --input ./data/molecules.csv \
  --structure-column SMILES \
  --model-checkpoint ./path/to/checkpoints/best.ckpt \
  --output ./predictions.csv
```

See [the CheMeleon tutorial](example/chemeleon_training/README.md) for
multitask regression, classification, and prediction details.

The trainer uses the official CheMeleon ChemProp graph featurizer,
architecture, and pretrained message-passing weights. The weights are
downloaded from Zenodo on first use and validated by SHA-256. Each run writes
the resolved configuration, data split manifest, best and last checkpoints,
training history, hold-out metrics, and test predictions to its configured
work directory.

For multi-GPU CUDA training, set these fields under
`[CheMeleonTrainingConfig]` (the configured `batch_size` applies to each GPU):

```toml
accelerator = "gpu"
devices = 2
strategy = "ddp"
```

`num_nodes` and `sync_batchnorm` are also supported. Apple MPS remains a
single-device backend and cannot use DDP.

CheMeleon writes full-state `last.ckpt` checkpoints. To continue an
interrupted run, keep the same work directory and configuration, set the new
total epoch count, and enable:

```toml
[CheMeleonTrainingConfig]
resume = true
# resume_checkpoint = "/optional/custom/path/last.ckpt"
```

Older weights-only checkpoints can still be used for prediction or transfer
learning, but not optimizer-level resume.

To evaluate a completely independent CSV instead of splitting test rows from
the training file, configure the external path and disable the internal test
split:

```toml
[DatasetConfig]
dataset_path = "./data/train.csv"
test_dataset_path = "./data/external_test.csv"
smiles_column = "SMILES"
target_column = "target"
val_fraction = 0.1
test_fraction = 0.0
```

The external file must contain the configured SMILES and target columns. Its
rows are used only for final evaluation of the best validation checkpoint.

### Generate molecules

```bash
chemflow generate gpt \
    --checkpoint "${checkpoint_path}" \
    --tokenizer "${tokenizer_path}" \
    --adapter_checkpoint "${adapter_checkpoint_path}" \
    --output "${output_path}" \
    --num_samples 128 \
    --max_new_tokens 128 \
    --temperature 0.8
```

### Run uncertainty bootstrap evaluation

```bash
chemflow uncertainty bootstrap \
    --input prediction_output.csv \
    --task "TaskA:target_A:pred_A" \
    --task "TaskB:target_B:pred_B" \
    --metrics r2 mae rmse pearson spearman kendall \
    --n-bootstrap 2000 \
    --confidence-level 0.95 \
    --seed 42 \
    --plot-distributions \
    --output-dir bootstrap_results
```

---

## Supported model families

| Category | Models |
|-----------|--------|
| Graph | Chemprop v2, Hugging Face Graphormer, CheMeleon, KERMT |
| Sequence | ChemBERTa, LSTM, GPT |
| Fine-tuning | LoRA |
| Learning | single-task and multitask learning |

---

## Examples

### Molecular similarity

- ECFP4, ECFP6, FCFP4, FCFP6
- MACCS fingerprinting
- RDKit 2D descriptors

### Similarity methods

- Tanimoto
- cosine

### Molecular generation

- GPT-based generation
- LSTM generation

### Property prediction

- regression
- classification
- multitask prediction
- uncertainty-aware evaluation

---

## Future roadmap

- [ ] protein-ligand co-design
- [ ] pocket-conditioned generation
- [ ] molecular docking integration
- [ ] free energy calculation
- [ ] multimodal foundation models
