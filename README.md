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

### Multitask Graphormer with grouped adaptors

ChemFlow now supports grouped multitask adaptor architectures, where related tasks share an adaptor while retaining task-specific output heads.

Key features:

- grouped task sharing via `task_groups`
- configurable `num_adapters`
- support for hard-sharing and soft-sharing multitask variants
- task-to-adaptor mapping with explicit group assignments

Example configuration:

```toml
[GraphormerConfig]
sharing_type = "hard"
num_adapters = 4
task_groups = [[0, 1], [2, 3], [4, 5], [6, 7, 8]]
```

This enables task families such as assay groups to share a common adaptor while preserving independent outputs per task.

### Split-aware dataset loading

The Graphormer dataset loader supports molecule-aware global splits, predefined
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

Graphormer also downloads and caches the official Microsoft PCQM4M v1
checkpoint when `GraphormerConfig.pretrained_path` is omitted. A custom URL,
path, and SHA-256 checksum can be configured when needed.

### Automatic task weighting

Task weights can now be computed automatically from label availability.

Supported methods:

- `inverse`
- `sqrt_inverse` or `sqr_inverse`
- `customed`

Example:

```toml
[GraphormerConfig]
task_weight_method = "sqr_inverse"
# task_weights = [0.09, 0.10, 0.10, 0.12, 0.11, 0.11, 0.15, 0.12, 0.10]
```

This is useful for multitask datasets with unequal data availability across tasks.

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

- Graphormer
- LSTM
- GPT-style generation
- diffusion-based molecular generation
- single-task and multitask learning
- LoRA fine-tuning
- grouped task adaptors

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
- optional native Qt and PyOpenGL desktop workspace
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

The desktop combines training, prediction, similarity search, molecular
generation, and uncertainty evaluation with a live task console. See the
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
`scripts/`. New code should import modules through `chemflow.*`. A small
compatibility package keeps historical `src.deep_learning.*` and
`src.chemflow.*` paths loadable for older checkpoints and serialized models.

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

If the cluster loads an incompatible system C++ runtime and reports errors such
as `GLIBCXX_* not found`, install an activation hook that gives the Conda
environment's `libstdc++` precedence:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/chemflow.sh" <<'EOF'
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
EOF
```

Add a matching deactivation hook so the preload is removed when leaving the
`chemflow` environment:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/chemflow.sh" <<'EOF'
export LD_PRELOAD="${LD_PRELOAD#"$CONDA_PREFIX/lib/libstdc++.so.6"}"
export LD_PRELOAD="${LD_PRELOAD#:}"
if [[ -z "$LD_PRELOAD" ]]; then
    unset LD_PRELOAD
fi
EOF
```

Reactivate the environment after creating the hooks:

```bash
mamba deactivate
mamba activate chemflow
```

Use this workaround only when the HPC runtime requires it; `LD_PRELOAD`
affects every dynamically linked program launched from the active environment.

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

python -c "import torch, torch_geometric, chemprop, lightgbm; print(torch.__version__)"
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

### Pretrained model downloads

The first Graphormer or CheMeleon training run downloads its official
pretrained weights automatically. Later runs reuse the cached files:

```text
~/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt
~/.cache/chemflow/chemeleon/chemeleon_mp.pt
```

Set `CHEMFLOW_CACHE_DIR` before training to use another cache location:

```bash
export CHEMFLOW_CACHE_DIR=/path/to/model_cache
```

You can also configure a local `pretrained_path` in the model's TOML section.
This is useful on compute nodes without internet access.

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

If Python reports a missing package such as `lightgbm` or `torch_geometric`,
confirm that the `chemflow` environment is active and reinstall from the
environment specification:

```bash
mamba activate chemflow
mamba env update --name chemflow --file environment.yaml --prune
python -m pip check
```

---

## Quick start

### Train a Graphormer model

```bash
chemflow train graphormer expansionrx_mtl_training/multitask_conf.toml
```

### Fine-tune the pretrained CheMeleon model

Copy and edit `example/chemeleon_training/regression_conf.toml`, then run:

```bash
chemflow train chemeleon example/chemeleon_training/regression_conf.toml
```

For multitask training, set `target_column` to a TOML list or use the included
example:

```bash
chemflow train chemeleon example/chemeleon_training/multitask_conf.toml
```

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

### Train a standard graphormer model with a config file

```bash
graphormer_training/config.json
```

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

### Predict properties

```bash
chemflow predict graphormer \
    --input molecules.csv \
    --task-names task_1,task_2,task_3 \
    --model-checkpoint ${best_model_checkpoint} \
    --batch_size 16 \
    --num_workers 4 \
    --output prediction_output.csv
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

## Example configuration pattern

A typical multitask configuration now looks like:

```toml
[DatasetConfig]
dataset_path = "/path/to/dataset.csv"
smiles_column = "SMILES"
target_column = ["TaskA", "TaskB", "TaskC"]
task_names = ["TaskA", "TaskB", "TaskC"]
split_column = "split"

[GraphormerConfig]
num_tasks = 3
loss_type = "laplace_NLL"
task_weight_method = "sqrt_inverse"  # "sqrt_inverse" | "inverse" | "customed"
sharing_type = "hard"
num_adapters = 2
task_groups = [[0, 1], [2]]
```

This is useful for assay groups, multi-endpoint prediction, and mixed task availability settings.

---

## Supported model families

| Category | Models |
|-----------|--------|
| Graph | Graphormer |
| Sequence | LSTM, GPT |
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
