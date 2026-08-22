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

The dataset loader supports pre-split datasets that already contain a split label such as `train`, `val`, and `test`.

```toml
[DatasetConfig]
split_column = "split"
val_fraction = 0.1
test_fraction = 0.1
```

When `split_column` is provided, the loader respects the existing split rather than randomly splitting the data.

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
- multi-GPU and single-GPU support

---

## Installation

### 1. Install Python

ChemFlow requires Python 3.10 or later.

```bash
python --version
```

### 2. Clone the repository

```bash
git clone https://github.com/yonglanliu/ChemFlow.git
cd ChemFlow
```

### 3. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Or install in editable mode:

```bash
pip install -e .
```

### 5. Verify installation

```bash
chemflow --help
```

---

## Quick start

### Train a Graphormer model

```bash
chemflow train graphormer expansionrx_mtl_training/multitask_conf.toml
```

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
task_weight_method = "sqr_inverse"
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
| Diffusion | Molecular diffusion |
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
- diffusion generation
- transformer-based generation

### Property prediction

- regression
- classification
- multitask prediction
- uncertainty-aware evaluation

---

## Future roadmap

- [ ] protein-ligand co-design
- [ ] pocket-conditioned generation
- [ ] reinforcement learning
- [ ] active learning
- [ ] molecular docking integration
- [ ] free energy calculation
- [ ] multimodal foundation models

