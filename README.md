# ChemFlow

<p align="center">
An extensible AI-powered platform for molecular design, property prediction,
virtual screening, and cheminformatics.
</p>

---

## Overview

ChemFlow is an open-source Python framework for modern AI-driven drug discovery.

It integrates cheminformatics, deep learning, molecular generation, and molecular property prediction into a modular platform that supports both research and production workflows.

ChemFlow is designed to make it easy to:

- train AI models
- generate novel molecules
- predict molecular properties
- perform virtual screening
- analyze chemical datasets
- rapidly prototype new drug discovery pipelines

---

## Features

### Cheminformatics

- Public dataset curation
- Molecular similarity search
- Molecular descriptors
- Molecular fingerprints
- Molecular visualization
- File conversion
- Dataset processing


---

### Molecular AI

Support for

- Transformer models
- Graph Neural Networks
- Graphormer
- Diffusion models
- GPT/LLM molecular generators
- Single-/Multi-task learning
- LoRA fine-tuning

---

### Drug Discovery

- Molecular generation
- Property prediction
- Virtual screening
- Lead optimization
- Hit prioritization

---

### Infrastructure

- PyTorch
- Multi-GPU distributed training
- Checkpoint management
- YAML/TOML configuration
- Modular CLI
- Streamlit applications

---


## Installation

### 1. Install Python

ChemFlow requires **Python 3.10 or later**.

Verify your installation:

```bash
python --version
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/yonglanliu/ChemFlow.git
cd ChemFlow
```

---

### 3. Create a Virtual Environment (Recommended)

Using `venv`:

```bash
python -m venv .chemflow
```

Activate the environment:

**macOS/Linux**

```bash
source .chemflow/bin/activate
```

**Windows**

```bash
.chemflow\Scripts\activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Or, if installing ChemFlow as a package:

```bash
pip install -e .
```

---

### 5. Verify the Installation

```bash
chemflow --help
```

If the help message is displayed, ChemFlow has been installed successfully.


---

## Quick Start

Train a model

```bash
chemflow train graphormer exmaples/graphormer_classification.toml
chemflow train graphormer exmaples/graphormer_regression.toml
chemflow train gpt examples/gpt.toml
```

## HPC Multi-GPU training (compartible with single-/multi-GPU, and MPS training)
```bash

#!/bin/bash
#SBATCH --job-name=smiles_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load cuda
source .chemflow/bin/activate

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

CONFIG="gpt_training/conf.toml"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-0}

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running DDP on ${NUM_GPUS} GPUs"

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NUM_GPUS}" \
        -m src.cli.main \
        train graphormer \
        --config "${CONFIG}"

elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single GPU"

    chemflow train graphormer --config "${CONFIG}"

else
    echo "Running on CPU"

    chemflow train graphormer --config "${CONFIG}"
fi
```

Generate molecules

```bash
checkpoint_path = ""
adapter_checkpoint_path = ""   # if finetune, path for LoRA adaptor
tokeinzer_path = ""
output_path = ""
num_samples = "" # integer
max_length = "" # integer
temperature = "" # float
chemflow generate gpt \
    --checkpoint "${checkpoint_path}" \
    --tokenizer "${tokenizer_path}" \
    --adapter_checkpoint "${adapter_checkpoint_path}" \
    --output "${output_path}" \
    --num_samples "${num_samples}" \
    --max_new_tokens "${max_length}" \
    --temperature "${temperature}" \
    --top_k "${top_k}"
```

Predict properties

```bash
chemflow predict graphormer --smiles molecules.smi
chemflow predict graphormer --input .smi/.smiles/.txt/.csv/.parquet \
--model-checkpoint ${best_model_checkpoint} \
--batch_size 16 \
--num_workers 4 \ # Number of GPU workers, support multi-GPU for ultra large dataset
--output ${output_path.csv/.parquet/.pq}

```

Launch Streamlit

```bash
streamlit run app.py
```

---

## Supported Models

| Category | Models |
|-----------|--------|
| Graph | Graphormer |
| Sequence | LSTM, GPT |
| Diffusion | Molecular Diffusion |
| Fine-tuning | LoRA |
| Learning | Single-/Multi-task Learning |

---

## Examples

### Molecular Similarity

- Morgan Fingerprints
- MACCS
- RDKit Fingerprints

### Similarity Method
- Tanimoto
- cosine

### Molecular Generation

- GPT
- Diffusion
- Transformer

### Property Prediction

- Regression
- Classification
- Multi-task prediction


---

## Future Roadmap

- [ ] Protein–ligand co-design
- [ ] Pocket-conditioned generation
- [ ] Reinforcement learning
- [ ] Active learning
- [ ] Molecular docking integration
- [ ] Free energy calculation
- [ ] Multi-modal foundation models

