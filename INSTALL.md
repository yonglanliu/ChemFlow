# Installing ChemFlow

ChemFlow requires Python 3.11. The supported installation uses Miniforge (or
another Conda-compatible distribution) and Mamba. Install ChemFlow and
AlchemForge in separate environments because their simulation and machine-
learning dependency stacks have different constraints.

## Prerequisites

Install these before creating the environment:

- Git
- Miniforge, Mambaforge, or another Conda-compatible installation
- Mamba (recommended) or Conda
- On an HPC system, access to a supported NVIDIA driver and GPU compute node

If Conda is installed but Mamba is not, install Mamba into the base
environment:

```bash
conda install --name base --channel conda-forge mamba
```

Verify the environment manager:

```bash
conda info --base
mamba --version
```

## Download ChemFlow

```bash
git clone https://github.com/yonglanliu/ChemFlow.git
cd ChemFlow
```

Run all environment creation and update commands from the repository root.
The requirements files contain the editable ChemFlow installation, so a
separate `pip install -e .` command is normally unnecessary.

## Local, CPU, and Apple Silicon installation

Create the local environment:

```bash
mamba env create --file environment.yaml
mamba activate chemflow
```

The local specification installs the compiled scientific stack with Conda and
the remaining Python packages with pip.

Verify the installation:

```bash
python --version
command -v python
chemflow --help
python -c "import torch, chemprop, lightgbm; print('PyTorch:', torch.__version__)"
python -m pip check
```

Python should report version 3.11, and the Python executable should belong to
the active `chemflow` environment.

## NVIDIA GPU installation on Linux or HPC

The GPU environment installs PyTorch built for CUDA 12.6. The NVIDIA kernel
driver is supplied by the operating system or cluster and must not be replaced
inside the Conda environment.

If the cluster requires a CUDA module, load the module recommended by its
documentation. The exact module name is site-specific:

```bash
module avail cuda
module load cuda/12.6
```

Create and activate the environment:

```bash
mamba env create --file environment-hpc.yaml
mamba activate chemflow
```

`descriptastorus` is intentionally installed from PyPI because it is not
provided by the configured Conda channels. PyTorch and Torchvision are
installed from the CUDA 12.6 PyTorch wheel index defined in
`requirements-hpc-torch.txt`.

Confirm the installation on the login node:

```bash
python --version
command -v python
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip check
```

Test CUDA only after obtaining a GPU allocation. For a Slurm cluster, a common
interactive request is:

```bash
srun \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=8G \
  --time=00:10:00 \
  --pty bash
```

Partition and resource names vary between clusters. Inside the allocation,
activate ChemFlow and run its real CUDA calculation:

```bash
mamba activate chemflow
nvidia-smi
chemflow gpu-test
```

A successful test ends with:

```text
CHEMFLOW GPU TEST: PASSED
```

Select another visible GPU or increase the diagnostic workload with:

```bash
chemflow gpu-test --device 1
chemflow gpu-test --matrix-size 2048 --iterations 5
```

## Updating an existing installation

Pull the source changes first:

```bash
git pull --ff-only
```

Update a local or CPU environment:

```bash
mamba env update \
  --name chemflow \
  --file environment.yaml \
  --prune
mamba activate chemflow
```

Update an NVIDIA HPC environment:

```bash
mamba env update \
  --name chemflow \
  --file environment-hpc.yaml \
  --prune
mamba activate chemflow
```

Then validate the installed dependencies and CLI:

```bash
python -m pip check
chemflow --help
```

## Optional model dependencies

Install optional extras only when required:

```bash
python -m pip install -e '.[chemberta]'
python -m pip install -e '.[hf-graphormer]'
python -m pip install -e '.[kermt]'
python -m pip install -e '.[data-curation]'
```

The environment specifications already include the dependencies used by the
standard bundled workflows. Extras are primarily useful for targeted or
minimal installations.

## Troubleshooting

### The wrong Python or Conda environment is active

```bash
conda info --base
conda env list
mamba activate chemflow
command -v python
```

The active Python must come from the `chemflow` environment. Environment names
are more portable than embedding a machine-specific absolute prefix in the
installation instructions.

### `descriptastorus` cannot be resolved by libmamba

Use the current `environment.yaml` or `environment-hpc.yaml`. In these files,
Conda resolves the compiled packages and pip installs `descriptastorus`. Do not
add `descriptastorus` to the Conda dependency list.

### CUDA is unavailable

Run `chemflow gpu-test` inside a scheduled GPU allocation. If it still fails,
check:

```bash
nvidia-smi
echo "${CUDA_VISIBLE_DEVICES:-not-set}"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A CPU-only PyTorch build reports `torch.version.cuda` as `None`. Recreate or
update the environment with `environment-hpc.yaml` in that case.

### Remove and recreate only the ChemFlow environment

```bash
mamba deactivate
mamba env remove --name chemflow
mamba env create --file environment.yaml
```

Use `environment-hpc.yaml` in the final command for the NVIDIA HPC build. This
removes the named environment only; it does not uninstall the base Conda
installation or delete the ChemFlow repository.

