# Running ChemFlow free-energy & PROTAC tools on HPC

This guide is for a **new user** setting up the free-energy (OpenFE, GROMACS/PMX,
OpenMM), PyRosetta and REINVENT4 workflows on a SLURM GPU cluster. The ChemFlow
desktop is optional on HPC — the same work runs from the command line, and the
desktop just builds these commands for you.

The free-energy tools (OpenFE, OpenMM, GROMACS, PMX, Open Babel, ACPYPE,
AmberTools) all live in **one** conda env named `alchemforge`. There is no
separate `openfe` or `gmxfe` env.

## 0. One-time: make conda available

Most clusters don't put conda on `PATH` by default. Load it however your site
does, for example:

```bash
module load miniconda          # or: module load mamba / anaconda
# or, if you installed your own:
source <conda-prefix>/etc/profile.d/conda.sh
```

## 1. Create the `alchemforge` environment

Two equivalent routes. Both use the same package pins
(`env/environment-alchemforge-hpc.yaml` and `chemflow fe-env install` are kept in sync
by a test).

**A. From the yaml (works without ChemFlow installed first).** Run from any
directory; the HPC file is headless (no Qt/GUI libraries) and includes the CUDA
12.6 toolkit:

```bash
conda env create -n alchemforge -f /path/to/ChemFlow/environment-alchemforge-hpc.yaml
conda activate alchemforge
python -m pip install -e /path/to/ChemFlow --no-deps --no-build-isolation        # chemflow + alchemforge
python -m pip install /path/to/ChemFlow/pmx --no-deps --no-build-isolation       # PMX (hybrid topologies)
```

**B. With the ChemFlow installer** (needs `chemflow` importable in some Python,
e.g. a base ML env where you ran `pip install -e /path/to/ChemFlow`):

```bash
chemflow fe-env install --cuda 12.6      # creates env 'alchemforge' (python 3.12) + links chemflow + PMX
```

`--cuda 12.6` adds `cuda-version`/`cuda-toolkit`, sets `CONDA_OVERRIDE_CUDA` (so the
GPU build resolves even from a **GPU-less login node**), and selects the **CUDA
GROMACS** build (`gromacs=*=nompi_cuda*`) instead of conda-forge's OpenCL build.
Pin the CUDA version to one your **node's driver supports** — `nvidia-smi` shows the
maximum as "CUDA Version: X.Y" top-right; choosing a higher toolkit gives OpenMM
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` at runtime. CUDA 12.6 still supports Pascal
GPUs (e.g. P100); newer GPUs such as Blackwell need a newer CUDA pin (`--cuda 12.8`
or later). Omit `--cuda` on a CPU-only machine. Override the GROMACS build with
`--gromacs-spec` if needed. Add `--dry-run` to print the commands first. On a
low-memory login node where the solve is OOM-killed (`exit -9`), add `--staged`.

Other tools keep their own envs:

```bash
chemflow rosetta-env install                   # PyRosetta (CPU; needs a RosettaCommons license)
chemflow reinvent-env install --reinvent-src /path/to/REINVENT4
```

Check what was detected at any time, and test the GPU from an allocated GPU node
(not the login node):

```bash
chemflow fe-env status
chemflow rosetta-env status
chemflow reinvent-env status
alchemforge gpu-test
```

### Troubleshooting `alchemforge gpu-test`

Run it **inside an allocated GPU job**, never the login node, and check `nvidia-smi`
first so you know the driver and its maximum CUDA version.

- **OpenMM: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)`** — the env's CUDA toolkit is
  newer than the node's driver. Reinstall with a CUDA pin at or below the driver's
  "CUDA Version" from `nvidia-smi`, e.g. `chemflow fe-env install --env alchemforge
  --cuda 12.2 --recreate` (or just `mamba install -n alchemforge -c conda-forge
  cuda-version=12.2 cuda-toolkit=12.2`).
- **GROMACS: `GPU support: OpenCL` / `Cannot run short-ranged nonbonded interactions
  on a GPU because no GPU is detected`** — the OpenCL GROMACS build was installed
  instead of the CUDA one. Reinstall the CUDA variant: `mamba install -n alchemforge
  -c conda-forge "gromacs=*=nompi_cuda*"`, or re-run the installer with `--cuda`
  (which now selects that build automatically).

After fixing both, re-run `alchemforge gpu-test`; a healthy node prints `[PASS]` for
OpenMM and GROMACS and ends with `GPU TEST: PASSED`.

## 2. Link the envs (only if conda lives somewhere unusual)

The tools auto-detect an env named `alchemforge` (then legacy `openfe` / `fep`),
`pyrosetta` and `reinvent` under the common conda roots. If your conda is in a
non-standard place, point at the interpreters explicitly, either with environment
variables:

```bash
export CHEMFLOW_FE_PYTHON=<conda-prefix>/envs/alchemforge/bin/python
export CHEMFLOW_GMX_PYTHON=$CHEMFLOW_FE_PYTHON            # GROMACS steps use the same env
export CHEMFLOW_ROSETTA_PYTHON=<conda-prefix>/envs/pyrosetta/bin/python
export REINVENT_COMMAND=<conda-prefix>/envs/reinvent/bin/reinvent
```

or once in `chemflow_tools.yaml` (see the one in the repo root; copy it to
`~/.config/chemflow/tools.yaml` to apply everywhere):

```yaml
fe:       {python: <conda-prefix>/envs/alchemforge/bin/python}
gromacs:  {python: <conda-prefix>/envs/alchemforge/bin/python}
```

Put the exports in your `~/.bashrc` or job script so batch jobs see them.

## 3. Run OpenFE RBFE

Build the config (desktop "Binding Free Energy · OpenFE → RBFE" fills these in),
then:

```bash
conda activate alchemforge
alchemforge openfe charge  --ligands ligands.sdf --out ligands_charged.sdf   # charge
alchemforge openfe rbfe validate rbfe.yaml
alchemforge openfe rbfe generate rbfe.yaml                                   # create_json
```

Submit the production FEP with a SLURM script. The desktop's **Job (HPC)** tab
writes one from your fields (partition, GPUs, walltime, **Modules**, conda init
+ env). A minimal one looks like:

```bash
#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
set -euo pipefail
source <conda-prefix>/etc/profile.d/conda.sh           # conda init script
conda activate alchemforge                             # conda environment
python -m alchemforge.cli openfe rbfe run rbfe.yaml
```

The conda env already carries the CUDA user-space toolkit, so no `module load
cuda` is needed; only the cluster's NVIDIA driver must be present on the node.

## 4. Run GROMACS/PMX RBFE

Same env, same config flow (see `src/chemflow/fe/gromacs/rbfe/README.md`):

```bash
conda activate alchemforge
alchemforge rbfe plan     my_rbfe.toml
alchemforge ligand-rbfe stage my_rbfe.toml
```

- Independent-window sampling runs with the conda env's `gmx`.
- **Hamiltonian replica exchange (HREX) needs an external MPI GROMACS** (normally
  `gmx_mpi`, e.g. from a cluster module). The conda `gmx` is thread-MPI and cannot
  run `-multidir`. Set the *HREX module* / *HREX gmx command* on the Job tab, or
  `[execution].hrex_module` / `hrex_gromacs_command` in the TOML.

## 5. MM-GBSA triage at scale

One array task per candidate; the array script can load modules + activate conda
so each task is self-contained:

```bash
alchemforge mmgbsa batch /path/to/md_runs --ligand-selection "resname LIG" \
    --partition gpu --conda-init <conda-prefix>/etc/profile.d/conda.sh --conda-env alchemforge
sbatch /path/to/md_runs/mmgbsa_batch/mmgbsa_array.slurm
alchemforge mmgbsa aggregate /path/to/md_runs/mmgbsa_batch     # ranked CSV
```

## 6. Notes for shared filesystems

- Install envs on a filesystem the compute nodes can read (home or project
  space), not node-local scratch.
- Generated scripts reference the env interpreter by absolute path and/or
  activate conda, so they don't rely on the submitting shell's environment.

## Quick reference

| Tool                          | Install                                                | Link / detect                                  |
|-------------------------------|--------------------------------------------------------|------------------------------------------------|
| OpenFE + GROMACS/PMX + OpenMM | `chemflow fe-env install --cuda X` or the HPC yaml     | env `alchemforge` / `CHEMFLOW_FE_PYTHON`, `CHEMFLOW_GMX_PYTHON` |
| PyRosetta                     | `chemflow rosetta-env install`                         | env `pyrosetta` / `CHEMFLOW_ROSETTA_PYTHON`    |
| REINVENT4                     | `chemflow reinvent-env install`                        | `reinvent` exe / `REINVENT_COMMAND`            |
