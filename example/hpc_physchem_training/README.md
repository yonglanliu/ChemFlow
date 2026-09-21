# Physchem training on Slurm

Submit `.slurm` files with `sbatch`; do not run them with `source`. Sourcing a
job file makes Bash ignore every `#SBATCH` directive and does not define
`SLURM_ARRAY_TASK_ID`.

## TensorBoard

CheMeleon, Graphormer, and ChemBERTa write TensorBoard events to the
`tensorboard` directory inside each run directory while retaining their CSV
histories. Start TensorBoard on the HPC host with, for example:

```bash
tensorboard --logdir /data/liuy48/model_training/adme/physchem \
  --host 127.0.0.1 --port 6006
```

Then forward the port from the local computer:

```bash
ssh -L 6006:127.0.0.1:6006 USER@HPC_LOGIN_HOST
```

Open `http://127.0.0.1:6006`. Set `tensorboard = false` in the model's
`TrainingConfig` to disable event logging, or change `tensorboard_dir` to use a
different run-relative location.

## Hugging Face Graphormer KSOL reference

The normal HPC `chemflow` environment includes the compatible Hugging Face
Graphormer dependencies. After pulling this branch, update that environment and
the editable package:

```bash
source "$HOME/bin/myconda"
conda activate chemflow
mamba install -c conda-forge "cython=0.29.37"
python -m pip install \
    "transformers==4.50.3" \
    "tokenizers==0.21.4" \
    "huggingface_hub==0.36.2"
python -m pip install -e /vf/users/liuy48/ChemFlow --no-deps
```

This keeps CheMeleon, traditional ML, and Hugging Face Graphormer in the same
`chemflow` environment. The pinned versions are `Cython 0.29.37`, Transformers
`4.50.3`, Tokenizers `0.21.4`, and Hugging Face Hub `0.36.2`.

The default job expects:

```text
/data/liuy48/model_training/adme/physchem/dataset/physchem_multitask.csv
$HOME/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt
```

Submit one full-fine-tuning job:

```bash
bash example/hpc_physchem_training/submit_hf_graphormer.sh
```

For single-node four-GPU DDP:

```bash
export HF_GRAPHORMER_DEVICES=4
export HF_GRAPHORMER_MODE=multitask
bash example/hpc_physchem_training/submit_hf_graphormer.sh
```

The submission helper requests that number of GPUs and the job launches one
process per GPU with `torchrun`. `HF_GRAPHORMER_BATCH_SIZE` is the per-GPU
batch size. CPU threads and default data workers are divided among processes.

CheMeleon already uses Lightning DDP. To give each CheMeleon array task four
GPUs, override both the Slurm GPU request and the array concurrency:

```bash
export CHEMELEON_DEVICES=4
sbatch --gres=gpu:4 --array=0-4%1 \
  example/hpc_physchem_training/chemeleon_array.slurm
```

The script sets `devices=4` and `strategy="ddp"` in each generated CheMeleon
configuration. Its batch size is also per GPU.

To resume interrupted CheMeleon array tasks from each run directory's
`checkpoints/last.ckpt`, increase the configured total epoch count if needed
and submit with:

```bash
export CHEMELEON_RESUME=true
sbatch example/hpc_physchem_training/chemeleon_array.slurm
```

Completed tasks will immediately stop when their checkpoint epoch already
meets the configured `num_epochs`.

The default is the single-task pH 7.4 KSOL model. To train one masked
multitask model for LogD, KSOL pH 6.8, KSOL pH 7.4, and ExpansionRX KSOL:

```bash
export HF_GRAPHORMER_MODE=multitask
bash example/hpc_physchem_training/submit_hf_graphormer.sh
```

Override locations or training settings before submission when necessary:

```bash
export PHYSCHEM_DATA_FILE=/data/liuy48/datasets/physchem_multitask.csv
export HF_GRAPHORMER_CHECKPOINT=/data/liuy48/checkpoints/graphormer-base-pcqm4mv1.pt
export HF_GRAPHORMER_BATCH_SIZE=8
export HF_GRAPHORMER_NUM_WORKERS=4
export HF_GRAPHORMER_EPOCHS=30
bash example/hpc_physchem_training/submit_hf_graphormer.sh
```

To resume the same output directory after interruption or wall-time expiry:

```bash
export HF_GRAPHORMER_RESUME=true
bash example/hpc_physchem_training/submit_hf_graphormer.sh
```

To add or rebuild fingerprint, embedding, and validation-calibration artifacts
for an existing best model without retraining it:

```bash
export HF_GRAPHORMER_MODE=multitask  # or single
export HF_GRAPHORMER_APPLICABILITY_ONLY=true
bash example/hpc_physchem_training/submit_hf_graphormer.sh
```

The job continues from the selected run directory's `checkpoints/last.pt`.

The job requests one GPU, eight CPUs, 64 GB host memory, and 48 hours. Reduce
the batch size if GPU memory is insufficient. Output is written to
`<training-directory>/hf_graphormer_KSol_pH7_4` for single-task mode or
`<training-directory>/hf_graphormer_physchem_multitask` for multitask mode,
including the best model,
training history, metrics, saved split assignments, and test predictions.

The checkpoint is loaded offline, so compute nodes do not need direct access to
Hugging Face and are unaffected by institutional HTTPS certificate interception.

## Five-run Graphormer array

`graphormer_array.slurm` generates and trains five configurations:

1. LogD, KSOL pH 6.8, KSOL pH 7.4, and ExpansionRX KSOL multitask
2. LogD single-task
3. KSOL pH 6.8 single-task
4. KSOL pH 7.4 single-task
5. ExpansionRX KSOL single-task

Submit all five, with at most four one-GPU jobs running concurrently:

```bash
sbatch example/hpc_physchem_training/graphormer_array.slurm
```

To reduce concurrent GPU use, override the array throttle, for example:

```bash
sbatch --array=0-4%2 example/hpc_physchem_training/graphormer_array.slurm
```

Set `HF_GRAPHORMER_RESUME=true` before submission to resume each run from its
own output directory. Paths, batch size, worker count, epochs, split type, and
checkpoint can be overridden with the environment variables documented at the
top of the script.

## Five-run ChemBERTa array

Before submitting from a compute node without internet access, populate the
shared Hugging Face cache from a login node:

```bash
hf download DeepChem/ChemBERTa-77M-MLM
```

Then submit the matching five-run ChemBERTa array:

```bash
sbatch example/hpc_physchem_training/chemberta_array.slurm
```

The task mapping is identical to the Graphormer array. To use a downloaded
model directory instead of the Hugging Face cache:

```bash
export CHEMBERTA_MODEL_NAME=/data/liuy48/models/ChemBERTa-77M-MLM
sbatch example/hpc_physchem_training/chemberta_array.slurm
```

Use `CHEMBERTA_RESUME=true` to resume interrupted runs and
`CHEMBERTA_APPLICABILITY_ONLY=true` to rebuild applicability/calibration
artifacts without further optimization. Override array concurrency with
`sbatch --array=0-4%2 ...` when fewer GPUs should run simultaneously.

For CheMeleon, Graphormer, or ChemBERTa, reuse an existing dataset `split`
column with either of these equivalent forms:

```bash
export PHYSCHEM_SPLIT_COLUMN=split
```

```bash
export CHEMELEON_SPLIT_TYPE=predefined       # CheMeleon
export HF_GRAPHORMER_SPLIT_TYPE=predefined   # Graphormer
export CHEMBERTA_SPLIT_TYPE=predefined       # ChemBERTa
```

When `predefined` is selected, the scripts default to the column name `split`
and do not pass `predefined` to the molecular split algorithm.

CheMeleon, Graphormer, and ChemBERTa can evaluate a physically separate test
CSV. The training file is split into training and validation only, and the
external file is used only for final testing:

```bash
export PHYSCHEM_DATA_FILE=/path/to/training.csv
export PHYSCHEM_TEST_DATA_FILE=/path/to/independent_test.csv
sbatch example/hpc_physchem_training/chemeleon_array.slurm
# or: sbatch example/hpc_physchem_training/graphormer_array.slurm
# or: sbatch example/hpc_physchem_training/chemberta_array.slurm
```

The generated configuration sets `test_fraction = 0.0` whenever
`PHYSCHEM_TEST_DATA_FILE` is present. Both files must contain `SMILES` and every
target required by the selected array task.
