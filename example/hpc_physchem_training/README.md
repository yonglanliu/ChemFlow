# Physchem training on Slurm

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
