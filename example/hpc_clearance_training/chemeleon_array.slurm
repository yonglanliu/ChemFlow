#!/usr/bin/env bash
#SBATCH --job-name=clearance-chemeleon
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

source "$HOME/bin/myconda"
conda activate chemflow

training_dir="${CLEARANCE_JOB_DIR:-/data/liuy48/model_training/adme/clearance}"
configs=(
    "multitask_conf.toml"
    "chemeleon_HLM_conf.toml"
    "chemeleon_MLM_conf.toml"
    "chemeleon_RLM_conf.toml"
)

config_path="$training_dir/${configs[$SLURM_ARRAY_TASK_ID]}"
if [[ ! -f "$config_path" ]]; then
    echo "Configuration file not found: $config_path" >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID}, array task: ${SLURM_ARRAY_TASK_ID}"
echo "Configuration: $config_path"
nvidia-smi || true

srun chemflow train chemeleon "$config_path"

echo "Finished: $(date --iso-8601=seconds)"
