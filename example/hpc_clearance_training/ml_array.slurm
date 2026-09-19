#!/usr/bin/env bash
#SBATCH --job-name=clearance-ml
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

source "$HOME/bin/myconda"
conda activate chemflow

training_dir="${CLEARANCE_JOB_DIR:-/data/liuy48/model_training/adme/clearance}"
configs=(
    "ml_HLM_conf.yaml"
    "ml_MLM_conf.yaml"
    "ml_RLM_conf.yaml"
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

srun chemflow train ml "$config_path"

echo "Finished: $(date --iso-8601=seconds)"
