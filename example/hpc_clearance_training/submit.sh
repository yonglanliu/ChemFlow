#!/usr/bin/env bash
set -euo pipefail

training_dir="${1:-/data/liuy48/model_training/adme/clearance}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log_dir="$training_dir/logs"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found. Run this script on a Slurm login node." >&2
    exit 1
fi

mkdir -p "$log_dir"

chemeleon_job_id="$(
    sbatch \
        --parsable \
        --export="ALL,CLEARANCE_JOB_DIR=$training_dir" \
        --output="$log_dir/%x_%A_%a.out" \
        --error="$log_dir/%x_%A_%a.err" \
        "$script_dir/chemeleon_array.slurm"
)"

ml_job_id="$(
    sbatch \
        --parsable \
        --export="ALL,CLEARANCE_JOB_DIR=$training_dir" \
        --output="$log_dir/%x_%A_%a.out" \
        --error="$log_dir/%x_%A_%a.err" \
        "$script_dir/ml_array.slurm"
)"

echo "Submitted CheMeleon array: $chemeleon_job_id (4 GPU jobs)"
echo "Submitted ML array:        $ml_job_id (12 CPU jobs; at most 3 concurrent)"
echo "Logs: $log_dir"
echo "Monitor: squeue -j $chemeleon_job_id,$ml_job_id"
