#!/usr/bin/env bash
set -euo pipefail

training_dir="${1:-/data/liuy48/model_training/adme/physchem}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log_dir="$training_dir/logs"
devices="${HF_GRAPHORMER_DEVICES:-1}"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found. Run this script on a Slurm login node." >&2
    exit 1
fi

mkdir -p "$log_dir"
job_id="$(
    sbatch \
        --parsable \
        --gres="gpu:$devices" \
        --export="ALL,PHYSCHEM_JOB_DIR=$training_dir" \
        --output="$log_dir/%x_%j.out" \
        --error="$log_dir/%x_%j.err" \
        "$script_dir/hf_graphormer.slurm"
)"

echo "Submitted Hugging Face Graphormer job: $job_id"
echo "Logs: $log_dir"
echo "Monitor: squeue -j $job_id"
