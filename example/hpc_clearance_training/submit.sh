#!/usr/bin/env bash
set -euo pipefail

training_dir="/data/liuy48/model_training/adme/clearance"
script_dir="$training_dir"
log_dir="$training_dir/logs"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found. Run this script on a Slurm login node." >&2
    exit 1
fi
mkdir -p "$log_dir" "$training_dir/generated_configs"

submit_array() {
    local script_name="$1"
    sbatch \
        --parsable \
        --output="$log_dir/%x_%A_%a.out" \
        --error="$log_dir/%x_%A_%a.err" \
        "$script_dir/$script_name"
}

graphormer_job_id="$(submit_array graphormer_array.slurm)"
chemprop_job_id="$(submit_array chemprop_array.slurm)"
chemeleon_job_id="$(submit_array chemeleon_array.slurm)"
kermt_job_id="$(submit_array kermt_array.slurm)"
ml_job_id="$(submit_array ml_array.slurm)"

echo "Submitted Graphormer: $graphormer_job_id (6 GPU tasks; max 3 concurrent)"
echo "Submitted Chemprop:   $chemprop_job_id (6 GPU tasks; max 3 concurrent)"
echo "Submitted CheMeleon:  $chemeleon_job_id (6 GPU tasks; max 3 concurrent)"
echo "Submitted KERMT:      $kermt_job_id (6 GPU tasks; max 2 concurrent)"
echo "Submitted ML:         $ml_job_id (18 CPU tasks; max 6 concurrent)"
echo "Logs: $log_dir"
echo "Monitor: squeue -j $graphormer_job_id,$chemprop_job_id,$chemeleon_job_id,$kermt_job_id,$ml_job_id"
