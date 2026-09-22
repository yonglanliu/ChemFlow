#!/usr/bin/env bash
#SBATCH --job-name=caco2-chemeleon
#SBATCH --partition=gpu
#SBATCH --array=0-11%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
model_kind="chemeleon"
source "/data/liuy48/model_training/adme/adsorption/caco2/slurm/run_caco2_array.sh"
