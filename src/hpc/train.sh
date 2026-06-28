#!/bin/bash
#SBATCH --job-name=smiles_ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module load cuda
source /path/to/your/venv/bin/activate

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  train.py