#!/bin/bash
#SBATCH --job-name=smiles_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load cuda
source /path/to/your/venv/bin/activate

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

CONFIG="./example/graphormer_training/classification_conf.toml"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-0}

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running DDP on ${NUM_GPUS} GPUs"

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NUM_GPUS}" \
        -m src.cli.main \
        train graphormer \
        --config "${CONFIG}"

elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single GPU"

    chemflow train graphormer --config "${CONFIG}"

else
    echo "Running on CPU"

    chemflow train graphormer --config "${CONFIG}"
fi