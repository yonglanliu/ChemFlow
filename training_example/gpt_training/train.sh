#!/bin/bash
#SBATCH --job-name=smiles_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4


cd "$SLURM_SUBMIT_DIR"


module load cuda
source /path/to/your/venv/bin/activate

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

CONFIG="./training_example/gpt_training/conf.toml"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-0}

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running DDP on ${NUM_GPUS} GPUs"

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NUM_GPUS}" \
        ./src/deep_learning/gpt/train_ddp.py \
        "${CONFIG}"

elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single GPU"

    python ./src/deep_learning/gpt/train_ddp.py "${CONFIG}"

else
    echo "Running on CPU"

    python ./src/deep_learning/gpt/train_ddp.py "${CONFIG}"
fi