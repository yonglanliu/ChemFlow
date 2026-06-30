
#!/bin/bash
#SBATCH --job-name=smiles_ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

mkdir -p logs

module load cuda
source /path/to/your/venv/bin/activate

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
PYTHONPATH=$(pwd):$PYTHONPATH

# Run single GPU training or CPU training
python ChemFlow/src/gpt/train_ddp.py /ChemFlow/gpt_training/conf.toml

# run distributed training with 4 GPUs
# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=4 \
#     ChemFlow/src/gpt/train_ddp.py \
#     ChemFlow/gpt_training/conf.toml