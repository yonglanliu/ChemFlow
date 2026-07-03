
#!/bin/bash
#SBATCH --job-name=smiles_ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# module load cuda
source ./.venv/bin/activate

export OMP_NUM_THREADS=4
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false

# Run single GPU training or CPU training
python ./src/deep_learning/diffusion/train_ddp.py ./training_example/diffusion_training/conf.toml

# run distributed training with 4 GPUs
# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=4 \
#     ChemFlow/src/deep_learning/diffusion/train_ddp.py \
#     ChemFlow/diffusion_training/conf.toml