#!/bin/bash
#SBATCH --job-name=smiles_generate
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


checkpoint_path="/path/to/best_model.pt"
tokenizer_path="/path/to/cache/tokenizer"
adapter_checkpoint_path="/path/to/best_adapter.pt"
output_path="/path/to/generated_smiles.txt"

num_samples=1000
max_length=100
temperature=1.0
top_k=50

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

chemflow generate gpt \
    --checkpoint "${checkpoint_path}" \
    --tokenizer "${tokenizer_path}" \
    --adapter_checkpoint "${adapter_checkpoint_path}" \
    --output "${output_path}" \
    --num_samples "${num_samples}" \
    --max_new_tokens "${max_length}" \
    --temperature "${temperature}" \
    --top_k "${top_k}"
