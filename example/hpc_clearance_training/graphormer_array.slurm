#!/usr/bin/env bash
#SBATCH --job-name=clearance-graphormer
#SBATCH --partition=gpu
#SBATCH --array=0-5%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00

set -euo pipefail

source /home/liuy48/bin/myconda
conda activate chemflow

training_dir="/data/liuy48/model_training/adme/clearance"
data_dir="$training_dir/dataset"
base_config="$training_dir/graphormer_conf.toml"
test_file="$data_dir/cl_test_splits.csv"
helper="$training_dir/prepare_config.py"

dataset_labels=("CL1" "CL2" "CL3")
dataset_files=("cl1_kmeans_splits.csv" "cl2_kmeans_splits.csv" "cl3_kmeans_splits.csv")
val_fractions=("0.1" "0.2")

array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( array_index < 0 || array_index >= 6 )); then
    echo "Invalid array index: $array_index; expected 0-5." >&2
    exit 1
fi
dataset_index="$(( array_index / 2 ))"
fraction_index="$(( array_index % 2 ))"
dataset_label="${dataset_labels[$dataset_index]}"
data_file="$data_dir/${dataset_files[$dataset_index]}"
val_fraction="${val_fractions[$fraction_index]}"
split_column="split_val_$val_fraction"
run_name="graphormer_${dataset_label}_val_${val_fraction}_seed42"
workdir="$training_dir/$run_name"
config_path="$training_dir/generated_configs/${run_name}.toml"

python "$helper" \
    --model graphormer \
    --base-config "$base_config" \
    --dataset "$data_file" \
    --test-dataset "$test_file" \
    --split-column "$split_column" \
    --workdir "$workdir" \
    --output "$config_path"

chemflow_executable="$(command -v chemflow || true)"
if [[ -z "$chemflow_executable" ]]; then
    echo "chemflow was not found in the activated environment." >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Run: $run_name"
echo "Dataset: $data_file"
echo "Independent test: $test_file"
echo "Split column: $split_column"
echo "Configuration: $config_path"
echo "Output: $workdir"
nvidia-smi
srun "$chemflow_executable" train hf-graphormer "$config_path"
