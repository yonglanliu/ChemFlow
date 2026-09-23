#!/usr/bin/env bash
#SBATCH --job-name=clearance-ml
#SBATCH --partition=norm
#SBATCH --array=0-17%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

source /home/liuy48/bin/myconda
conda activate chemflow

training_dir="/data/liuy48/model_training/adme/clearance"
data_dir="$training_dir/dataset"
base_config="$training_dir/ml_conf.yaml"
test_file="$data_dir/cl_test_splits.csv"
helper="$training_dir/prepare_config.py"

dataset_labels=("CL1" "CL2" "CL3")
dataset_files=("cl1_kmeans_splits.csv" "cl2_kmeans_splits.csv" "cl3_kmeans_splits.csv")
val_fractions=("0.1" "0.2")
endpoint_labels=("HLM" "RLM" "MLM")
target_columns=(
    "Log10_HLM_CLint_mL_min_kg"
    "Log10_RLM_CLint_mL_min_kg"
    "Log10_MLM_CLint_mL_min_kg"
)

array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( array_index < 0 || array_index >= 18 )); then
    echo "Invalid array index: $array_index; expected 0-17." >&2
    exit 1
fi
dataset_index="$(( array_index / 6 ))"
within_dataset="$(( array_index % 6 ))"
fraction_index="$(( within_dataset / 3 ))"
endpoint_index="$(( within_dataset % 3 ))"
dataset_label="${dataset_labels[$dataset_index]}"
data_file="$data_dir/${dataset_files[$dataset_index]}"
val_fraction="${val_fractions[$fraction_index]}"
split_column="split_val_$val_fraction"
endpoint="${endpoint_labels[$endpoint_index]}"
target="${target_columns[$endpoint_index]}"
run_name="ml_${dataset_label}_${endpoint}_val_${val_fraction}_seed42"
workdir="$training_dir/$run_name"
config_path="$training_dir/generated_configs/${run_name}.yaml"

python "$helper" \
    --model ml \
    --base-config "$base_config" \
    --dataset "$data_file" \
    --test-dataset "$test_file" \
    --split-column "$split_column" \
    --target "$target" \
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
export PYTHONUNBUFFERED=1

echo "Run: $run_name"
echo "Endpoint: $endpoint ($target)"
echo "Dataset: $data_file"
echo "Independent test: $test_file"
echo "Split column: $split_column"
echo "Representation: ECFP4"
echo "Configuration: $config_path"
echo "Output: $workdir"
srun "$chemflow_executable" train ml "$config_path"
