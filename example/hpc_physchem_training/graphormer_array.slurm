#!/usr/bin/env bash
#SBATCH --job-name=physchem-graphormer
#SBATCH --partition=gpu
# One multitask model plus four single-task models.
# Up to four tasks may run concurrently, using four GPUs in total.
#SBATCH --array=0-4%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

set -euo pipefail

source "$HOME/bin/myconda"
conda activate chemflow

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"

# These locations can be overridden at submission time, for example:
#   PHYSCHEM_JOB_DIR=/data/liuy48/model_training/adme/physchem \
#   PHYSCHEM_DATA_FILE=/data/liuy48/model_training/adme/physchem/dataset/physchem_multitask.csv \
#   PHYSCHEM_TEST_DATA_FILE=/data/liuy48/model_training/adme/physchem/dataset/physchem_test.csv \
#   sbatch graphormer_array.slurm
training_dir="${PHYSCHEM_JOB_DIR:-/data/liuy48/model_training/adme/physchem}"
base_config="${HF_GRAPHORMER_BASE_CONFIG:-$repo_dir/example/hf_graphormer_training/conf.toml}"
data_file="${PHYSCHEM_DATA_FILE:-$training_dir/dataset/physchem_multitask.csv}"
test_data_file="${PHYSCHEM_TEST_DATA_FILE:-}"
checkpoint="${HF_GRAPHORMER_CHECKPOINT:-$HOME/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt}"
split_type="${HF_GRAPHORMER_SPLIT_TYPE:-scaffold_balanced}"
split_column="${PHYSCHEM_SPLIT_COLUMN:-}"
generated_config_dir="$training_dir/generated_configs"

run_names=(
    "multitask"
    "LogD_pH7_4"
    "KSol_pH6_8"
    "KSol_pH7_4"
    "KSol_ExpansionRX"
)
target_specs=(
    "LogD_pH7_4,Log_KSOL_pH6_8_mol_L,Log_KSOL_pH7_4_mol_L,Log_KSOL_ExpansionRX_mol_L"
    "LogD_pH7_4"
    "Log_KSOL_pH6_8_mol_L"
    "Log_KSOL_pH7_4_mol_L"
    "Log_KSOL_ExpansionRX_mol_L"
)

array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
task_count="${#run_names[@]}"
if (( array_index < 0 || array_index >= task_count )); then
    echo "Invalid array index: $array_index; expected 0-$((task_count - 1))." >&2
    exit 1
fi

run_name="${run_names[$array_index]}"
target_spec="${target_specs[$array_index]}"
workdir="$training_dir/graphormer_$run_name"
config_path="$generated_config_dir/graphormer_${run_name}_conf.toml"

for required_file in "$base_config" "$data_file" "$checkpoint"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done
if [[ -n "$test_data_file" && ! -f "$test_data_file" ]]; then
    echo "External test dataset not found: $test_data_file" >&2
    exit 1
fi

case "$split_type" in
    random|random_with_repeated_smiles|scaffold_balanced|kennard_stone|kmeans) ;;
    predefined)
        split_column="${split_column:-split}"
        ;;
    *)
        echo "Unsupported HF_GRAPHORMER_SPLIT_TYPE: $split_type" >&2
        exit 1
        ;;
esac

mkdir -p "$generated_config_dir" "$workdir"

export CHEMFLOW_HF_BASE_CONFIG="$base_config"
export CHEMFLOW_HF_GENERATED_CONFIG="$config_path"
export CHEMFLOW_HF_DATA_FILE="$data_file"
export CHEMFLOW_HF_TEST_DATA_FILE="$test_data_file"
export CHEMFLOW_HF_CHECKPOINT="$checkpoint"
export CHEMFLOW_HF_RUN_NAME="$run_name"
export CHEMFLOW_HF_TARGETS="$target_spec"
export CHEMFLOW_HF_WORKDIR="$workdir"
export CHEMFLOW_HF_SPLIT_TYPE="$split_type"
export CHEMFLOW_HF_SPLIT_COLUMN="$split_column"
export CHEMFLOW_HF_BATCH_SIZE="${HF_GRAPHORMER_BATCH_SIZE:-8}"
export CHEMFLOW_HF_NUM_WORKERS="${HF_GRAPHORMER_NUM_WORKERS:-8}"
export CHEMFLOW_HF_EPOCHS="${HF_GRAPHORMER_EPOCHS:-30}"
export CHEMFLOW_HF_RESUME="${HF_GRAPHORMER_RESUME:-false}"
export CHEMFLOW_HF_APPLICABILITY_ONLY="${HF_GRAPHORMER_APPLICABILITY_ONLY:-false}"

python - <<'PY'
import os
from pathlib import Path

import pandas as pd
import toml

base_path = Path(os.environ["CHEMFLOW_HF_BASE_CONFIG"])
output_path = Path(os.environ["CHEMFLOW_HF_GENERATED_CONFIG"])
data_path = Path(os.environ["CHEMFLOW_HF_DATA_FILE"])
test_data_file = os.environ["CHEMFLOW_HF_TEST_DATA_FILE"].strip()
checkpoint_path = Path(os.environ["CHEMFLOW_HF_CHECKPOINT"]).expanduser()
run_name = os.environ["CHEMFLOW_HF_RUN_NAME"]
workdir = Path(os.environ["CHEMFLOW_HF_WORKDIR"])
split_column = os.environ["CHEMFLOW_HF_SPLIT_COLUMN"].strip()
targets = [
    value.strip()
    for value in os.environ["CHEMFLOW_HF_TARGETS"].split(",")
    if value.strip()
]

config = toml.load(base_path)
if not isinstance(config, dict):
    raise ValueError(f"Base configuration must be a TOML mapping: {base_path}")

available_columns = set(pd.read_csv(data_path, nrows=0).columns)
required_columns = {"SMILES", *targets}
if split_column:
    required_columns.add(split_column)
missing_columns = sorted(required_columns - available_columns)
if missing_columns:
    raise ValueError(
        f"Dataset {data_path} is missing required columns: {missing_columns}. "
        f"Available columns: {sorted(available_columns)}"
    )
if test_data_file:
    test_columns = set(pd.read_csv(test_data_file, nrows=0).columns)
    missing_test_columns = sorted({"SMILES", *targets} - test_columns)
    if missing_test_columns:
        raise ValueError(
            f"External test dataset {test_data_file} is missing required "
            f"columns: {missing_test_columns}. Available columns: "
            f"{sorted(test_columns)}"
        )

base = config.setdefault("BaseConfig", {})
base["workdir"] = str(workdir)
base["task"] = "regression"

dataset = config.setdefault("DatasetConfig", {})
dataset["dataset_path"] = str(data_path)
if test_data_file:
    dataset["test_dataset_path"] = test_data_file
    dataset["test_fraction"] = 0.0
else:
    dataset.pop("test_dataset_path", None)
dataset["smiles_column"] = "SMILES"
dataset["target_column"] = targets if run_name == "multitask" else targets[0]
if split_column:
    dataset["split_column"] = split_column
    dataset["split_type"] = "predefined"
else:
    dataset.pop("split_column", None)
    dataset["split_type"] = os.environ["CHEMFLOW_HF_SPLIT_TYPE"]

model = config.setdefault("ModelConfig", {})
model["checkpoint_path"] = str(checkpoint_path)
model["local_files_only"] = True
model["freeze_encoder"] = False

training = config.setdefault("TrainingConfig", {})
training["device"] = "cuda"
training["devices"] = 1
training["strategy"] = "auto"
training["batch_size"] = int(os.environ["CHEMFLOW_HF_BATCH_SIZE"])
training["num_workers"] = int(os.environ["CHEMFLOW_HF_NUM_WORKERS"])
training["num_epochs"] = int(os.environ["CHEMFLOW_HF_EPOCHS"])
training["resume"] = os.environ["CHEMFLOW_HF_RESUME"].lower() in {
    "1", "true", "yes", "on"
}
training["applicability_only"] = os.environ[
    "CHEMFLOW_HF_APPLICABILITY_ONLY"
].lower() in {"1", "true", "yes", "on"}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as stream:
    toml.dump(config, stream)
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID}; array task: ${SLURM_ARRAY_TASK_ID}"
echo "Run: $run_name"
echo "Targets: $target_spec"
echo "Dataset: $data_file"
if [[ -n "$test_data_file" ]]; then
    echo "External test dataset: $test_data_file"
fi
echo "Checkpoint: $checkpoint"
if [[ -n "$split_column" ]]; then
    echo "Split column: $split_column"
else
    echo "Split type: $split_type"
fi
echo "Base configuration: $base_config"
echo "Generated configuration: $config_path"
echo "Output directory: $workdir"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("Slurm allocated a GPU but PyTorch cannot see CUDA.")
PY
nvidia-smi

chemflow_executable="$(command -v chemflow || true)"
if [[ -z "$chemflow_executable" ]]; then
    echo "chemflow was not found after activating the chemflow environment." >&2
    exit 1
fi
srun "$chemflow_executable" train hf-graphormer "$config_path"

echo "Finished: $(date --iso-8601=seconds)"
