#!/usr/bin/env bash
#SBATCH --job-name=physchem-chemeleon
# One multitask model plus four single-task models = five GPU array tasks.
# This permits four jobs to run concurrently and therefore may use four
# GPUs, 32 CPUs, and 256 GB of memory. Reduce %4 if required by the allocation.
#SBATCH --array=0-4%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

source "$HOME/bin/myconda"
conda activate chemflow

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"

# Override paths when submitting if the dataset or output directory is stored
# elsewhere on the HPC filesystem:
#   PHYSCHEM_JOB_DIR=/data/liuy48/model_training/adme/physchem \
#   PHYSCHEM_DATA_FILE=/data/liuy48/datasets/physchem_multitask.csv \
#   sbatch chemeleon_array.slurm
training_dir="${PHYSCHEM_JOB_DIR:-/data/liuy48/model_training/adme/physchem}"
base_config="${CHEMELEON_BASE_CONFIG:-$repo_dir/example/chemeleon_training/multitask_conf.toml}"
data_file="${PHYSCHEM_DATA_FILE:-$training_dir/dataset/physchem_multitask.csv}"
split_type="${CHEMELEON_SPLIT_TYPE:-scaffold_balanced}"
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
workdir="$training_dir/chemeleon_$run_name"
config_path="$generated_config_dir/chemeleon_${run_name}_conf.toml"

if [[ ! -f "$base_config" ]]; then
    echo "Base CheMeleon configuration not found: $base_config" >&2
    exit 1
fi
if [[ ! -f "$data_file" ]]; then
    echo "Training dataset not found: $data_file" >&2
    echo "Set PHYSCHEM_DATA_FILE to its location on the HPC filesystem." >&2
    exit 1
fi

case "$split_type" in
    random|random_with_repeated_smiles|scaffold_balanced|kennard_stone|kmeans) ;;
    predefined)
        if [[ -z "$split_column" ]]; then
            echo "CHEMELEON_SPLIT_TYPE=predefined requires PHYSCHEM_SPLIT_COLUMN." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported CHEMELEON_SPLIT_TYPE: $split_type" >&2
        exit 1
        ;;
esac

mkdir -p "$generated_config_dir" "$workdir"

export CHEMFLOW_CHEMELEON_BASE_CONFIG="$base_config"
export CHEMFLOW_CHEMELEON_GENERATED_CONFIG="$config_path"
export CHEMFLOW_CHEMELEON_DATA_FILE="$data_file"
export CHEMFLOW_CHEMELEON_RUN_NAME="$run_name"
export CHEMFLOW_CHEMELEON_TARGETS="$target_spec"
export CHEMFLOW_CHEMELEON_WORKDIR="$workdir"
export CHEMFLOW_CHEMELEON_SPLIT_TYPE="$split_type"
export CHEMFLOW_CHEMELEON_SPLIT_COLUMN="$split_column"
export CHEMFLOW_CHEMELEON_DEVICES="${CHEMELEON_DEVICES:-1}"
if (( CHEMFLOW_CHEMELEON_DEVICES < 1 )); then
    echo "CHEMELEON_DEVICES must be at least 1." >&2
    exit 1
fi
workers_per_process="$(( ${SLURM_CPUS_PER_TASK:-1} / CHEMFLOW_CHEMELEON_DEVICES ))"
if (( workers_per_process < 1 )); then workers_per_process=1; fi
export CHEMFLOW_CHEMELEON_NUM_WORKERS="${CHEMELEON_NUM_WORKERS:-$workers_per_process}"
export CHEMFLOW_CHEMELEON_BATCH_SIZE="${CHEMELEON_BATCH_SIZE:-64}"

python - <<'PY'
import os
from pathlib import Path

import pandas as pd
import toml

base_path = Path(os.environ["CHEMFLOW_CHEMELEON_BASE_CONFIG"])
output_path = Path(os.environ["CHEMFLOW_CHEMELEON_GENERATED_CONFIG"])
data_path = Path(os.environ["CHEMFLOW_CHEMELEON_DATA_FILE"])
run_name = os.environ["CHEMFLOW_CHEMELEON_RUN_NAME"]
workdir = Path(os.environ["CHEMFLOW_CHEMELEON_WORKDIR"])
split_column = os.environ["CHEMFLOW_CHEMELEON_SPLIT_COLUMN"].strip()
targets = [
    target.strip()
    for target in os.environ["CHEMFLOW_CHEMELEON_TARGETS"].split(",")
    if target.strip()
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

base = config.setdefault("BaseConfig", {})
base["workdir"] = str(workdir)
base["task"] = "regression"

dataset = config.setdefault("DatasetConfig", {})
dataset["dataset_path"] = str(data_path)
dataset.pop("test_dataset_path", None)
dataset["smiles_column"] = "SMILES"
dataset["target_column"] = targets if run_name == "multitask" else targets[0]
dataset["split_type"] = os.environ["CHEMFLOW_CHEMELEON_SPLIT_TYPE"]
if split_column:
    dataset["split_column"] = split_column
else:
    dataset.pop("split_column", None)

training = config.setdefault("CheMeleonTrainingConfig", {})
training["accelerator"] = "gpu"
devices = int(os.environ["CHEMFLOW_CHEMELEON_DEVICES"])
training["devices"] = devices
training["strategy"] = "ddp" if devices > 1 else "auto"
training["num_nodes"] = 1
training["num_workers"] = int(os.environ["CHEMFLOW_CHEMELEON_NUM_WORKERS"])
training["batch_size"] = int(os.environ["CHEMFLOW_CHEMELEON_BATCH_SIZE"])

pretrained_path = os.environ.get("CHEMELEON_PRETRAINED_PATH", "").strip()
model = config.setdefault("CheMeleonConfig", {})
if pretrained_path:
    model["pretrained_path"] = pretrained_path
else:
    model.pop("pretrained_path", None)

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as stream:
    toml.dump(config, stream)
PY

threads_per_process="$(( ${SLURM_CPUS_PER_TASK:-1} / CHEMFLOW_CHEMELEON_DEVICES ))"
if (( threads_per_process < 1 )); then threads_per_process=1; fi
export OMP_NUM_THREADS="$threads_per_process"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID}; array task: ${SLURM_ARRAY_TASK_ID}"
echo "Run: $run_name"
echo "Lightning devices / GPUs: $CHEMFLOW_CHEMELEON_DEVICES"
echo "Targets: $target_spec"
echo "Dataset: $data_file"
if [[ -n "$split_column" ]]; then
    echo "Split column: $split_column"
else
    echo "Split type: $split_type"
fi
echo "Base configuration: $base_config"
echo "Generated configuration: $config_path"
echo "Output directory: $workdir"
nvidia-smi || true

srun chemflow train chemeleon "$config_path"

echo "Finished: $(date --iso-8601=seconds)"
