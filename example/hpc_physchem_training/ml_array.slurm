#!/usr/bin/env bash
#SBATCH --job-name=physchem-ml
# Four endpoints x four molecular-representation sets = 16 array tasks.
# Run no more than three tasks concurrently. Change %3 only if the HPC
# allocation can support more than 24 CPUs and 96 GB of memory at one time.
#SBATCH --array=0-15%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

source "$HOME/bin/myconda"
conda activate chemflow

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"

# These paths can be overridden when submitting the job, for example:
#   PHYSCHEM_JOB_DIR=/data/liuy48/model_training/adme/physchem \
#   PHYSCHEM_DATA_FILE=/data/liuy48/datasets/physchem_multitask.csv \
#   sbatch ml_array.slurm
training_dir="${PHYSCHEM_JOB_DIR:-/data/liuy48/model_training/adme/physchem}"
base_config="${ML_BASE_CONFIG:-$repo_dir/example/ml_training/conf.yaml}"
data_file="${PHYSCHEM_DATA_FILE:-$training_dir/dataset/physchem_multitask.csv}"
data_split_method="${PHYSCHEM_SPLIT_METHOD:-scaffold}"
split_column="${PHYSCHEM_SPLIT_COLUMN:-split}"
generated_config_dir="$training_dir/generated_configs"

endpoints=(
    "LogD_pH7_4"
    "KSol_pH6_8"
    "KSol_pH7_4"
    "KSol_ExpansionRX"
)
target_columns=(
    "LogD_pH7_4"
    "Log_KSOL_pH6_8_mol_L"
    "Log_KSOL_pH7_4_mol_L"
    "Log_KSOL_ExpansionRX_mol_L"
)
mol_rep_names=(
    "ecfp4"
    "ecfp4_rdkit2d"
    "rdkit2d"
    "ecfp4_rdkit2d_erg_avalon"
)
mol_rep_features=(
    "ecfp4"
    "ecfp4,rdkit2d"
    "rdkit2d"
    "ecfp4,rdkit2d,erg,avalon"
)

array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
representation_count="${#mol_rep_names[@]}"
task_count="$(( ${#endpoints[@]} * representation_count ))"

if (( array_index < 0 || array_index >= task_count )); then
    echo "Invalid array index: $array_index; expected 0-$((task_count - 1))." >&2
    exit 1
fi

endpoint_index="$(( array_index / representation_count ))"
representation_index="$(( array_index % representation_count ))"
endpoint="${endpoints[$endpoint_index]}"
target_column="${target_columns[$endpoint_index]}"
mol_rep_name="${mol_rep_names[$representation_index]}"
mol_rep_csv="${mol_rep_features[$representation_index]}"

endpoint_workdir="$training_dir/$endpoint/$mol_rep_name"
config_path="$generated_config_dir/ml_${endpoint}_${mol_rep_name}_conf.yaml"

if [[ ! -f "$base_config" ]]; then
    echo "Base configuration file not found: $base_config" >&2
    exit 1
fi
if [[ ! -f "$data_file" ]]; then
    echo "Training dataset not found: $data_file" >&2
    echo "Set PHYSCHEM_DATA_FILE to its location on the HPC filesystem." >&2
    exit 1
fi

case "$data_split_method" in
    random|scaffold|predefined|cluster|butina) ;;
    *)
        echo "Unsupported PHYSCHEM_SPLIT_METHOD: $data_split_method" >&2
        exit 1
        ;;
esac

mkdir -p "$generated_config_dir" "$endpoint_workdir"

export CHEMFLOW_ML_BASE_CONFIG="$base_config"
export CHEMFLOW_ML_GENERATED_CONFIG="$config_path"
export CHEMFLOW_ML_DATA_FILE="$data_file"
export CHEMFLOW_ML_ENDPOINT="$endpoint"
export CHEMFLOW_ML_TARGET_COLUMN="$target_column"
export CHEMFLOW_ML_WORKDIR="$endpoint_workdir"
export CHEMFLOW_ML_REPRESENTATION_NAME="$mol_rep_name"
export CHEMFLOW_ML_FEATURES="$mol_rep_csv"
export CHEMFLOW_ML_SPLIT_METHOD="$data_split_method"
export CHEMFLOW_ML_SPLIT_COLUMN="$split_column"
export CHEMFLOW_ML_N_JOBS="${SLURM_CPUS_PER_TASK:-1}"

python - <<'PY'
import os
from pathlib import Path

import pandas as pd
import yaml

base_path = Path(os.environ["CHEMFLOW_ML_BASE_CONFIG"])
output_path = Path(os.environ["CHEMFLOW_ML_GENERATED_CONFIG"])
data_path = Path(os.environ["CHEMFLOW_ML_DATA_FILE"])
endpoint = os.environ["CHEMFLOW_ML_ENDPOINT"]
target_column = os.environ["CHEMFLOW_ML_TARGET_COLUMN"]
workdir = Path(os.environ["CHEMFLOW_ML_WORKDIR"])
split_method = os.environ["CHEMFLOW_ML_SPLIT_METHOD"]
split_column = os.environ["CHEMFLOW_ML_SPLIT_COLUMN"]

with base_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

if not isinstance(config, dict):
    raise ValueError(f"Base configuration must be a YAML mapping: {base_path}")

available_columns = set(pd.read_csv(data_path, nrows=0).columns)
required_columns = {"SMILES", target_column}
if split_method == "predefined":
    required_columns.add(split_column)
missing_columns = sorted(required_columns - available_columns)
if missing_columns:
    raise ValueError(
        f"Dataset {data_path} is missing required columns: {missing_columns}. "
        f"Available columns: {sorted(available_columns)}"
    )

data = config.setdefault("data", {})
data["data_file"] = str(data_path)
data["X_col"] = "SMILES"
data["y_col"] = target_column

split = config.setdefault("data_split", {})
split["split_method"] = split_method
if split_method == "predefined":
    split["split_column"] = split_column
else:
    split.pop("split_column", None)
split["save_split_data"] = True
split["save_dir"] = str(workdir / "split_data")
split["split_name"] = (
    f"single_task_{endpoint}_{os.environ['CHEMFLOW_ML_REPRESENTATION_NAME']}_"
    f"{split_method}_split"
)

featurization = config.setdefault("featurization", {})
featurization["features"] = [
    feature.strip()
    for feature in os.environ["CHEMFLOW_ML_FEATURES"].split(",")
    if feature.strip()
]

config["workdir"] = str(workdir)
config["task_type"] = "regression"
config["cv_strategy"] = "scaffold-grouped"
config["n_jobs"] = int(os.environ["CHEMFLOW_ML_N_JOBS"])

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY

# Prevent each estimator from using more threads than Slurm assigned.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID}; array task: ${SLURM_ARRAY_TASK_ID}"
echo "Endpoint: $endpoint ($target_column)"
echo "Molecular representations: $mol_rep_csv"
echo "Dataset: $data_file"
echo "Outer split method: $data_split_method"
echo "Inner CV strategy: scaffold-grouped"
echo "Base configuration: $base_config"
echo "Generated configuration: $config_path"
echo "Output directory: $endpoint_workdir"

srun chemflow train ml "$config_path"

echo "Finished: $(date --iso-8601=seconds)"
