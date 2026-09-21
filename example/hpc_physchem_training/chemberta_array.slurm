#!/usr/bin/env bash
#SBATCH --job-name=physchem-chemberta
#SBATCH --partition=gpu
# One multitask model plus three single-task models.
# Up to four tasks may run concurrently, using four GPUs in total.
#SBATCH --array=0-3%4
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

# Override these values at submission time when needed:
#   PHYSCHEM_JOB_DIR=/data/liuy48/model_training/adme/adsorption/physchem \
#   PHYSCHEM_DATA_FILE=/data/liuy48/model_training/adme/adsorption/dataset/physchem_multitask.csv \
#   CHEMBERTA_MODEL_NAME=/data/liuy48/models/ChemBERTa-77M-MLM \
#   sbatch chemberta_array.slurm
training_dir="${PHYSCHEM_JOB_DIR:-/data/liuy48/model_training/adme/adsorption/physchem}"
base_config="${CHEMBERTA_BASE_CONFIG:-$repo_dir/example/chemberta_training/conf.toml}"
data_file="${PHYSCHEM_DATA_FILE:-/data/liuy48/model_training/adme/adsorption/dataset/physchem_multitask.csv}"
model_name="${CHEMBERTA_MODEL_NAME:-DeepChem/ChemBERTa-77M-MLM}"
split_type="${CHEMBERTA_SPLIT_TYPE:-scaffold_balanced}"
split_column="${PHYSCHEM_SPLIT_COLUMN:-}"
generated_config_dir="$training_dir/generated_configs"

run_names=(
    "multitask"
    "LogD_pH7_4"
    "KSol_pH6_8"
    "KSol_pH7_4"
)
target_specs=(
    "LogD_pH7_4,Log_KSOL_ExpansionRX_mol_L"
    "LogD_pH7_4"
    "Log_KSOL_pH6_8_mol_L"
    "Log_KSOL_pH7_4_mol_L"
)

array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
task_count="${#run_names[@]}"
if (( array_index < 0 || array_index >= task_count )); then
    echo "Invalid array index: $array_index; expected 0-$((task_count - 1))." >&2
    exit 1
fi

run_name="${run_names[$array_index]}"
target_spec="${target_specs[$array_index]}"
workdir="$training_dir/chemberta_$run_name"
config_path="$generated_config_dir/chemberta_${run_name}_conf.toml"

for required_file in "$base_config" "$data_file"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done
if [[ "$model_name" == /* && ! -d "$model_name" ]]; then
    echo "Local ChemBERTa model directory not found: $model_name" >&2
    exit 1
fi

case "$split_type" in
    random|random_with_repeated_smiles|scaffold_balanced|kennard_stone|kmeans) ;;
    predefined)
        split_column="${split_column:-split}"
        ;;
    *)
        echo "Unsupported CHEMBERTA_SPLIT_TYPE: $split_type" >&2
        exit 1
        ;;
esac

mkdir -p "$generated_config_dir" "$workdir"

export CHEMFLOW_CHEMBERTA_BASE_CONFIG="$base_config"
export CHEMFLOW_CHEMBERTA_GENERATED_CONFIG="$config_path"
export CHEMFLOW_CHEMBERTA_DATA_FILE="$data_file"
export CHEMFLOW_CHEMBERTA_MODEL_NAME="$model_name"
export CHEMFLOW_CHEMBERTA_RUN_NAME="$run_name"
export CHEMFLOW_CHEMBERTA_TARGETS="$target_spec"
export CHEMFLOW_CHEMBERTA_WORKDIR="$workdir"
export CHEMFLOW_CHEMBERTA_SPLIT_TYPE="$split_type"
export CHEMFLOW_CHEMBERTA_SPLIT_COLUMN="$split_column"
export CHEMFLOW_CHEMBERTA_BATCH_SIZE="${CHEMBERTA_BATCH_SIZE:-32}"
export CHEMFLOW_CHEMBERTA_NUM_WORKERS="${CHEMBERTA_NUM_WORKERS:-8}"
export CHEMFLOW_CHEMBERTA_EPOCHS="${CHEMBERTA_EPOCHS:-20}"
export CHEMFLOW_CHEMBERTA_RESUME="${CHEMBERTA_RESUME:-false}"
export CHEMFLOW_CHEMBERTA_APPLICABILITY_ONLY="${CHEMBERTA_APPLICABILITY_ONLY:-false}"

python - <<'PY'
import os
from pathlib import Path

import pandas as pd
import toml

base_path = Path(os.environ["CHEMFLOW_CHEMBERTA_BASE_CONFIG"])
output_path = Path(os.environ["CHEMFLOW_CHEMBERTA_GENERATED_CONFIG"])
data_path = Path(os.environ["CHEMFLOW_CHEMBERTA_DATA_FILE"])
run_name = os.environ["CHEMFLOW_CHEMBERTA_RUN_NAME"]
workdir = Path(os.environ["CHEMFLOW_CHEMBERTA_WORKDIR"])
split_column = os.environ["CHEMFLOW_CHEMBERTA_SPLIT_COLUMN"].strip()
targets = [
    value.strip()
    for value in os.environ["CHEMFLOW_CHEMBERTA_TARGETS"].split(",")
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

base = config.setdefault("BaseConfig", {})
base["workdir"] = str(workdir)
base["task"] = "regression"

dataset = config.setdefault("DatasetConfig", {})
dataset["dataset_path"] = str(data_path)
dataset.pop("test_dataset_path", None)
dataset["smiles_column"] = "SMILES"
dataset["target_column"] = targets if run_name == "multitask" else targets[0]
if split_column:
    dataset["split_column"] = split_column
    dataset["split_type"] = "predefined"
else:
    dataset.pop("split_column", None)
    dataset["split_type"] = os.environ["CHEMFLOW_CHEMBERTA_SPLIT_TYPE"]

model = config.setdefault("ModelConfig", {})
model["model_name"] = os.environ["CHEMFLOW_CHEMBERTA_MODEL_NAME"]
model["architecture"] = "roberta"
model["local_files_only"] = True
model["freeze_encoder"] = False

training = config.setdefault("TrainingConfig", {})
training["device"] = "cuda"
training["devices"] = 1
training["strategy"] = "auto"
training["batch_size"] = int(os.environ["CHEMFLOW_CHEMBERTA_BATCH_SIZE"])
training["num_workers"] = int(os.environ["CHEMFLOW_CHEMBERTA_NUM_WORKERS"])
training["num_epochs"] = int(os.environ["CHEMFLOW_CHEMBERTA_EPOCHS"])
training["resume"] = os.environ["CHEMFLOW_CHEMBERTA_RESUME"].lower() in {
    "1", "true", "yes", "on"
}
training["applicability_only"] = os.environ[
    "CHEMFLOW_CHEMBERTA_APPLICABILITY_ONLY"
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
# The model must already be in the shared Hugging Face cache or supplied as a
# local directory. These settings prevent compute-node network retries.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID}; array task: ${SLURM_ARRAY_TASK_ID}"
echo "Run: $run_name"
echo "Targets: $target_spec"
echo "Dataset: $data_file"
echo "ChemBERTa model: $model_name"
if [[ -n "$split_column" ]]; then
    echo "Split column: $split_column"
else
    echo "Split type: $split_type"
fi
echo "Base configuration: $base_config"
echo "Generated configuration: $config_path"
echo "Output directory: $workdir"

python - <<'PY'
import os

import torch
from transformers import RobertaConfig, RobertaTokenizerFast

model_name = os.environ["CHEMFLOW_CHEMBERTA_MODEL_NAME"]
RobertaConfig.from_pretrained(model_name, local_files_only=True)
RobertaTokenizerFast.from_pretrained(model_name, local_files_only=True)
print("PyTorch:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("ChemBERTa cache/model check: OK", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("Slurm allocated a GPU but PyTorch cannot see CUDA.")
PY
nvidia-smi

srun chemflow train chemberta "$config_path"

echo "Finished: $(date --iso-8601=seconds)"
