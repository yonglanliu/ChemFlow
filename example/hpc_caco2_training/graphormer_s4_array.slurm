#!/usr/bin/env bash
#SBATCH --job-name=caco2-graphormer-s4
#SBATCH --partition=gpu
# S4 single-task, ExpansionRX multitask baseline, and S4 multitask.
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

source "/home/liuy48/bin/myconda"
conda activate chemflow

training_dir="/data/liuy48/model_training/adme/adsorption/caco2"
config_dir="/data/liuy48/model_training/adme/adsorption/caco2/conf"
generated_config_dir="/data/liuy48/model_training/adme/adsorption/caco2/generated_configs"
data_file="/data/liuy48/model_training/adme/adsorption/caco2/dataset/caco2_expansionrx_train.csv"
test_data_file="/data/liuy48/model_training/adme/adsorption/caco2/dataset/caco2_expansionrx_test.csv"
base_config="/data/liuy48/model_training/adme/adsorption/caco2/conf/graphormer_conf.toml"
official_checkpoint="/data/liuy48/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt"
public3_best_model="/data/liuy48/model_training/adme/adsorption/caco2/graphormer_S1_public3_scaffold_val_0.2_seed42/best_model"

run_names=(
    "S4_transfer_single_scaffold_val_0.2_seed42"
    "S2_expansionrx_multitask_baseline_scaffold_val_0.2_seed42"
    "S4_transfer_multitask_scaffold_val_0.2_seed42"
)
target_specs=(
    "Log10_Caco_Papp_AB_cm_s"
    "Log10_Caco_Papp_AB_cm_s,Log10_Caco_ER"
    "Log10_Caco_Papp_AB_cm_s,Log10_Caco_ER"
)
transfer_modes=("full" "none" "encoder")

array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( array_index < 0 || array_index >= ${#run_names[@]} )); then
    echo "Invalid array index: $array_index; expected 0-2." >&2
    exit 1
fi

run_name="${run_names[$array_index]}"
target_spec="${target_specs[$array_index]}"
transfer_mode="${transfer_modes[$array_index]}"
workdir="$training_dir/graphormer_$run_name"
config_path="$generated_config_dir/graphormer_${run_name}_conf.toml"

for required_file in "$base_config" "$data_file" "$test_data_file"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done
if [[ "$transfer_mode" == "none" ]]; then
    if [[ ! -f "$official_checkpoint" ]]; then
        echo "Official Graphormer checkpoint not found: $official_checkpoint" >&2
        exit 1
    fi
elif [[ ! -f "$public3_best_model/config.json" ]]; then
    echo "Public3 best_model directory is incomplete: $public3_best_model" >&2
    exit 1
fi

mkdir -p "$generated_config_dir" "$workdir"
exec 9>"$workdir/.training.lock"
if ! flock -n 9; then
    echo "Another job is already using: $workdir" >&2
    exit 1
fi

export CF_S4_BASE_CONFIG="$base_config"
export CF_S4_CONFIG_PATH="$config_path"
export CF_S4_DATA_FILE="$data_file"
export CF_S4_TEST_FILE="$test_data_file"
export CF_S4_TARGETS="$target_spec"
export CF_S4_WORKDIR="$workdir"
export CF_S4_OFFICIAL_CHECKPOINT="$official_checkpoint"
export CF_S4_TRANSFER_CHECKPOINT="$public3_best_model"
export CF_S4_TRANSFER_MODE="$transfer_mode"

python - <<'PY'
import os
from pathlib import Path

import pandas as pd
import toml

base_path = Path(os.environ["CF_S4_BASE_CONFIG"])
output_path = Path(os.environ["CF_S4_CONFIG_PATH"])
data_path = Path(os.environ["CF_S4_DATA_FILE"])
test_path = Path(os.environ["CF_S4_TEST_FILE"])
targets = [
    value.strip()
    for value in os.environ["CF_S4_TARGETS"].split(",")
    if value.strip()
]
transfer_mode = os.environ["CF_S4_TRANSFER_MODE"]

for label, path in (("training", data_path), ("test", test_path)):
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted({"SMILES", *targets} - columns)
    if missing:
        raise ValueError(f"{label} dataset {path} is missing columns: {missing}")

config = toml.load(base_path)
config.setdefault("BaseConfig", {}).update(
    {
        "workdir": os.environ["CF_S4_WORKDIR"],
        "task": "regression",
    }
)
dataset = config.setdefault("DatasetConfig", {})
dataset.update(
    {
        "dataset_path": str(data_path),
        "test_dataset_path": str(test_path),
        "smiles_column": "SMILES",
        "target_column": targets[0] if len(targets) == 1 else targets,
        "split_type": "scaffold_balanced",
        "val_fraction": 0.2,
        "test_fraction": 0.0,
    }
)
dataset.pop("split_column", None)

model = config.setdefault("ModelConfig", {})
model["freeze_encoder"] = False
model["local_files_only"] = True
if transfer_mode == "none":
    model["checkpoint_path"] = os.environ["CF_S4_OFFICIAL_CHECKPOINT"]
    model.pop("transfer_checkpoint", None)
    model.pop("transfer_encoder_only", None)
else:
    model["transfer_checkpoint"] = os.environ["CF_S4_TRANSFER_CHECKPOINT"]
    model["transfer_encoder_only"] = transfer_mode == "encoder"
    model.pop("checkpoint_path", None)

training = config.setdefault("TrainingConfig", {})
training.update(
    {
        "seed": 42,
        "device": "cuda",
        "devices": 1,
        "strategy": "auto",
        "batch_size": 8,
        "num_workers": 4,
        "num_epochs": 30,
        "resume": False,
        "applicability_only": False,
        "early_stopping_patience": 5,
    }
)
if transfer_mode == "none":
    training["encoder_learning_rate"] = 1e-5
    training["head_learning_rate"] = 1e-4
else:
    training["encoder_learning_rate"] = 1e-6
    training["head_learning_rate"] = 1e-5

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as stream:
    toml.dump(config, stream)
print(f"Generated configuration: {output_path}", flush=True)
PY

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Run: $run_name"
echo "Training dataset: $data_file"
echo "Independent test dataset: $test_data_file"
echo "Targets: $target_spec"
echo "Transfer mode: $transfer_mode"
if [[ "$transfer_mode" == "none" ]]; then
    echo "Initial weights: $official_checkpoint"
else
    echo "Initial weights: $public3_best_model"
fi
echo "Split: scaffold_balanced; validation fraction: 0.2"
echo "Generated configuration: $config_path"
echo "Output directory: $workdir"
nvidia-smi

chemflow_executable="$(command -v chemflow || true)"
if [[ -z "$chemflow_executable" ]]; then
    echo "chemflow was not found after environment activation." >&2
    exit 1
fi
srun "$chemflow_executable" train hf-graphormer "$config_path"
