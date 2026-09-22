#!/usr/bin/env bash
#SBATCH --job-name=caco2-graphormer-transfer
#SBATCH --partition=gpu
# Direct ExpansionRX versus Public3-to-ExpansionRX transfer at val=0.1/0.2.
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail

source "/home/liuy48/bin/myconda"
conda activate chemflow

training_dir="/data/liuy48/model_training/adme/adsorption/caco2"
generated_config_dir="$training_dir/generated_configs"
base_config="$training_dir/conf/graphormer_conf.toml"
data_file="$training_dir/dataset/caco2_expansionrx_train.csv"
test_data_file="$training_dir/dataset/caco2_expansionrx_test.csv"
official_checkpoint="/data/liuy48/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt"
public3_best_model="$training_dir/graphormer_S1_public3_scaffold_val_0.2_seed42/best_model"
target="Log10_Caco_Papp_AB_cm_s"

# Array mapping:
#   0 = direct,   val=0.1
#   1 = direct,   val=0.2
#   2 = transfer, val=0.1
#   3 = transfer, val=0.2
strategies=("direct" "direct" "transfer" "transfer")
val_fractions=("0.1" "0.2" "0.1" "0.2")

array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( array_index < 0 || array_index >= ${#strategies[@]} )); then
    echo "Invalid array index: $array_index; expected 0-3." >&2
    exit 1
fi

strategy="${strategies[$array_index]}"
val_fraction="${val_fractions[$array_index]}"
seed=42
run_name="comparison_${strategy}_single_scaffold_val_${val_fraction}_seed${seed}"
workdir="$training_dir/graphormer_$run_name"
config_path="$generated_config_dir/graphormer_${run_name}_conf.toml"

for required_file in "$base_config" "$data_file" "$test_data_file"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done
if [[ "$strategy" == "direct" ]]; then
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

export CF_COMPARE_BASE_CONFIG="$base_config"
export CF_COMPARE_CONFIG_PATH="$config_path"
export CF_COMPARE_DATA_FILE="$data_file"
export CF_COMPARE_TEST_FILE="$test_data_file"
export CF_COMPARE_TARGET="$target"
export CF_COMPARE_WORKDIR="$workdir"
export CF_COMPARE_STRATEGY="$strategy"
export CF_COMPARE_VAL_FRACTION="$val_fraction"
export CF_COMPARE_SEED="$seed"
export CF_COMPARE_OFFICIAL_CHECKPOINT="$official_checkpoint"
export CF_COMPARE_TRANSFER_CHECKPOINT="$public3_best_model"

python - <<'PY'
import os
from pathlib import Path

import pandas as pd
import toml

base_path = Path(os.environ["CF_COMPARE_BASE_CONFIG"])
output_path = Path(os.environ["CF_COMPARE_CONFIG_PATH"])
data_path = Path(os.environ["CF_COMPARE_DATA_FILE"])
test_path = Path(os.environ["CF_COMPARE_TEST_FILE"])
target = os.environ["CF_COMPARE_TARGET"]
strategy = os.environ["CF_COMPARE_STRATEGY"]
val_fraction = float(os.environ["CF_COMPARE_VAL_FRACTION"])

for label, path in (("training", data_path), ("test", test_path)):
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted({"SMILES", target} - columns)
    if missing:
        raise ValueError(f"{label} dataset {path} is missing columns: {missing}")

config = toml.load(base_path)
config.setdefault("BaseConfig", {}).update(
    {
        "workdir": os.environ["CF_COMPARE_WORKDIR"],
        "task": "regression",
    }
)
dataset = config.setdefault("DatasetConfig", {})
dataset.update(
    {
        "dataset_path": str(data_path),
        "test_dataset_path": str(test_path),
        "smiles_column": "SMILES",
        "target_column": target,
        "split_type": "scaffold_balanced",
        "val_fraction": val_fraction,
        "test_fraction": 0.0,
    }
)
dataset.pop("split_column", None)

model = config.setdefault("ModelConfig", {})
model["freeze_encoder"] = False
model["local_files_only"] = True
if strategy == "direct":
    model["checkpoint_path"] = os.environ["CF_COMPARE_OFFICIAL_CHECKPOINT"]
    model.pop("transfer_checkpoint", None)
    model.pop("transfer_encoder_only", None)
else:
    model["transfer_checkpoint"] = os.environ[
        "CF_COMPARE_TRANSFER_CHECKPOINT"
    ]
    model["transfer_encoder_only"] = False
    model.pop("checkpoint_path", None)

training = config.setdefault("TrainingConfig", {})
training.update(
    {
        "seed": int(os.environ["CF_COMPARE_SEED"]),
        "device": "cuda",
        "devices": 1,
        "strategy": "auto",
        "batch_size": 8,
        "num_workers": 4,
        "num_epochs": 100,
        "early_stopping_patience": 10,
        "resume": False,
        "applicability_only": False,
    }
)
if strategy == "direct":
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
echo "Strategy: $strategy"
echo "Training dataset: $data_file"
echo "Independent test dataset: $test_data_file"
echo "Target: $target"
echo "Split: scaffold_balanced"
echo "Validation fraction: $val_fraction"
if [[ "$strategy" == "direct" ]]; then
    echo "Initial weights: $official_checkpoint"
else
    echo "Initial weights: $public3_best_model"
fi
echo "Generated configuration: $config_path"
echo "Output directory: $workdir"
nvidia-smi

chemflow_executable="$(command -v chemflow || true)"
if [[ -z "$chemflow_executable" ]]; then
    echo "chemflow was not found after environment activation." >&2
    exit 1
fi
srun "$chemflow_executable" train hf-graphormer "$config_path"
