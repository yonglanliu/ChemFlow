#!/usr/bin/env bash
# Shared implementation for the three Caco-2 model arrays.
# This file is sourced by the model-specific Slurm submission scripts.

set -euo pipefail
source "/home/liuy48/bin/myconda"
conda activate chemflow

training_dir="/data/liuy48/model_training/adme/adsorption/caco2"
dataset_dir="/data/liuy48/model_training/adme/adsorption/caco2/dataset"
config_dir="/data/liuy48/model_training/adme/adsorption/caco2/conf"
generated_config_dir="/data/liuy48/model_training/adme/adsorption/caco2/generated_configs"
test_data_file="/data/liuy48/model_training/adme/adsorption/caco2/dataset/caco2_expansionrx_test.csv"
target_spec="Log10_Caco_Papp_AB_cm_s"

strategies=("S1_public3" "S2_expansionrx" "S3_joint")
data_files=(
    "/data/liuy48/model_training/adme/adsorption/caco2/dataset/caco2_public_3source_training.csv"
    "/data/liuy48/model_training/adme/adsorption/caco2/dataset/caco2_expansionrx_train.csv"
    "/data/liuy48/model_training/adme/adsorption/caco2/dataset/caco2_public3_expansionrx_train.csv"
)
split_types=("scaffold_balanced" "scaffold_balanced" "random" "random")
split_labels=("scaffold" "scaffold" "random" "random")
val_fractions=("0.1" "0.2" "0.1" "0.2")

array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
condition_count=4
task_count=12
if (( array_index < 0 || array_index >= task_count )); then
    echo "Invalid array index $array_index; expected 0-11." >&2
    exit 1
fi

strategy_index=$((array_index / condition_count))
condition_index=$((array_index % condition_count))
strategy="${strategies[$strategy_index]}"
data_file="${data_files[$strategy_index]}"
split_type="${split_types[$condition_index]}"
split_label="${split_labels[$condition_index]}"
val_fraction="${val_fractions[$condition_index]}"
seed=42
checkpoint=""
chemberta_model=""

case "$model_kind" in
    chemeleon)
        base_config="$config_dir/chemeleon_conf.toml"
        batch_size=64
        epochs=30
        freeze_epochs=5
        run_name="${strategy}_freeze${freeze_epochs}_${split_label}_val_${val_fraction}_seed${seed}"
        command_name="chemeleon"
        ;;
    graphormer)
        base_config="$config_dir/graphormer_conf.toml"
        batch_size=8
        epochs=30
        freeze_epochs=0
        checkpoint="/data/liuy48/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt"
        run_name="${strategy}_${split_label}_val_${val_fraction}_seed${seed}"
        command_name="hf-graphormer"
        ;;
    chemberta)
        base_config="$config_dir/chemberta_conf.toml"
        batch_size=32
        epochs=30
        freeze_epochs=0
        chemberta_model="DeepChem/ChemBERTa-77M-MLM"
        run_name="${strategy}_${split_label}_val_${val_fraction}_seed${seed}"
        command_name="chemberta"
        ;;
    *)
        echo "Unsupported model kind: $model_kind" >&2
        exit 1
        ;;
esac

workdir="$training_dir/${model_kind}_$run_name"
config_path="$generated_config_dir/${model_kind}_${run_name}_conf.toml"
for required_file in "$base_config" "$data_file" "$test_data_file"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done
if [[ "$model_kind" == "graphormer" && ! -f "$checkpoint" ]]; then
    echo "Graphormer checkpoint not found: $checkpoint" >&2
    exit 1
fi
mkdir -p "$generated_config_dir" "$workdir"

# Do not allow two submissions to write the same checkpoints and histories.
exec 9>"$workdir/.training.lock"
if ! flock -n 9; then
    echo "Another job is already using this output directory: $workdir" >&2
    exit 1
fi

export CF_MODEL_KIND="$model_kind"
export CF_BASE_CONFIG="$base_config"
export CF_CONFIG_PATH="$config_path"
export CF_DATA_FILE="$data_file"
export CF_TEST_DATA_FILE="$test_data_file"
export CF_TARGET="$target_spec"
export CF_WORKDIR="$workdir"
export CF_SPLIT_TYPE="$split_type"
export CF_VAL_FRACTION="$val_fraction"
export CF_SEED="$seed"
export CF_BATCH_SIZE="$batch_size"
export CF_EPOCHS="$epochs"
export CF_FREEZE_EPOCHS="$freeze_epochs"
export CF_CHECKPOINT="$checkpoint"
export CF_CHEMBERTA_MODEL="$chemberta_model"

python - <<'PY'
import os
from pathlib import Path
import pandas as pd
import toml

kind = os.environ["CF_MODEL_KIND"]
base_path = Path(os.environ["CF_BASE_CONFIG"])
output_path = Path(os.environ["CF_CONFIG_PATH"])
data_path = Path(os.environ["CF_DATA_FILE"])
test_path = Path(os.environ["CF_TEST_DATA_FILE"])
target = os.environ["CF_TARGET"]
val_fraction = float(os.environ["CF_VAL_FRACTION"])
if not 0.0 < val_fraction < 1.0:
    raise ValueError("Validation fraction must be between 0 and 1.")
for label, path in (("training", data_path), ("test", test_path)):
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted({"SMILES", target} - columns)
    if missing:
        raise ValueError(f"{label} dataset {path} is missing columns: {missing}")

config = toml.load(base_path)
base = config.setdefault("BaseConfig", {})
base.update({"workdir": os.environ["CF_WORKDIR"], "task": "regression", "seed": int(os.environ["CF_SEED"])})
dataset = config.setdefault("DatasetConfig", {})
dataset.update({
    "dataset_path": str(data_path),
    "test_dataset_path": str(test_path),
    "smiles_column": "SMILES",
    "target_column": target,
    "split_type": os.environ["CF_SPLIT_TYPE"],
    "val_fraction": val_fraction,
    "test_fraction": 0.0,
})
dataset.pop("split_column", None)

if kind == "chemeleon":
    training = config.setdefault("CheMeleonTrainingConfig", {})
    training.update({"accelerator": "gpu", "devices": 1, "strategy": "auto", "num_nodes": 1, "num_workers": 4, "batch_size": int(os.environ["CF_BATCH_SIZE"]), "num_epochs": int(os.environ["CF_EPOCHS"]), "resume": False})
    model = config.setdefault("CheMeleonConfig", {})
    model["freeze_epochs"] = int(os.environ["CF_FREEZE_EPOCHS"])
    model.pop("transfer_checkpoint", None)
else:
    training = config.setdefault("TrainingConfig", {})
    training.update({"device": "cuda", "devices": 1, "strategy": "auto", "num_workers": 4, "batch_size": int(os.environ["CF_BATCH_SIZE"]), "num_epochs": int(os.environ["CF_EPOCHS"]), "resume": False, "applicability_only": False})
    model = config.setdefault("ModelConfig", {})
    model["freeze_encoder"] = False
    if kind == "graphormer":
        model.update({"checkpoint_path": os.environ["CF_CHECKPOINT"], "local_files_only": True})
    else:
        model.update({"model_name": os.environ["CF_CHEMBERTA_MODEL"], "architecture": "roberta", "local_files_only": True})

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as stream:
    toml.dump(config, stream)
print(f"Generated configuration: {output_path}", flush=True)
PY

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [[ "$model_kind" == "chemberta" ]]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

echo "Model: $model_kind"
echo "Run: $run_name"
echo "Strategy: $strategy"
echo "Training dataset: $data_file"
echo "Independent test dataset: $test_data_file"
echo "Split: $split_type; validation fraction: $val_fraction"
echo "Generated configuration: $config_path"
echo "Output directory: $workdir"
nvidia-smi

chemflow_executable="$(command -v chemflow || true)"
if [[ -z "$chemflow_executable" ]]; then
    echo "chemflow was not found after environment activation." >&2
    exit 1
fi
srun "$chemflow_executable" train "$command_name" "$config_path"
