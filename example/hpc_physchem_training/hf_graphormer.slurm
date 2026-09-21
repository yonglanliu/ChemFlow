#!/usr/bin/env bash
#SBATCH --job-name=physchem-hf-graphormer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

source "$HOME/bin/myconda"
conda activate chemflow

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
training_dir="${PHYSCHEM_JOB_DIR:-/data/liuy48/model_training/adme/physchem}"
mode="${HF_GRAPHORMER_MODE:-single}"
case "$mode" in
    single)
        default_config="$repo_dir/example/hf_graphormer_training/conf.toml"
        default_run_name="hf_graphormer_KSol_pH7_4"
        ;;
    multitask)
        default_config="$repo_dir/example/hf_graphormer_training/conf.toml"
        default_run_name="hf_graphormer_physchem_multitask"
        ;;
    *)
        echo "HF_GRAPHORMER_MODE must be 'single' or 'multitask', got: $mode" >&2
        exit 1
        ;;
esac
base_config="${HF_GRAPHORMER_BASE_CONFIG:-$default_config}"
data_file="${PHYSCHEM_DATA_FILE:-$training_dir/dataset/physchem_multitask.csv}"
test_data_file="${PHYSCHEM_TEST_DATA_FILE:-}"
checkpoint="${HF_GRAPHORMER_CHECKPOINT:-$HOME/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt}"
workdir="${HF_GRAPHORMER_WORKDIR:-$training_dir/$default_run_name}"
config_path="$training_dir/generated_configs/${default_run_name}_conf.toml"

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

mkdir -p "$(dirname -- "$config_path")" "$workdir"

export CHEMFLOW_HF_BASE_CONFIG="$base_config"
export CHEMFLOW_HF_CONFIG="$config_path"
export CHEMFLOW_HF_DATA_FILE="$data_file"
export CHEMFLOW_HF_TEST_DATA_FILE="$test_data_file"
export CHEMFLOW_HF_CHECKPOINT="$checkpoint"
export CHEMFLOW_HF_WORKDIR="$workdir"
export CHEMFLOW_HF_BATCH_SIZE="${HF_GRAPHORMER_BATCH_SIZE:-8}"
export CHEMFLOW_HF_DEVICES="${HF_GRAPHORMER_DEVICES:-1}"
if (( CHEMFLOW_HF_DEVICES < 1 )); then
    echo "HF_GRAPHORMER_DEVICES must be at least 1." >&2
    exit 1
fi
workers_per_process="$(( ${SLURM_CPUS_PER_TASK:-1} / CHEMFLOW_HF_DEVICES ))"
if (( workers_per_process < 1 )); then workers_per_process=1; fi
export CHEMFLOW_HF_NUM_WORKERS="${HF_GRAPHORMER_NUM_WORKERS:-$workers_per_process}"
export CHEMFLOW_HF_EPOCHS="${HF_GRAPHORMER_EPOCHS:-30}"
export CHEMFLOW_HF_RESUME="${HF_GRAPHORMER_RESUME:-false}"
export CHEMFLOW_HF_APPLICABILITY_ONLY="${HF_GRAPHORMER_APPLICABILITY_ONLY:-false}"
export CHEMFLOW_HF_MODE="$mode"

python - <<'PY'
import os
from pathlib import Path

import toml

base_path = Path(os.environ["CHEMFLOW_HF_BASE_CONFIG"])
output_path = Path(os.environ["CHEMFLOW_HF_CONFIG"])
config = toml.load(base_path)

config.setdefault("BaseConfig", {})["workdir"] = os.environ["CHEMFLOW_HF_WORKDIR"]

dataset = config.setdefault("DatasetConfig", {})
dataset["dataset_path"] = os.environ["CHEMFLOW_HF_DATA_FILE"]
test_data_file = os.environ["CHEMFLOW_HF_TEST_DATA_FILE"].strip()
if test_data_file:
    dataset["test_dataset_path"] = test_data_file
    dataset["test_fraction"] = 0.0
else:
    dataset.pop("test_dataset_path", None)
if os.environ["CHEMFLOW_HF_MODE"] == "single":
    dataset["target_column"] = "Log_KSOL_pH7_4_mol_L"
else:
    dataset["target_column"] = [
        "LogD_pH7_4",
        "Log_KSOL_pH6_8_mol_L",
        "Log_KSOL_pH7_4_mol_L",
        "Log_KSOL_ExpansionRX_mol_L",
    ]
dataset["smiles_column"] = "SMILES"

model = config.setdefault("ModelConfig", {})
model["checkpoint_path"] = os.environ["CHEMFLOW_HF_CHECKPOINT"]
model["local_files_only"] = True
model["freeze_encoder"] = False

training = config.setdefault("TrainingConfig", {})
training["device"] = "cuda"
devices = int(os.environ["CHEMFLOW_HF_DEVICES"])
training["devices"] = devices
training["strategy"] = "ddp" if devices > 1 else "auto"
training["batch_size"] = int(os.environ["CHEMFLOW_HF_BATCH_SIZE"])
training["num_workers"] = int(os.environ["CHEMFLOW_HF_NUM_WORKERS"])
training["num_epochs"] = int(os.environ["CHEMFLOW_HF_EPOCHS"])
training["resume"] = os.environ["CHEMFLOW_HF_RESUME"].strip().lower() in {
    "1", "true", "yes", "on"
}
training["applicability_only"] = os.environ[
    "CHEMFLOW_HF_APPLICABILITY_ONLY"
].strip().lower() in {"1", "true", "yes", "on"}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as stream:
    toml.dump(config, stream)
PY

threads_per_process="$(( ${SLURM_CPUS_PER_TASK:-1} / CHEMFLOW_HF_DEVICES ))"
if (( threads_per_process < 1 )); then threads_per_process=1; fi
export OMP_NUM_THREADS="$threads_per_process"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID:-not-set}"
echo "Environment: chemflow"
echo "Training mode: $mode"
echo "DDP processes / GPUs: $CHEMFLOW_HF_DEVICES"
echo "Dataset: $data_file"
if [[ -n "$test_data_file" ]]; then
    echo "External test dataset: $test_data_file"
fi
echo "Checkpoint: $checkpoint"
echo "Generated configuration: $config_path"
echo "Output directory: $workdir"
python --version
python - <<'PY'
import Cython
import torch
import transformers

print("Cython:", Cython.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Slurm allocated a GPU but PyTorch cannot see CUDA.")
PY
nvidia-smi

if (( CHEMFLOW_HF_DEVICES > 1 )); then
    chemflow_executable="$(command -v chemflow)"
    srun torchrun \
        --standalone \
        --nproc_per_node="$CHEMFLOW_HF_DEVICES" \
        "$chemflow_executable" train hf-graphormer "$config_path"
else
    srun chemflow train hf-graphormer "$config_path"
fi

echo "Finished: $(date --iso-8601=seconds)"
