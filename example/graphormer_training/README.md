# Graphormer training and prediction tutorial

This directory contains example configurations for:

- single-task regression;
- binary or multiclass classification;
- multitask regression;
- prediction from a trained checkpoint.

Run the commands below from the ChemFlow repository root.

## 1. Activate ChemFlow

```bash
conda activate chemflow
cd /path/to/ChemFlow
```

Confirm that the command-line interface is available:

```bash
chemflow --help
```

## 2. Prepare the dataset

Training data must be a CSV or Parquet file with a SMILES column and one or
more target columns.

Single-task regression example:

```csv
SMILES,LogS
CCO,-0.31
CCN,-0.52
c1ccccc1,-2.13
```

Binary classification example:

```csv
SMILES,class_id
CCO,0
CCN,1
c1ccccc1,1
```

Multitask regression may contain missing values. Each target becomes one task:

```csv
SMILES,LogD,LogS,HLM
CCO,0.2,-0.31,1.4
CCN,0.5,,1.1
c1ccccc1,2.1,-2.13,
```

Invalid SMILES and rows without any usable target are written to
`rejected_rows.csv` in the training work directory.

## 3. Configure data splitting

The example configs support these generated split methods:

- `random`
- `random_with_repeated_smiles`
- `scaffold_balanced`
- `kennard_stone`
- `kmeans`

For a generated split, configure:

```toml
[DatasetConfig]
dataset_path = "./dataset/my_data.csv"
smiles_column = "SMILES"
target_column = "LogS"
split_type = "scaffold_balanced"
val_fraction = 0.1
test_fraction = 0.1
```

To use split labels already present in the dataset, set:

```toml
split_column = "split"
```

The column values should be `train`, `val`, or `test`.

To evaluate against a completely independent test file, set
`test_fraction = 0.0` and provide:

```toml
test_dataset_path = "./dataset/my_external_test.csv"
```

Do not combine an external test dataset with test rows in the primary dataset.

## 4. Pretrained weights

The examples use pretrained PCQM4Mv1 Graphormer weights:

```toml
[GraphormerConfig]
use_pretrained = true
```

The checkpoint is downloaded once and cached at:

```text
~/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt
```

To use an existing checkpoint file:

```toml
use_pretrained = true
pretrained_path = "/path/to/graphormer-checkpoint.pt"
```

To initialize Graphormer randomly:

```toml
use_pretrained = false
```

## 5. Single-task regression

Edit `regression_conf.toml` and verify these fields:

```toml
[BaseConfig]
workdir = "./example/graphormer_training/my_regression_run"
task = "regression"

[DatasetConfig]
dataset_path = "./dataset/my_regression_data.csv"
smiles_column = "SMILES"
target_column = "LogS"

[GraphormerConfig]
loss_type = "huber"
```

Supported regression losses include `huber`, `mse`, `mae`, `laplace_nll`, and
`gaussian_nll`.

Start training:

```bash
chemflow train graphormer example/graphormer_training/regression_conf.toml
```

Predict one molecule:

```bash
chemflow predict graphormer \
  --smiles "CCO" \
  --task-names LogS_prediction \
  --model-checkpoint ./example/graphormer_training/my_regression_run/checkpoints/best_model.pt \
  --output ./example/graphormer_training/my_regression_run/ethanol_prediction.csv
```

Predict a CSV or Parquet dataset:

```bash
chemflow predict graphormer \
  --input ./dataset/my_prediction_data.csv \
  --structure-column SMILES \
  --task-names LogS_prediction \
  --model-checkpoint ./example/graphormer_training/my_regression_run/checkpoints/best_model.pt \
  --batch-size 64 \
  --output ./example/graphormer_training/my_regression_run/predictions.csv
```

## 6. Classification

Edit `classification_conf.toml` and replace its example absolute paths.

For binary classification:

```toml
[BaseConfig]
workdir = "./example/graphormer_training/my_classification_run"
task = "classification"

[DatasetConfig]
dataset_path = "./dataset/my_classification_data.csv"
smiles_column = "SMILES"
target_column = "class_id"

[GraphormerConfig]
num_classes = 1
loss_type = "bce"
```

`positive_weight` is optional. If used, calculate it from the training split
only.

For multiclass classification, use:

```toml
num_classes = 4
loss_type = "cross_entropy"
class_weights = [1.0, 1.0, 1.0, 1.0]
```

Start training:

```bash
chemflow train graphormer example/graphormer_training/classification_conf.toml
```

Run binary prediction with a probability threshold:

```bash
chemflow predict graphormer \
  --input ./dataset/my_classification_test.csv \
  --structure-column SMILES \
  --task-names probability_inactive probability_active predicted_class \
  --model-checkpoint ./example/graphormer_training/my_classification_run/checkpoints/best_model.pt \
  --threshold 0.5 \
  --output ./example/graphormer_training/my_classification_run/predictions.csv
```

For binary classification, `--task-names` must provide three output-column
names: negative probability, positive probability, and predicted label. For
multiclass classification, provide one probability-column name per class plus
one predicted-label name. A four-class example therefore needs five names:

```bash
--task-names probability_0 probability_1 probability_2 probability_3 predicted_class
```

The checkpoint records whether the model is binary or multiclass.

## 7. Multitask regression

Edit `multitask_conf.toml` and replace its example absolute paths. The order of
`target_column` and `task_names` must match, and `num_tasks` must equal their
length.

```toml
[BaseConfig]
workdir = "./example/graphormer_training/my_multitask_run"
task = "multitask"

[DatasetConfig]
dataset_path = "./dataset/my_multitask_data.csv"
smiles_column = "SMILES"
target_column = ["LogD", "LogS", "HLM"]
task_names = ["LogD", "LogS", "HLM"]

[GraphormerConfig]
num_tasks = 3
loss_type = "laplace_NLL"
```

Also update `task_groups` so every task index is assigned appropriately. For
three tasks and one shared group:

```toml
num_adapters = 1
task_groups = [[0, 1, 2]]
```

Start training:

```bash
chemflow train graphormer example/graphormer_training/multitask_conf.toml
```

Predict all tasks in the same order used for training:

```bash
chemflow predict graphormer \
  --input ./dataset/my_multitask_test.csv \
  --structure-column SMILES \
  --task-names LogD_prediction LogS_prediction HLM_prediction \
  --model-checkpoint ./example/graphormer_training/my_multitask_run/checkpoints/best_model.pt \
  --batch-size 64 \
  --output ./example/graphormer_training/my_multitask_run/predictions.csv
```

## 8. Devices and batch size

ChemFlow selects devices in this order:

1. CUDA GPU;
2. Apple MPS;
3. CPU.

Reduce `batch_size` and `eval_batch_size` if training runs out of memory. For
prediction, a device can be selected explicitly:

```bash
chemflow predict graphormer \
  --smiles "CCO" \
  --task-names LogS_prediction \
  --model-checkpoint /path/to/best_model.pt \
  --device mps \
  --output prediction.csv
```

Use `--device cuda`, `--device cuda:0`, or `--device cpu` as appropriate.

## 9. Resume training

Enable resume in the task configuration:

```toml
[GraphormerTrainingConfig]
resume = true
resume_checkpoint = "./example/graphormer_training/my_run/checkpoints/last_model.pt"
```

Then rerun the normal training command. Resume is intended for regular
training and is not supported for cross-validation runs.

## 10. Slurm and multi-GPU DDP

`train.sh` is a Slurm template. Before submitting it:

1. change the partition, GPU, memory, and time requests for the cluster;
2. replace the environment activation placeholder;
3. set `CONFIG` to the desired regression, classification, or multitask TOML;
4. create the log directory before calling `sbatch`.

For a Conda/Mamba environment, its activation section can be:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chemflow
```

Submit the job from the repository root:

```bash
mkdir -p logs
sbatch example/graphormer_training/train.sh
```

When more than one CUDA GPU is allocated, the script launches `torchrun` and
ChemFlow enables DDP. DDP requires CUDA and is not available for Apple MPS.

## 11. Outputs

The configured `workdir` contains the training artifacts, including:

- `checkpoints/best_model.pt`: best checkpoint selected by the monitored metric;
- `checkpoints/last_model.pt`: most recent checkpoint for resuming;
- `config.json`: resolved training configuration;
- `data_splits.csv`: assigned train, validation, and test rows;
- `rejected_rows.csv`: invalid or unusable input rows;
- training history, metrics, plots, and prediction files when enabled.

Use the best checkpoint for normal prediction and final evaluation.
