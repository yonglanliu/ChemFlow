# CheMeleon training and prediction

ChemFlow supports single-task and multitask regression or binary
classification with the pretrained CheMeleon message-passing encoder.

Run commands from the repository root after activating the `chemflow`
environment.

## Multitask data

Set `BaseConfig.task` to `regression` or `classification`. A string
`target_column` trains one task; a TOML list trains multiple tasks:

```toml
[BaseConfig]
task = "regression"

[DatasetConfig]
dataset_path = "./dataset/admet.csv"
smiles_column = "SMILES"
target_column = ["LogD", "LogS", "HLM"]
split_column = "split"
```

Missing labels are allowed in multitask data. A molecule is retained when at
least one target is present, and each task loss masks its missing values. Every
target must have at least one finite label in both the training and validation
splits. Binary classification targets must be `0`, `1`, or missing.

## Training

Single-task regression:

```bash
chemflow train chemeleon example/chemeleon_training/regression_conf.toml
```

Multitask regression:

```bash
chemflow train chemeleon example/chemeleon_training/multitask_conf.toml
```

Training writes `best.ckpt` and `last.ckpt` under the configured
`workdir/checkpoints`. Test metrics are calculated separately for each task and
macro-averaged across tasks. `test_predictions.csv` contains true and predicted
values for every endpoint. The checkpoint directory also contains:

- `history.csv`, with epoch-level training and validation logs;
- `test_metrics.csv`, with one row per endpoint and an `overall_macro` row.

For regression, the per-task CSV reports loss, MAE, RMSE, median absolute
error, R², Pearson correlation, and Spearman correlation. For classification,
it reports loss, accuracy, balanced accuracy, precision, recall, F1, MCC,
ROC-AUC, and PR-AUC.

## Prediction

Task names are read from the checkpoint automatically:

```bash
chemflow predict chemeleon \
  --input ./dataset/molecules.csv \
  --structure-column SMILES \
  --model-checkpoint ./example/chemeleon_training/chemeleon_multitask_admet/checkpoints/best.ckpt \
  --output ./example/chemeleon_training/chemeleon_multitask_admet/predictions.csv
```

Predict one molecule:

```bash
chemflow predict chemeleon \
  --smiles "CCO" \
  --model-checkpoint ./path/to/best.ckpt \
  --output ./prediction.csv
```

Use `--task-names name1 name2 ...` to override output names. The number of
names must equal the checkpoint's number of tasks. Regression produces one
column per task. Classification produces `prob_<task>` and `class_<task>`
columns; change the default decision threshold with `--threshold`.

CSV, Parquet, SMI, SMILES, and text inputs are supported. Invalid molecules
remain in the output with missing prediction values.
