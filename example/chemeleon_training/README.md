# CheMeleon training and prediction

ChemFlow supports single-task, homogeneous multitask, and mixed
regression/binary-classification learning with the pretrained CheMeleon
message-passing encoder.

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

For mixed regression and binary classification endpoints, use:

```toml
[BaseConfig]
task = "mixed"

[DatasetConfig]
target_column = ["solubility", "active", "toxic"]
task_types = ["regression", "classification", "classification"]
```

The ordering of `task_types` must match `target_column` (a name-to-type TOML
table is also accepted). Regression outputs are scaled and trained with MSE;
classification outputs use BCE-with-logits and sigmoid probabilities.
Validation loss and early stopping combine standardized regression MSE with
classification log loss.

An editable template is provided at `mixed_conf.toml`. After replacing its
dataset and target names, run:

```bash
chemflow train chemeleon example/chemeleon_training/mixed_conf.toml
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

CheMeleon's `best.ckpt` also contains an applicability-domain package by
default: training and validation SMILES/targets, compact PCA-projected
fine-tuned embeddings, validation predictions, and calibration inputs.

With `inspect_task_metrics = true` (the default), every epoch evaluates and
prints training and validation metrics separately for each endpoint.
Regression reports MAE, RMSE, median absolute error, R², Pearson, and
Spearman; classification reports accuracy, balanced accuracy, precision,
recall, F1, MCC, ROC-AUC, and PR-AUC. The values are saved to
`training_task_metrics.csv` with one row per epoch, split, and task plus an
`overall_macro` row, and are also sent to the Lightning CSV logger. This extra
full-dataset evaluation increases training time.

For multitask datasets with unequal label counts, configure optimization loss
weighting under `[CheMeleonTrainingConfig]`:

```toml
task_loss_weighting = "sqrt_inverse_frequency"
```

The supported policies are `uniform`, `sqrt_inverse_frequency` (recommended
as the first imbalanced/noisy ADMET experiment), and `inverse_frequency`.
Explicit weights may be supplied in `target_column` order with
`task_loss_weights = [1.0, 2.0, 0.5]`. Weights are normalized to mean 1,
printed at startup, and saved in `config.json`. Metrics remain unweighted.
Packaging happens after gradient-based training and requires no training
configuration. Optimizer state is not stored, so these files do not support
optimizer-level training resume.

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

Estimate inference-time epistemic uncertainty with Monte Carlo dropout:

```bash
chemflow predict chemeleon \
  --input ./dataset/molecules.csv \
  --structure-column SMILES \
  --model-checkpoint ./path/to/best.ckpt \
  --mc-dropout-samples 30 \
  --output ./predictions.csv
```

This adds `mc_mean_<task>` and `mc_std_<task>` columns, and the MC mean is used
as the raw prediction. For regression, the validation-calibrated interval is
widened when the MC-dropout radius is larger. The checkpoint must have been
trained with `dropout > 0`; 20-50 passes are typically practical and inference
time grows approximately with the number of passes.

Use `--task-names name1 name2 ...` to override output names. The number of
names must equal the checkpoint's number of tasks. Regression produces raw and
validation-calibrated values plus a validation-residual interval. Classification
produces raw and isotonic-calibrated probabilities and classes; change the
decision threshold with `--threshold`.

Checkpoints with applicability data also report `train_max_tanimoto` and
`train_embedding_cosine_distance`, together with the nearest training SMILES
for each measure. High Tanimoto similarity and low embedding distance indicate
that a query is closer to the training domain, but do not guarantee accuracy.

Applicability and calibration behavior is configured during inference:

```bash
chemflow predict chemeleon \
  --input ./dataset/molecules.csv \
  --model-checkpoint ./path/to/best.ckpt \
  --embedding-dimensions 128 \
  --calibration-confidence 0.90 \
  --similarity-radius 2 \
  --similarity-bits 2048 \
  --mc-dropout-samples 30 \
  --output ./predictions.csv
```

Use `--no-applicability-domain` to return model predictions without calculating
calibration, similarity, or embedding-distance columns.

CSV, Parquet, SMI, SMILES, and text inputs are supported. Invalid molecules
remain in the output with missing prediction values.
