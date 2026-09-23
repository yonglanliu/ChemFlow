# Hugging Face Graphormer training

This is ChemFlow's supported Graphormer backend. It fine-tunes Hugging Face's
implementation with regression and binary classification, split-aware
evaluation, checkpoint resume, multitask loss weighting, DDP, and
applicability-domain calibration.

Install the optional dependencies into the active ChemFlow environment:

```bash
pip install -e '.[hf-graphormer]'
```

The optional dependency intentionally pins `Cython<3`; Graphormer's
preprocessor does not compile with Cython 3.

Run the pH 7.4 kinetic-solubility example:

```bash
chemflow train hf-graphormer example/hf_graphormer_training/conf.toml
```

The same annotated file supports masked multitask regression after changing
`target_column` to a TOML list:

```bash
chemflow train hf-graphormer example/hf_graphormer_training/conf.toml
```

For binary single-task or masked multitask classification, set:

```toml
[BaseConfig]
task = "classification"

[DatasetConfig]
target_column = "active" # or ["endpoint_a", "endpoint_b"]

[TrainingConfig]
classification_threshold = 0.5
```

Then run the same configuration after setting its dataset and target columns:

```bash
chemflow train hf-graphormer example/hf_graphormer_training/conf.toml
```

Classification labels must be `0`, `1`, or missing. Every training endpoint
must contain both classes. The masked loss is binary cross-entropy with logits;
outputs are positive-class probabilities and thresholded classes. Early
stopping minimizes mean validation log loss. Per-task metrics include log
loss, accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC, and
PR-AUC. Validation probabilities are calibrated independently with isotonic
regression when both classes are available.

Mixed multitask learning uses one shared encoder and output head with a
task-specific loss and output transform for each target:

```toml
[BaseConfig]
task = "mixed"

[DatasetConfig]
target_column = ["solubility", "active", "toxic"]
task_types = ["regression", "classification", "classification"]
```

Regression columns use standardized MSE, while classification columns use
BCE-with-logits and sigmoid probabilities. Missing labels remain masked.
Mixed-task early stopping minimizes the macro mean of standardized regression
MSE and classification log loss, preventing a regression endpoint's physical
units from dominating model selection.

After replacing its dataset and target names, run:

```bash
chemflow train hf-graphormer example/hf_graphormer_training/conf.toml
```

For multitask training, `target_column` is a TOML list. A molecule is retained
when at least one target is present, and missing labels are masked. Regression
standardizes each target from its own training labels and minimizes finite-label
squared error. Classification retains binary labels and minimizes finite-label
binary cross-entropy with logits. The configured task weights are applied in
both modes. Early stopping uses mean validation RMSE for regression and mean
validation log loss for classification. The history and final metrics report
every endpoint separately.

The example loads the previously downloaded checkpoint from
`~/.cache/chemflow/graphormer/graphormer-base-pcqm4mv1.pt`, so training does
not contact Hugging Face. Remove `checkpoint_path` and set
`local_files_only = false` only when direct Hugging Face HTTPS access works.

Graphormer's three dropout probabilities are independently configurable under
`[ModelConfig]` and are applied whether initialization uses the local state
dictionary, the Hugging Face Hub/cache, or a ChemFlow transfer checkpoint:

```toml
dropout = 0.0
attention_dropout = 0.1
activation_dropout = 0.1
```

Each value must be in `[0, 1)`. `dropout` controls general residual/embedding
dropout, `attention_dropout` controls attention probabilities, and
`activation_dropout` controls feed-forward activations. Increase them only as
a validated regularization experiment; changing them also changes the
stochastic distribution used by MC-dropout inference.

For a fair comparison, use the `data_splits.csv` produced by the existing run
as `dataset_path` and set `split_column = "split"`. The input must also contain
the configured SMILES and target columns.

Alternatively, keep an independent test set in a physically separate CSV:

```toml
[DatasetConfig]
dataset_path = "/path/to/training.csv"
test_dataset_path = "/path/to/independent_test.csv"
split_type = "scaffold_balanced"
val_fraction = 0.1
test_fraction = 0.0
```

The external file must contain the configured SMILES and target columns. It is
assigned only to the test split and is excluded from generated training and
validation splits, target scaling, early stopping, and calibration. Do not also
place `test` rows in a predefined training file when `test_dataset_path` is set.

Outputs include `data_splits.csv`, `training_history.csv`, `metrics.json`,
`training_task_metrics.csv`, `test_predictions.csv`, and a reloadable
`best_model/` directory. Regression targets are standardized using training
data only and predictions are converted back to the original units before
metrics are calculated. Classification targets are not standardized.

During every epoch, the console reports task-appropriate training and
validation metrics for each endpoint independently. Regression reports RMSE,
MAE, R², Pearson, and Spearman; classification reports the metrics listed
above. `training_history.csv` stores the same values in wide form, while
`training_task_metrics.csv` stores one row per epoch, split, and task plus an
`overall_macro` row. Computing training-set metrics adds one inference pass
over the training set per epoch.

Multitask loss weighting is configured under `[TrainingConfig]`:

```toml
task_loss_weighting = "sqrt_inverse_frequency"
```

Available policies are `uniform`, `sqrt_inverse_frequency`, and
`inverse_frequency`. The square-root policy is the recommended starting point
for imbalanced and potentially noisy ADMET endpoints; full inverse frequency
makes every task contribute approximately equal total training weight but can
strongly amplify a small task. Explicit positive weights can instead be given
in `target_column` order with `task_loss_weights = [1.0, 2.0, 0.5]`. All
weights are normalized to mean 1. Validation and test metrics are never
weighted.

The final `best_model/` also contains:

- `applicability.pt`: byte-packed 2,048-bit radius-2 Morgan fingerprints and
  PCA-compressed fine-tuned Graphormer graph-token embeddings for the training
  chemical space, stored both globally and separately for each labeled task;
- validation structures, targets, predictions, fingerprints, and embeddings
  for local calibration;
- per-task validation residual calibration for regression or isotonic
  probability calibration for classification, plus validation-derived
  fingerprint and embedding OOD thresholds;
- `calibration.json`: a human-readable copy of the residual and OOD calibration.

Local regression calibration uses the nearest labeled validation compounds
that pass both Morgan-Tanimoto and fine-tuned embedding-cosine thresholds. It
uses up to 100 neighbors and requires at least 20; otherwise inference must
fall back to the global validation residual calibration and flag weak local
support.

The test split is not used to construct the applicability domain or calibrate
predictions. On a resumed run, these artifacts are rebuilt from the selected
best model and the unchanged training/validation split.

Run calibrated inference from the saved best model with:

```bash
chemflow predict hf-graphormer \
  --input ./dataset/molecules.csv \
  --structure-column SMILES \
  --model-directory ./hf_graphormer_run/best_model \
  --mc-dropout-samples 50 \
  --calibration-confidence 0.90 \
  --output ./hf_graphormer_run/test_predictions_mc_calibrated.csv
```

For regression, the MC mean becomes the raw prediction, validation calibration
adds global or locally supported bias correction, and validation residuals
anchor the prediction interval. The interval is widened when the Gaussian
MC-dropout radius is larger. The output includes raw predictions,
`mc_mean_<task>`, `mc_std_<task>`, `calibrated_<task>`, interval bounds, and
applicability-domain diagnostics. MC dropout requires a checkpoint containing
nonzero dropout and increases inference time approximately linearly with the
number of passes.

To backfill these files into an already trained `best_model/` without changing
its weights, retain its original work directory and set:

```toml
[TrainingConfig]
applicability_only = true
```

Then run the normal `chemflow train hf-graphormer CONFIG` command. This mode
reads the saved `data_splits.csv` and `target_scaling.json`; it does not run any
optimization epochs.

At startup, the trainer prints total, trainable, and frozen parameter counts,
including separate encoder and prediction-head totals. Each epoch displays a
batch progress bar with current/running loss and both learning rates. Set
`progress_bar = false` under `[TrainingConfig]` to disable it.

To continue an interrupted run, keep the same dataset, split configuration,
work directory, and model settings, then set:

```toml
[TrainingConfig]
resume = true
```

Training resumes from `<workdir>/checkpoints/last.pt`, which is saved after
every completed epoch. Set `resume_checkpoint = "/path/to/last.pt"` to use a
different file. The checkpoint restores the model, optimizer, epoch, early
stopping state, history, target scaling, and random-number states. It loads on
CPU first and then moves state to the currently selected training device; a
different GPU count skips only incompatible CUDA RNG restoration.

## Multi-GPU DDP

HF Graphormer supports single-node CUDA DDP. The configured batch size is per
GPU, so the effective global batch size is `batch_size × number of GPUs`.
Launching four GPUs manually looks like:

```bash
torchrun --standalone --nproc_per_node=4 \
  "$(command -v chemflow)" train hf-graphormer \
  example/hf_graphormer_training/conf.toml
```

Use matching training settings:

```toml
[TrainingConfig]
device = "cuda"
devices = 4
strategy = "ddp"
```

Each rank receives a disjoint training shard through `DistributedSampler`, and
the sampler is reshuffled deterministically each epoch. DDP synchronizes model
gradients and the global training loss. Rank 0 alone runs full validation,
early stopping, checkpointing, test evaluation, and applicability/calibration
artifact generation.
