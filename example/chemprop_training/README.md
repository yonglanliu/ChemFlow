# Chemprop v2 baseline

This backend launches the official Chemprop v2 CLI while using ChemFlow's
TOML configuration and dataset checks.

```bash
chemflow train chemprop example/chemprop_training/conf.toml
```

The default example trains a three-endpoint CL-3 multitask model with the
precomputed cluster-disjoint validation split and evaluates it on CL-Test.
The consolidated dataset contains both `split_val_0.1` and `split_val_0.2`;
select the desired design with `DatasetConfig.split_column`.
Change `target_column` to a single column for a single-task baseline.

Chemprop automatically masks missing endpoint values during multitask
training. The example uses a regularized, higher-capacity D-MPNN configuration
for ADME fine-tuning; treat it as a starting point and compare it with a
single-task and a smaller-capacity control.

After fitting, ChemFlow explicitly predicts with every ensemble member's
validation-selected `best.pt`. This avoids Chemprop 2.2's single-GPU behavior
of evaluating the final in-memory epoch. The following artifacts are written
directly below `workdir`:

- `test_predictions.csv`: truth, direct prediction, MC-dropout mean and SD,
  validation-local calibrated prediction and interval, OOD/local-support
  flags, uncertainty flag, and interval-coverage result.
- `metrics.json` and `test_metrics.json`: RMSE, MAE, median absolute error, R2,
  Pearson, Spearman, and Kendall for direct, MC-dropout, and locally calibrated
  predictions, plus confidence-coverage diagnostics.
- `training_history.csv`: epoch-level scalars exported from Lightning logs.
- `final_model_test_predictions.csv`: Chemprop's original final-epoch output,
  retained for auditing when available.
- `chemprop_launch_manifest.json`: resolved command and reproducibility audit.

TensorBoard events, checkpoints, and split records remain under each
`model_*` directory.

## Early stopping

ChemFlow forwards `early_stopping_patience` to Chemprop v2's native Lightning
early-stopping callback. `tracking_metric` controls both early stopping and
selection of `best.pt`:

```toml
early_stopping_patience = 10
tracking_metric = "val_loss"
```

Chemprop's public CLI uses a fixed `min_delta = 0`; every strict improvement
resets patience. The resolved monitor and patience are printed at launch and
recorded in `chemprop_launch_manifest.json`.

## Continue a run

Set `TrainingConfig.resume = true` to continue from each ensemble member's
`model_*/checkpoints/last.ckpt`. You may instead set `resume_checkpoint` to one
path (or a list of paths for an ensemble). Chemprop v2's public CLI performs a
weights-only continuation: model weights are restored, while optimizer,
scheduler, and epoch-counter state start fresh. Therefore, `num_epochs` is the
number of additional epochs. The selected checkpoint paths and SHA-256 hashes
are recorded in the launch manifest and printed before training. The native
early-stopping callback state is also reset, so continuation starts a new
patience window.

This is a randomly initialized Chemprop D-MPNN baseline. It deliberately does
not pass `--from-foundation CHEMELEON`; runs initialized from CheMeleon weights
must be reported as CheMeleon transfer learning rather than Chemprop baseline.
