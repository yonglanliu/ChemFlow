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
training. TensorBoard events, checkpoints, test predictions, reproducible
split records, and `chemprop_launch_manifest.json` are written below `workdir`.

## Continue a run

Set `TrainingConfig.resume = true` to continue from each ensemble member's
`model_*/checkpoints/last.ckpt`. You may instead set `resume_checkpoint` to one
path (or a list of paths for an ensemble). Chemprop v2's public CLI performs a
weights-only continuation: model weights are restored, while optimizer,
scheduler, and epoch-counter state start fresh. Therefore, `num_epochs` is the
number of additional epochs. The selected checkpoint paths and SHA-256 hashes
are recorded in the launch manifest and printed before training.

This is a randomly initialized Chemprop D-MPNN baseline. It deliberately does
not pass `--from-foundation CHEMELEON`; runs initialized from CheMeleon weights
must be reported as CheMeleon transfer learning rather than Chemprop baseline.
