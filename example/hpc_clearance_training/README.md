# Clearance model comparison on Slurm

This suite compares Graphormer, Chemprop v2, CheMeleon, KERMT, and a
traditional ECFP4/LightGBM baseline on the curated clearance datasets.

The paths are intentionally fixed for the NIH HPC setup:

```text
Repository:     /vf/users/liuy48/ChemFlow
Slurm bundle:   /data/liuy48/model_training/adme/clearance
Datasets:       /data/liuy48/model_training/adme/clearance/dataset/splits
Outputs:        /data/liuy48/model_training/adme/clearance
```

The four deep-learning arrays each run six multitask experiments:

```text
CL-1, CL-2, CL-3 × split_val_0.1, split_val_0.2
```

Each model predicts HLM, RLM, and MLM clearance jointly and uses CL-Test only
as the independent final test set. The traditional ML implementation is
single-task, so its array runs 18 experiments: the same six dataset/split
conditions crossed with HLM, RLM, and MLM. It uses the ECFP4 representation
and the LightGBM model defined in `example/ml_training/conf.yaml`.

## Submit all arrays

From the HPC login node:

```bash
cd /data/liuy48/model_training/adme/clearance
bash submit.sh
```

To submit one model only:

```bash
sbatch graphormer_array.slurm
sbatch chemprop_array.slurm
sbatch chemeleon_array.slurm
sbatch kermt_array.slurm
sbatch ml_array.slurm
```

Do not run or source a `.slurm` file directly. `sbatch` supplies
`SLURM_ARRAY_TASK_ID` and allocates the requested GPU/CPU resources.

Generated configurations are written to:

```text
/data/liuy48/model_training/adme/clearance/generated_configs
```

When submitted through `submit.sh`, standard output and error logs are written
below `/data/liuy48/model_training/adme/clearance/logs`. Each run has its own
work directory named with model, dataset, validation fraction, and seed.

KERMT uses the cached official checkpoint and accelerated
`cuik_molmaker` featurization. Graphormer likewise uses the cached local
checkpoint. Confirm both checkpoint files exist before submission.

The runnable bundle also includes local base templates named
`graphormer_conf.toml`, `chemprop_conf.toml`, `chemeleon_conf.toml`,
`kermt_conf.toml`, and `ml_conf.yaml`. The array scripts do not depend on the
example configuration directories in the repository checkout.
