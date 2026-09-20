# Parallel clearance training on Slurm

This example submits the four CheMeleon configurations as independent GPU
array tasks and twelve traditional-ML configurations as independent CPU array
tasks. The ML jobs cover three endpoints and four molecular-representation
sets. Array concurrency limits control how many run simultaneously.

From a Slurm login node, run:

```bash
bash example/hpc_clearance_training/submit.sh
```

The default configuration directory is:

```text
/data/liuy48/model_training/adme/clearance
```

To use another directory, pass it as the first argument:

```bash
bash example/hpc_clearance_training/submit.sh /path/to/clearance
```

Logs are written to `<configuration-directory>/logs`. Monitor or cancel the
jobs using the job IDs printed by `submit.sh`:

```bash
squeue -j JOB_ID_1,JOB_ID_2
scancel JOB_ID_1 JOB_ID_2
```

Resource defaults are defined in `chemeleon_array.slurm` and `ml_array.slurm`.
Adjust memory, wall time, GPU requests, or add a site-specific `--partition`
or `--account` directive before submitting. Each CheMeleon task currently
requests one GPU; a configuration that enables multi-GPU DDP must request the
same number of GPUs and Slurm tasks expected by that configuration.

## Generated traditional-ML configurations

`ml_array.slurm` uses `example/ml_training/conf.yaml` as its base configuration.
Each array task creates an endpoint- and representation-specific file before
training. The combinations are HLM, MLM, and RLM crossed with:

```text
[ecfp4]
[ecfp4, rdkit2d]
[rdkit2d]
[ecfp4, rdkit2d, erg, avalon]
```

Generated configuration names follow this pattern:

```text
<training-directory>/generated_configs/ml_<endpoint>_<representations>_conf.yaml
```

The generated files inherit models, features, search settings, and scoring from
the base YAML. The job changes `workdir`, `data.data_file`, `data.y_col`, split
cache directory/name, molecular representations, and scaffold-grouped
cross-validation. Every endpoint/representation combination uses a separate
feature cache, preventing concurrent array tasks from overwriting the same NPZ.

For each endpoint, the ML script first looks for an existing split file:

```text
<training-directory>/<endpoint>/data_splits.csv
```

For example, HLM uses
`<training-directory>/HLM/data_splits.csv` when that file exists. The generated
configuration then uses `split_method: predefined` and `split_column: split`,
so the existing assignments are preserved. If the endpoint file does not
exist, the job uses the curated combined dataset and creates a scaffold split.
`CLEARANCE_DATA_FILE` overrides both choices; set
`CLEARANCE_SPLIT_METHOD=predefined` too when an override file already contains
a `split` column.

Override the template or dataset at submission time when needed:

```bash
export ML_BASE_CONFIG=/data/liuy48/model_training/adme/clearance/ml_base_conf.yaml
export CLEARANCE_DATA_FILE=/data/liuy48/ChemFlow/dataset/curated/Biogen_ExpansionRX_clearance_training.csv
bash example/hpc_clearance_training/submit.sh
```
