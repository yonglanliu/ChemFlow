# Parallel clearance training on Slurm

This example submits the four CheMeleon configurations as independent GPU
array tasks and the three traditional-ML configurations as independent CPU
array tasks. All seven tasks can run concurrently when resources are available.

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
