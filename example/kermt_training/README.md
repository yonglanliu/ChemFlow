# KERMT fine-tuning

ChemFlow vendors the KERMT model, data pipeline, and fine-tuning runtime under
`src/chemflow/deep_learning/kermt/vendor`. No separate KERMT checkout or
`PYTHONPATH` configuration is required. Install ChemFlow's dependencies; the
pinned official pretrained checkpoint and vocabulary files are downloaded on
first use.

For a portable installation, including the RDKit KERMT fallback, run:

```bash
pip install -e '.[kermt]'
```

On NVIDIA Linux/HPC, `environment-hpc.yaml` installs `cuik_molmaker`. On
macOS or CPU-only Linux, use the portable RDKit featurization settings shown in
`conf.toml`.

With an empty `checkpoint_path`, ChemFlow downloads NVIDIA's official release
from `nvidia/NV-KERMT-70M-v2` into
`~/.cache/chemflow/kermt/NV-KERMT-70M-v2`. The download is pinned to an exact
repository revision and logs every cached or downloaded artifact plus the
checkpoint SHA-256. Then edit `conf.toml`, especially:

- `KERMTConfig.checkpoint_path`
- `KERMTConfig.cache_dir`
- `KERMTConfig.local_files_only`
- `KERMTConfig.use_cuikmolmaker_featurization`
- `KERMTConfig.features_generator`
- the dataset, targets, split, and work directory

Run:

```bash
chemflow train kermt example/kermt_training/conf.toml
```

ChemFlow writes KERMT-ready `train.csv`, `val.csv`, and `test.csv` files with a
lowercase `smiles` column, plus `data_splits.csv` and
`kermt_run_manifest.json`. The manifest records the exact command and SHA256 of
the pretrained weights. KERMT output is written below
`WORKDIR/kermt_output`.

To prepare a shared cache on a login node, run one normal dry run with network
access. Compute nodes can subsequently set:

```toml
local_files_only = true
```

An explicit non-empty `checkpoint_path` always overrides automatic download.

For accelerated NVIDIA featurization, use:

```toml
use_cuikmolmaker_featurization = true
features_generator = "rdkit_2d_normalized_cuik_molmaker"
```

For macOS or a CPU-only environment, use:

```toml
use_cuikmolmaker_featurization = false
features_generator = "rdkit_2d_normalized"
```

Set `TrainingConfig.dry_run = true` to validate paths, prepare the splits, and
inspect the command without starting GPU training.

KERMT early stopping is configured with:

```toml
early_stopping_patience = 10
early_stopping_min_delta = 0.001
early_stopping_monitor = "metric" # or "loss"
```

The counter and best monitored value are stored in `last_checkpoint.pt`, so a
resumed run continues the same patience window. `model.pt` remains the checkpoint
with the best validation score and is used for final test evaluation.

The vendored source is derived from NVIDIA-BioNeMo/KERMT commit
`8828743036675d1b6d5f4586ef7bcfea70233d59`. Its license and attribution
notices are included with the package.
