# ChemFlow Studio and ADMET Desktop

ChemFlow Studio is an optional native desktop interface for the existing
ChemFlow command-line workflows. It provides visual forms for training,
prediction, similarity search, GPT generation, and uncertainty evaluation,
plus a live process console and an interactive native molecular scene.

The **ADME clearance** workspace deploys two CheMeleon checkpoints as one
inference product:

- a multitask regression checkpoint for HLM and RLM intrinsic clearance;
- a single-task regression checkpoint for MLM intrinsic clearance.

The complete explanation of model routing, prediction calibration, local and
global uncertainty, fingerprint/embedding similarity, and validation-calibrated
OOD scoring is available in [ADMET_GUIDE.md](ADMET_GUIDE.md). The same guide
opens inside the standalone desktop from the **?** help icon.

For each new molecule and endpoint, ChemFlow compares its training-space FP and
embedding similarity profile with validation compounds. At least 20 comparable
validation compounds enable local bias calibration and a local residual
interval (using up to the closest 100). When local support is insufficient, the
desktop uses the global validation calibration and interval and adds an OOD
warning.

Set `inference.mc_dropout_samples` in the automatic deployment TOML to at
least 2 (typically 20-50) to add MC-dropout epistemic uncertainty. This
requires checkpoints trained with nonzero dropout and increases inference time
approximately linearly with the number of stochastic passes.

Molecules can be entered as SMILES, loaded from CSV, TSV, Parquet, SMI, TXT,
or SDF files, or drawn with the embedded Ketcher editor. Results retain
the input metadata and report both the model's log10 predictions and values in
`mL/min/kg`, calculated with `10 ** prediction`.

## Install

From the repository root, activate the `chemflow` environment and install the
desktop extra:

```bash
mamba activate chemflow
python -m pip install -e ".[desktop]"
chemflow-install-ketcher
```

PySide6 is an optional dependency; installing standard ChemFlow without the
`desktop` extra does not install Qt.

The second command installs the Apache-2.0 Ketcher standalone editor in the
user cache. Ketcher runs locally inside Qt WebEngine, so drawn structures do
not need to be uploaded to an external service. Set `CHEMFLOW_KETCHER_PATH`
to a standalone Ketcher directory (or its `index.html`) to use a managed
installation instead.

## Run

```bash
chemflow-desktop
```

During development, the equivalent command is:

```bash
python -m chemflow.desktop.app
```

The activity console executes workflows using the currently active Python
interpreter, so launch the desktop from the environment containing ChemFlow and
all model dependencies.

## Training center

The **Train** workspace supports conventional ML, Chemprop, Graphormer,
CheMeleon, ChemBERTa, KERMT, GPT, and LSTM experiments. A configuration can be launched in the
desktop's local Python environment or submitted to a Slurm cluster through
SSH. The HPC form records the login host, remote ChemFlow checkout, remote
configuration, Conda environment, partition, GPU/CPU/memory request, and wall
time. It never stores a password or private key; SSH uses the user's existing
agent and SSH configuration.

The job monitor keeps the most recent 100 jobs, displays local PIDs or Slurm
job IDs, streams local output, polls remote `squeue`/`sacct` state every 15
seconds, and can terminate a local process or call `scancel` for a selected
Slurm job. Remote configurations and datasets must already be present on the
HPC filesystem. The desktop submits the equivalent of:

```bash
chemflow train MODEL /remote/path/to/config
```

The SSH host should normally be a login node, not a transient compute node.
Public-key authentication is recommended so monitoring does not repeatedly
prompt for a password.

Each model offers two configuration modes:

- **Existing configuration** runs an advanced TOML, JSON, or YAML file without
  changing it.
- **Build configuration** displays the model's canonical configuration as
  typed controls grouped by section. Choice fields constrain valid values,
  numeric fields enforce ranges, and every parameter has a `?` explanation.

The parameter schemas and defaults live in
`chemflow.desktop.training_parameters`, so desktop forms and generated files
are versioned with the source code. Generated Chemprop, Graphormer, CheMeleon,
ChemBERTa, KERMT, GPT, and LSTM configurations use TOML; conventional ML uses JSON.
For local execution, the generated file is passed directly to ChemFlow. For
Slurm execution, it is saved locally, uploaded over SCP to the configured
remote path, and then submitted. The existing-file mode assumes the specified
remote configuration already exists on the cluster.

## Standalone ADMET Desktop

Launch the focused clearance-deployment application with:

```bash
admet-desktop
```

The desktop automatically loads `example/admet_desktop/config.toml`, so the
usual launch command needs no configuration argument. Automatic discovery uses
this priority:

1. `CHEMFLOW_ADMET_CONFIG`;
2. `~/.config/chemflow/admet_desktop.toml`;
3. the repository example config.

An explicit path remains available as an optional override:

```bash
admet-desktop --config ./example/admet_desktop/config.toml
```

The `[checkpoints]` entries accept absolute paths, paths relative to the TOML
file, `~`, and environment variables such as `${HOME}`. Deployment settings
are deliberately hidden in the prediction interface; edit the automatically
loaded TOML when a checkpoint or inference setting changes.

The optional `[quality_control]` table defines the review thresholds:

```toml
[quality_control]
min_tanimoto_similarity = 0.35
max_embedding_distance = 0.35
max_log_interval_width = 1.0
ood_score_threshold = 0.95
```

Current checkpoints use each endpoint's validation set to calibrate empirical
fingerprint-novelty and embedding-distance OOD scores. Scores approach 1 as a
compound becomes more extreme relative to validation chemistry, and the OOD
cutoff defaults to 0.95. The raw similarity and distance thresholds remain a
fallback for older checkpoints. A result is also flagged when its calibrated
log10 interval is wider than the configured maximum. Checkpoints without
calibration or applicability-domain data are marked `NOT ASSESSED` and
conservatively flagged for review.

During development, use:

```bash
python -m chemflow.desktop.admet_app
```

This standalone application contains only the deployed ADMET workflow: the
multitask HLM/RLM checkpoint, single-task MLM checkpoint, molecule input and
sketching tools, prediction table, unit conversion, and CSV export. The same
workflow also remains available inside the broader `chemflow-desktop` studio.

The left property panel groups **HLM CLint**, **MLM CLint**, and **RLM CLint**
under **Metabolism**. These endpoints support multiple selection, and the
**Select all properties** action enables the complete group. Only checkpoints
required by the active selection are loaded and only the selected result
columns are exported. The categorized panel is the extension point for
additional ADMET endpoints.
