# ChemFlow Studio and ADMET Desktop

ChemFlow Studio is an optional native desktop interface for the existing
ChemFlow command-line workflows. It provides visual forms for prediction,
similarity search, and GPT generation, plus deployed physicochemical and ADME
models, a live process console, and an interactive native molecular scene.

The **Property prediction** workspace puts **Physicochemical** first and
**Clearance** second. Molecular input is carried between the two tabs so
the same compounds can be evaluated across both model families. The
physicochemical tab includes the packaged `logd_best.pt` and
`solubility_best.pt` Chemprop models. It provides LogD plus kinetic and
thermodynamic-solubility endpoints and exports predictions to CSV. No
checkpoint selection is required.

The workspace can run MC-dropout inference (30 passes by default) and reports
the MC mean, standard deviation, and 95% interval. Solubility values are shown
in µM while LogS remains on its native log10(mol/L) scale. Endpoint-specific
Morgan similarity identifies the nearest training compound, and validation
similarity distributions calibrate an empirical OOD score and flag at the
95% threshold.

After prediction, **Open selection agent…** launches a separate agentic window.
The assistant uses constrained tools to inspect numeric ranges, filter and rank
rows, and explicitly select compounds; selected rows are highlighted and can be
exported as a separate CSV with the selection rationale. **Continue with
ChatGPT** opens browser authorization and lets eligible users run the agent with
their ChatGPT plan. OAuth credentials are kept in the operating-system keychain,
can be disconnected from the window, and are never placed in exports or logs.
Managed Work or Enterprise workspaces may disallow third-party ChatGPT-plan
usage through workspace policy. In that case, use an eligible personal Plus/Pro
workspace or the API-key option; ChemFlow cannot override workspace policy.

As an alternative, paste an API key into the password-masked field; it remains
in memory only until that window closes. The environment variable also remains
supported:

```bash
export OPENAI_API_KEY="your-api-key"
# Optional; defaults to gpt-5.5
export CHEMFLOW_AGENT_MODEL="gpt-5.5"
chemflow-desktop
```

Queried prediction rows, including structures and requested metrics, are sent
to OpenAI when the agent is used. The agent is decision support only: model
predictions and AI selections require experimental review and confirmation.

The **Clearance** tab deploys two CheMeleon checkpoints as one
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

This standalone application contains the deployed property workflows: the
multitask HLM/RLM checkpoint, single-task MLM checkpoint, packaged LogD and
solubility models, molecule input tools, prediction tables, unit conversion,
and CSV export. The same workflows also remain available inside the broader
`chemflow-desktop` studio.

The standalone sidebar also opens the packaged Physicochemical workspace for
LogD and kinetic-solubility prediction.

The left property panel groups **HLM CLint**, **MLM CLint**, and **RLM CLint**
under **Metabolism**. These endpoints support multiple selection, and the
**Select all properties** action enables the complete group. Only checkpoints
required by the active selection are loaded and only the selected result
columns are exported. The categorized panel is the extension point for
additional ADMET endpoints.
