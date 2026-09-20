# ADMET Desktop prediction guide

This guide describes which CheMeleon model is used for each clearance endpoint
and how ChemFlow calculates calibration, uncertainty, chemical similarity,
embedding similarity, and out-of-domain (OOD) scores.

## Model routing

| Endpoint | Measurement | Deployed checkpoint |
|---|---|---|
| HLM | Human liver microsomal intrinsic clearance | Multitask CheMeleon |
| RLM | Rat liver microsomal intrinsic clearance | Multitask CheMeleon |
| MLM | Mouse liver microsomal intrinsic clearance | Single-task CheMeleon |

The multitask checkpoint contains an MLM output internally, but the desktop
does not use it. Every displayed MLM prediction comes from the single-task MLM
checkpoint.

All models predict `log10(mL/min/kg)`. The desktop converts a log prediction
`z` to the displayed normal unit with:

```text
CLint (mL/min/kg) = 10^z
```

## Data flow

```text
TRAINING SET
  ├─ trains the CheMeleon model
  └─ builds the endpoint-specific reference chemical space
       ├─ packed Morgan fingerprints
       └─ PCA-projected CheMeleon embeddings

VALIDATION / CALIBRATION SET
  ├─ generates held-out predictions
  ├─ calculates prediction residuals
  ├─ measures FP similarity to the training reference
  ├─ measures embedding similarity to the training reference
  ├─ calibrates prediction bias and intervals
  └─ calibrates empirical OOD scores

NEW COMPOUND
  ├─ raw and calibrated prediction
  ├─ calibrated low/high interval
  ├─ endpoint-specific FP and embedding similarity
  └─ endpoint-specific and combined OOD scores
```

The test split is not used for calibration.

## Raw and calibrated predictions

For validation molecule `i`, let `y_i` be its observed log10 clearance and
`p_i` the model prediction. ChemFlow calculates a robust median bias separately
for HLM, RLM, and MLM:

```text
bias = median(y_i - p_i)
calibrated prediction = raw prediction + bias
```

The result table reports both values after conversion to `mL/min/kg`:

- `*_pred_raw`: the uncorrected model prediction;
- `*_pred_calibrated`: the validation-bias-corrected prediction.

## Prediction uncertainty

When `mc_dropout_samples` is at least 2, ChemFlow keeps dropout layers
stochastic for repeated inference passes while all other layers remain in
evaluation mode. Their mean and standard deviation estimate epistemic model
instability. This requires a checkpoint trained with nonzero dropout.

The held-out validation residual interval remains the coverage anchor. If the
Gaussian MC-dropout radius is larger, ChemFlow widens that interval rather
than reporting an overconfident bound.

After bias correction, ChemFlow calculates absolute validation residuals:

```text
residual_i = |y_i - (p_i + bias)|
```

The configured validation quantile (90% by default) becomes the log10 interval
radius. For a calibrated log prediction `c` and radius `r`:

```text
low_log  = c - r
high_log = c + r
```

After conversion to normal units:

```text
prediction = 10^c
low        = prediction / 10^r
high       = prediction * 10^r
```

This explains why the normal-unit interval is multiplicative and asymmetric.
`*_uncertainty_log` is the complete log10 interval width, `2r`.

The task-level interval is used as a global fallback. ChemFlow first attempts
the local procedure below. For OOD molecules, nominal interval coverage is not
guaranteed.

## Local calibration and local uncertainty

For each new molecule and endpoint, ChemFlow searches the endpoint's validation
set directly for nearest neighbors in two molecular spaces:

```text
Morgan-fingerprint Tanimoto similarity
fine-tuned embedding cosine similarity
```

Neighbors must pass both similarity thresholds, and the closest 100 are
retained using their combined fingerprint and embedding distance.

If at least 20 comparable compounds are available, their signed prediction
errors determine a local median bias and their locally bias-corrected residuals
determine the local uncertainty radius:

```text
local bias = median(local observed - local predicted)
local radius = 90% finite-sample quantile of local absolute residuals
```

This makes the calibrated prediction and uncertainty molecule-dependent.
The result table reports `*_local_n` and `*_calibration = local`.

If fewer than 20 comparable compounds are available, ChemFlow retains the
endpoint's global bias and global residual interval, reports
`*_calibration = global_fallback`, and adds an OOD warning. This avoids
estimating a local interval from an unstable, undersized neighborhood.

## Fingerprint similarity

Morgan fingerprints use radius 2 and 2,048 bits. For each endpoint, ChemFlow
reports the largest Tanimoto similarity between the query and any labeled
training molecule:

```text
HLM_FP_sim
RLM_FP_sim
MLM_FP_sim
```

Higher similarity means closer chemical structure and generally stronger
training-set support.

## CheMeleon embedding similarity

ChemFlow extracts the fine-tuned model embedding and projects it into the
128-dimensional PCA space stored in the checkpoint. Internally it finds the
smallest cosine distance to an endpoint's labeled training embeddings, then
displays the equivalent cosine similarity:

```text
embedding similarity = 1 - embedding distance
```

The result columns are `HLM_EB_sim`, `RLM_EB_sim`, and `MLM_EB_sim`.
Higher similarity means closer to the learned training representation.

## Validation-calibrated OOD scores

Validation molecules are first compared with the appropriate endpoint-specific
training reference. ChemFlow stores their fingerprint novelty (`1 - FP_sim`)
and embedding-distance distributions.

For a new molecule, each diagnostic is converted to an empirical upper-tail
score. The endpoint OOD score is the larger of the fingerprint and embedding
scores:

```text
endpoint OOD score = max(FP tail score, embedding tail score)
```

Scores closer to 1 are more unusual relative to validation chemistry. With
`ood_score_threshold = 0.95`, approximately the most extreme validation tail
defines OOD. The endpoint columns are:

```text
HLM_OOD_score
RLM_OOD_score
MLM_OOD_score
```

When several endpoints are selected, the desktop also reports:

- `OOD_score_best`: lowest endpoint OOD score;
- `OOD_score_worst`: highest endpoint OOD score;
- `FP_sim_best` / `FP_sim_worst`: highest / lowest endpoint similarity;
- `EB_sim_best` / `EB_sim_worst`: highest / lowest endpoint similarity.

Per the current deployment policy, the combined `OOD` flag uses
`OOD_score_best`. Therefore, a molecule passes the combined domain check if at
least one selected model considers it in-domain. Inspect the endpoint-specific
scores and worst score before trusting every endpoint prediction.

## Flags

| Flag | Meaning |
|---|---|
| `PASS` | Domain and uncertainty checks passed |
| `OOD` | The validation-calibrated OOD score crossed its threshold |
| `HIGH UNCERTAINTY` | The validation residual interval is wider than configured |
| `OOD + HIGH UNCERTAINTY` | Both conditions apply |
| `NOT ASSESSED` | The checkpoint lacks required calibration/domain information |

Flagged rows are highlighted in the table.

## Configuration

The desktop automatically loads `example/admet_desktop/config.toml`. Important
settings include:

```toml
[inference]
calibration_confidence = 0.90
embedding_dimensions = 128
similarity_radius = 2
similarity_bits = 2048
mc_dropout_samples = 30  # use 0 for checkpoints trained without dropout

[quality_control]
ood_score_threshold = 0.95
max_log_interval_width = 1.0
```

`min_tanimoto_similarity` and `max_embedding_distance` remain fallback rules
for older checkpoints without validation-calibrated OOD distributions.

## Interpretation guidance

- Calibration corrects systematic validation bias; it does not make an OOD
  prediction reliable.
- Prediction intervals describe historical validation error, not certainty
  about an individual molecule.
- FP similarity measures structural proximity; embedding similarity measures
  proximity in the learned CheMeleon representation.
- OOD is a warning about applicability, not proof that a prediction is wrong.
- Review endpoint-specific metrics whenever the selected endpoints use
  different models or labeled reference subsets.
