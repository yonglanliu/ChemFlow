# Traditional machine-learning training and prediction

This tutorial shows how to train and use ChemFlow's traditional machine-learning
models for single-task regression and classification. Run all commands from the
ChemFlow repository root.

Set top-level `resume: true` to restart a multi-model or multi-seed workflow.
ChemFlow verifies that both the completed summary and serialized model package
exist before skipping a fit. An interrupted individual scikit-learn CV search
restarts that model because sklearn does not expose candidate-level checkpoint
state.

## 1. Activate ChemFlow

```bash
conda activate chemflow
cd /path/to/ChemFlow
chemflow --help
```

If optional model dependencies are missing, install them in the ChemFlow
environment:

```bash
mamba install -n chemflow -c conda-forge scikit-learn xgboost lightgbm rdkit
```

## 2. Prepare a dataset

For a normal training run, use a CSV or Parquet file containing one SMILES
column and one target column. Invalid SMILES, missing SMILES, and rows with a
missing target are removed before training.

Regression example:

```csv
SMILES,LogD
CCO,-0.31
CCN,-0.52
c1ccccc1,2.13
```

Binary classification example:

```csv
SMILES,active
CCO,0
CCN,1
c1ccccc1,1
```

ChemFlow currently trains one traditional-ML target per configuration. To
train several endpoints, create one configuration and run per target.

The supplied `conf.yaml` expects this file:

```text
example/ml_training/single_task_LogD.csv
```

Create that file or change `data.data_file` to the path of your own dataset.

## 3. Choose molecular features

Set one or more entries under `featurization.features`. Multiple feature types
are concatenated in the order listed.

| Feature | Description |
| --- | --- |
| `ecfp4` | ECFP fingerprint with radius 2 |
| `ecfp6` | ECFP fingerprint with radius 3 |
| `fcfp4` | Feature-connectivity fingerprint with radius 2 |
| `fcfp6` | Feature-connectivity fingerprint with radius 3 |
| `maccs` | 167-bit MACCS keys |
| `avalon` | 2,048-bit Avalon path and topology fingerprint |
| `erg` | 315-value ErG pharmacophore fingerprint |
| `descriptor` | Legacy nine-descriptor RDKit physicochemical subset |
| `rdkit2d` | Expanded RDKit 2D set; includes all nine legacy descriptors |

For fingerprint features, `fp_bits` sets the fingerprint length:

```yaml
featurization:
  features:
    - ecfp4
    - rdkit2d
  fp_bits: 2048
```

The expanded set contains 209 descriptors with RDKit 2023.09.6. The exact
available set is RDKit-version dependent; ChemFlow excludes the numerically
unstable `Ipc` descriptor, records the ordered descriptor names and RDKit
version in the model package, and reuses that order during prediction. Its
first nine entries are `MolWt`, `MolLogP`, `TPSA`, `NumHAcceptors`,
`NumHDonors`, `NumRotatableBonds`, `RingCount`, `HeavyAtomCount`, and
`FractionCSP3`.

The saved ChemFlow model package includes the fitted feature preprocessing, so
prediction data should contain SMILES rather than manually generated features.

### How the molecular representations differ

Each representation describes a different aspect of molecular structure:

| Representation | Main information | Typical dimensions | ChemFlow support |
| --- | --- | ---: | --- |
| RDKit2D | Whole-molecule physicochemical, topological, charge, ring, and fragment properties | 209 with RDKit 2023.09.6 | Yes: `rdkit2d` |
| ECFP4 | Circular atom environments extending two bonds from each atom | 2,048 bits by default | Yes: `ecfp4` |
| Avalon | Hashed molecular paths, atom environments, and broader topological patterns | 2,048 bits | Yes: `avalon` |
| ErG | Pharmacophore feature types and the topological distances between them | 315 values | Yes: `erg` |

The **Avalon fingerprint** provides a structural view that differs from ECFP4.
It hashes paths, atom environments, and other molecular patterns into a
fixed-length binary vector. This can capture scaffold and connectivity
information that may not be represented identically by circular fingerprints.

The **ErG fingerprint** is an extended reduced-graph representation. It groups
atoms into pharmacophore categories, such as hydrogen-bond donors, acceptors,
aromatic groups, and hydrophobic groups, and then records their topological
separations. It therefore describes the arrangement of pharmacophore features
without requiring a generated three-dimensional conformer.

These representations are complementary rather than interchangeable:

- RDKit2D describes global molecular properties;
- ECFP4 describes local chemical environments;
- Avalon provides another view of molecular topology;
- ErG describes pharmacophore-feature arrangements.

For clearance regression, start by comparing these currently supported
configurations on identical scaffold splits:

```yaml
featurization:
  features: [rdkit2d]
```

```yaml
featurization:
  features: [rdkit2d, ecfp4]
```

Additional combinations worth benchmarking are:

```yaml
featurization:
  features: [rdkit2d, erg, ecfp4, avalon]
```

```yaml
featurization:
  features: [rdkit2d, erg, avalon]
```

Adding more representations does not guarantee better performance. Select the
feature combination using validation results, preferably across repeated
scaffold splits, and reserve the test set for the final evaluation.

### Descriptor reduction and leakage prevention

ChemFlow performs feature reduction inside the sklearn training pipeline. This
means every filter is fitted using only the training portion of each
cross-validation fold; validation-fold statistics are never used to choose
features.

The default sequence is:

1. Remove zero-variance features.
2. Among RDKit descriptor columns, retain the first descriptor from each group
   whose absolute Pearson correlation exceeds `0.95`.
3. Standardize the retained descriptor columns.
4. Pass fingerprint bits through without correlation pruning or supervised
   selection.
5. Remove fingerprint bits that are constant in the current training fold.

Correlation filtering is limited to RDKit2D because its continuous molecular
measurements contain substantial redundancy. For example, molecular weight,
heavy-atom molecular weight, atom counts, surface-area terms, and related
topological quantities can carry nearly identical information.

ECFP4 and Avalon dimensions instead indicate particular structural patterns.
Two bits may be statistically correlated in one training set while still
representing different chemical fragments, so discarding one can remove useful
structural identity. ErG values encode distinct pharmacophore-type and
topological-distance combinations. Although ErG is not binary, correlated ErG
dimensions can likewise retain different chemical meanings. Pearson
correlation pruning is therefore not applied to ECFP, FCFP, Avalon, MACCS, or
ErG features.

| Representation | Zero-variance filter | Correlation filter | Supervised descriptor selection |
| --- | ---: | ---: | ---: |
| RDKit2D or legacy descriptors | Yes | Yes | Optional for linear/SVM models |
| ECFP or FCFP | Yes | No | No |
| Avalon | Yes | No | No |
| MACCS | Yes | No | No |
| ErG | Yes | No | No |

This policy is particularly suitable for tree ensembles, which select useful
fingerprint dimensions through their splits and feature subsampling. For a
linear model or SVM, sparse fingerprints may still be too large. A future
fingerprint-specific selector could use mutual information, chi-squared
selection for binary bits, or truncated SVD. Such a selector must also be
fitted inside each cross-validation training fold; ordinary Pearson
correlation filtering is not the preferred method for sparse fingerprint bits.

Configure the correlation cutoff under `featurization`:

```yaml
featurization:
  features: [rdkit2d, ecfp4]
  descriptor_correlation_threshold: 0.95
```

An optional Extra Trees selector can reduce the descriptor block further for
linear and SVM estimators. It is disabled by default because supervised feature
selection adds computation and must demonstrate an improvement during
validation:

```yaml
featurization:
  features: [rdkit2d, ecfp4]
  descriptor_correlation_threshold: 0.95
  descriptor_model_selection: true
  descriptor_selection_threshold: mean
  descriptor_selection_estimators: 128
```

The supervised selector is automatically skipped for Random Forest, XGBoost,
LightGBM, and other tree estimators. Their ECFP4 bits remain unchanged except
for the universal zero-variance filter. The fitted correlation and selection
masks are stored inside the serialized model pipeline and reused during
prediction.

Because the retained dimension depends on the compounds in each training set,
it cannot be known exactly before fitting. Every completed model now records a
`feature_reduction` object in both its `*_summary.json` and
`*_model_info.json` files. For example:

```json
{
  "raw_total": 4620,
  "descriptor_raw": 209,
  "descriptor_after_zero_variance": 180,
  "descriptor_after_correlation": 120,
  "descriptor_after_model_selection": 120,
  "fingerprint_raw": 4411,
  "fingerprint_after_zero_variance": 2600,
  "final_model_input": 2720
}
```

These are illustrative values; use the JSON produced by a particular run for
its exact fitted dimension. The same diagnostics are stored in the model
package metrics.

## 4. Configure data splitting

The supported split methods are:

- `random`: randomly assign molecules;
- `scaffold`: keep each Bemis-Murcko scaffold in only one split;
- `butina`: group similar molecules using Butina clustering;
- `cluster`: group molecules using fingerprint clustering;
- `predefined`: directly use existing split labels from a dataset column
  (`splitted` remains accepted as a legacy alias).

For a generated random or scaffold split:

```yaml
data_split:
  split_method: scaffold
  test_fraction: 0.1
  valid_fraction: 0.1
  random_seed: 42
  save_split_data: true
  save_dir: ./example/ml_training/my_run/split_data
  split_name: logd_scaffold_seed42
```

For `butina`, also provide `fp_radius`, `fp_n_bits`, and `butina_cutoff`. For
`cluster`, provide `fp_radius`, `fp_n_bits`, and `n_clusters`.

To use predefined splits, add a split column to the data:

```csv
SMILES,LogD,split
CCO,-0.31,train
CCN,-0.52,val
c1ccccc1,2.13,test
```

Then configure:

```yaml
data_split:
  split_method: predefined
  split_column: split
  save_split_data: true
  save_dir: ./example/ml_training/my_run/split_data
  split_name: logd_predefined_v1
```

Accepted labels include `train`, `training`, `val`, `valid`, `validation`,
`dev`, `test`, and `testing`. A test split is required; validation is optional.
ChemFlow uses these assignments exactly and does not randomly or
scaffold-split the rows again.

If the splits are separate CSV, Parquet, or pickle files, provide them directly:

```yaml
data:
  train_data_file: ./data/train.csv
  validation_data_file: ./data/validation.csv  # optional
  test_data_file: ./data/test.csv
  X_col: SMILES
  y_col: LogD
```

In this form, `data_file` is not required. ChemFlow featurizes each supplied
file independently and preserves its train/validation/test role.

When `save_split_data` is enabled, ChemFlow directly reuses the cached NPZ
split on later runs. Older caches containing split indices are upgraded by
rebuilding features for those exact indices; molecules are not repartitioned.
The requested feature list is stored in the NPZ. If it changes, ChemFlow keeps
the saved train/validation/test indices but automatically rebuilds the feature
arrays. If you change the input data or split settings, use a new `split_name`
or remove the old cached split because those changes define a different split.

ChemFlow also writes `<split_name>_split_data.csv` in `save_dir`. This
task-specific file contains only the configured structure column, selected
target column, and a canonical `split` column containing `train`, `validation`,
or `test`. Other endpoints from a multitarget source table are excluded. The
NPZ remains the fast feature cache used by training; the CSV is intended for
inspection and reuse in later configurations.

## 5. Single-task regression

The included `conf.yaml` is a regression example with hyperparameter tuning.
After updating its dataset path and columns, run:

```bash
chemflow train ml example/ml_training/conf.yaml
```

A smaller regular-training configuration looks like this:

```yaml
workdir: ./example/ml_training/logd_regression
task_type: regression

data:
  data_file: ./dataset/logd.csv
  X_col: SMILES
  y_col: LogD

featurization:
  features: [ecfp4, rdkit2d]
  fp_bits: 2048

data_split:
  split_method: scaffold
  test_fraction: 0.1
  valid_fraction: 0.1
  random_seed: 42
  save_split_data: true
  save_dir: ./example/ml_training/logd_regression/split_data
  split_name: logd_scaffold_seed42

models:
  - model_name: Random Forest
    estimator: RandomForestRegressor
    model_params:
      n_estimators: 500
      max_depth: 20
      n_jobs: -1
  - model_name: LightGBM
    estimator: LGBMRegressor
    model_params:
      n_estimators: 500
      learning_rate: 0.05

hyperparameter_tuning: false
seeds: [42]
scoring_metrics:
  - r2
  - root_mean_squared_error
  - mean_absolute_error
  - pearson
refit_metric: root_mean_squared_error
```

Available regression metrics are `r2`, `root_mean_squared_error`,
`mean_absolute_error`, `pearson`, `spearman`, and `kendall`.

## 6. Classification

Classification targets should be integer class labels. Set `n_classes` to the
number of classes and use classifier estimators:

```yaml
workdir: ./example/ml_training/activity_classification
task_type: classification

data:
  data_file: ./dataset/activity.csv
  X_col: SMILES
  y_col: active
  n_classes: 2

featurization:
  features: [ecfp4]
  fp_bits: 2048

data_split:
  split_method: scaffold
  test_fraction: 0.1
  valid_fraction: 0.1
  random_seed: 42
  save_split_data: true
  save_dir: ./example/ml_training/activity_classification/split_data
  split_name: activity_scaffold_seed42

models:
  - model_name: Random Forest
    estimator: RandomForestClassifier
    model_params:
      n_estimators: 500
      class_weight: balanced
      n_jobs: -1
  - model_name: Logistic Regression
    estimator: LogisticRegression
    model_params:
      C: 1.0
      class_weight: balanced
      max_iter: 2000

hyperparameter_tuning: false
seeds: [42]
scoring_metrics:
  - roc_auc
  - average_precision
  - balanced_accuracy
  - f1
refit_metric: roc_auc
```

Binary classification supports `roc_auc`, `average_precision`,
`balanced_accuracy`, `f1`, `precision`, and `recall`. Multiclass
classification supports all of those except `average_precision`.

Train it with:

```bash
chemflow train ml ./path/to/classification_conf.yaml
```

## 7. Available models

Use the displayed `model_name` exactly and select the matching estimator for
the task.

| Model name | Regression estimator | Classification estimator |
| --- | --- | --- |
| `Random Forest` | `RandomForestRegressor` | `RandomForestClassifier` |
| `Extra Trees` | `ExtraTreesRegressor` | `ExtraTreesClassifier` |
| `Gradient Boosting` | `GradientBoostingRegressor` | `GradientBoostingClassifier` |
| `XGBoost` | `XGBRegressor` | `XGBClassifier` |
| `LightGBM` | `LGBMRegressor` | `LGBMClassifier` |
| `SVM_RBF` | `SVR` | `SVC` |
| `KNN` | `KNeighborsRegressor` | `KNeighborsClassifier` |
| `MLP` | `MLPRegressor` | `MLPClassifier` |
| `Logistic Regression` | - | `LogisticRegression` |
| `Ridge Regression` | `Ridge` | - |
| `Lasso Regression` | `Lasso` | - |
| `PLS` | `PLSRegression` | - |

## 8. Hyperparameter tuning

Set `hyperparameter_tuning: true` to use randomized cross-validation search:

```yaml
hyperparameter_tuning: true
n_iter: 50
cv: 5
cv_strategy: scaffold-grouped
cv_random_seed: 42
n_jobs: -1

scoring_metrics:
  - r2
  - root_mean_squared_error
  - mean_absolute_error
refit_metric: root_mean_squared_error
```

`cv_strategy` controls the inner folds used to select hyperparameters:

- `cv` uses shuffled `KFold` for regression and shuffled,
  class-stratified folds for classification.
- `scaffold-grouped` groups molecules by their Bemis-Murcko scaffold. A
  scaffold and all molecules carrying it stay entirely within one fold, so
  the validation portion of a fold tests structural generalization. For
  molecular-property modelling, this is the more conservative recommended
  option.

These inner CV folds are created only from the outer training set. The outer
validation and test sets defined under `data_split` are not used for
hyperparameter fitting. Scaffold-grouped CV requires at least `cv` distinct
scaffolds in the training set. ChemFlow stores the aligned training SMILES in
the split cache; an older cache without them is regenerated from the source
dataset when possible.

For classification, scaffold grouping and approximate class stratification
are applied together. Exact class proportions cannot always be preserved
because complete scaffold groups may not be divided between folds.

The selected strategy, fold count, and number of training scaffolds are
recorded in the model summary and `*_model_info.json` files.

For backward-friendly configuration parsing, `kfold` is accepted as an alias
for `cv`, while `scaffold` and `scaffold_grouped` are accepted as aliases for
`scaffold-grouped`.

ChemFlow uses its built-in parameter grid when a model has no `param_grid`.
The default LightGBM search is deliberately regularized for molecular feature
sets: it tunes learning rate and tree count together with leaf count, maximum
depth, minimum child size, row sampling, column sampling, and L1/L2 penalties.
`subsample_freq` is set to `1`, so the sampled-row fractions are actually
active. LightGBM remains a tree-based model; descriptor scaling is harmless,
while the tree model keeps fingerprint bits intact except for constant bits.

The other built-in searches follow the same conservative policy:

These are robust starting ranges, not universally optimal values. Dataset
size, endpoint noise, class imbalance, and chemical diversity still determine
the selected model; compare models using identical outer splits and CV folds.

| Model family | Search emphasis |
| --- | --- |
| Random Forest / Extra Trees | Tree count, bounded or unrestricted depth, leaf size, feature subsampling, bootstrap, and class weighting for classification |
| Gradient Boosting | Learning-rate/tree-count tradeoff, shallow trees, leaf size, row and feature subsampling; squared-error or Huber loss for regression |
| XGBoost | Learning rate, depth, child weight, row/column sampling, split penalty, and L1/L2 regularization |
| SVM RBF | Log-scale `C` and `gamma`; `epsilon` for regression and optional balanced class weights for classification |
| KNN | Neighbor count, uniform/distance weighting, and Manhattan/Euclidean Minkowski distance; brute-force search avoids ineffective tree indices in high dimensions |
| MLP | Network size, activation, learning rate, and L2 regularization, with early stopping enabled |
| Logistic / Ridge / Lasso | Broad log-scale regularization strengths; Logistic also tests balanced class weights |
| PLS | Number of latent components and internal scaling |

KNN neighbor counts and PLS component counts are automatically limited to
values that can fit the smallest inner-CV training fold. Individual numerical
candidate failures are recorded as `NaN` rather than aborting the entire
search; the run still fails clearly if every candidate is invalid. If `n_iter`
is larger than the number of possible parameter combinations, ChemFlow now
evaluates each unique combination once.

To override it for one model:

```yaml
models:
  - model_name: Random Forest
    estimator: RandomForestRegressor
    param_grid:
      n_estimators: [200, 500, 1000]
      max_depth: [10, 20, null]
      min_samples_leaf: [1, 2, 4]
```

Model entries may also override the global `n_iter` and `cv`. This is useful
for exact RBF SVMs, whose time and memory requirements grow much faster with
the number of training compounds than tree models:

```yaml
models:
  - model_name: SVM_RBF
    estimator: SVR
    n_iter: 8
    cv: 3
    model_params:
      cache_size: 1024  # MB for each concurrently fitted SVR process
    param_grid:
      C: [0.1, 1.0, 10.0]
      gamma: [scale, 0.001, 0.01]
      epsilon: [0.05, 0.1, 0.2]
```

This example performs 24 fits instead of 250 while leaving the global search
settings unchanged for other models. `cache_size` can reduce repeated kernel
recomputation, but it is allocated per concurrent fit, so keep
`cache_size * n_jobs` within the Slurm memory request. For roughly 10,000 or
more compounds, prefer LightGBM/XGBoost or a linear/approximate-kernel model
unless RBF SVM is essential; reducing CV does not change the unfavorable
scaling of exact SVR.

`refit_metric` must also appear in `scoring_metrics`. Error metrics such as
RMSE and MAE are negated internally during cross-validation so that larger
scores remain better; the reported evaluation files contain the interpretable
metric values.

## 9. Prediction

Use the generated `*_model_package.pkl` file for prediction. It contains both
the trained estimator and the molecular feature configuration. Do not use the
raw `*_trained_model.pkl` or `*_best_model.pkl` file when you want ChemFlow to
featurize SMILES automatically.

Predict one molecule:

```bash
chemflow predict ml \
  --smiles "CCO" \
  --model ./path/to/random_forest_model_package.pkl \
  --task-name LogD_prediction \
  --output ./predictions/ethanol.csv
```

Predict a CSV or Parquet file:

```bash
chemflow predict ml \
  --input ./dataset/molecules.csv \
  --structure-column SMILES \
  --model ./path/to/random_forest_model_package.pkl \
  --task-name LogD_prediction \
  --output ./predictions/logd_predictions.csv
```

The input command also accepts `.smi`, `.smiles`, and `.txt` files. Output can
be `.csv`, `.parquet`, or `.pq`. Classification output includes the predicted
class and, when the estimator supports probabilities, class-probability and
confidence columns. Molecules that cannot be featurized are omitted from the
prediction output.

## 10. Training outputs

Every run writes status and configuration information under `workdir`,
including `status.json`, `training.log`, and `feature_config.json`.

With hyperparameter tuning, each model writes files such as:

```text
<workdir>/<model>_summary.json
<workdir>/<model>_cv_results.csv
<workdir>/<model>_test_predictions.csv
<workdir>/<model>_best_model.pkl
<workdir>/<model>_model_package.pkl
```

With regular training, outputs are grouped by model and seed:

```text
<workdir>/<model>/seed_<seed>/<model>_summary.json
<workdir>/<model>/seed_<seed>/<model>_test_predictions.csv
<workdir>/<model>/seed_<seed>/<model>_trained_model.pkl
<workdir>/<model>/seed_<seed>/<model>_model_package.pkl
```

The test-prediction CSV starts with `Molecule Name` and `SMILES`. ChemFlow
recognizes common name columns such as `Molecule Name`, `Compound Name`,
`Molecule ID`, `Compound ID`, or `Name`; if none exists, it creates sequential
test names. For regression, the remaining columns are `test_row`, `true_value`,
`predicted_value`, `residual` (`true_value - predicted_value`), and
`absolute_error`. Classification includes true and predicted labels plus one
probability column per class when the estimator supports `predict_proba`.

Use `status.json` and `training.log` first when diagnosing a failed run.

## 11. Compare molecular representations

After the HPC representation array finishes, compare the four representations
for one model with paired bootstrap confidence intervals:

```bash
python example/ml_training/compare_representations.py \
  --root "$TRAINING_DIR" \
  --model lightgbm \
  --model-label LightGBM \
  --n-bootstrap 2000
```

The script aligns prediction files by molecule name, SMILES, and test-row index,
then uses the same resampled molecules for every representation. All tasks and
metrics are placed in one grouped error-bar figure, saved as both PNG and PDF.
It also creates a multi-task pairwise significance heatmap; select the heatmap
metric with `--pairwise-metric` (default: `RMSE`). Green means the row
representation is better, pink means the column representation is better, and
white means the Holm-adjusted difference is not significant. Training outputs
are read from `{task}_ml/<representation>/<model>/`. It
also writes paired comparisons for every representation pair and metric. A
positive `Improvement_A_over_B` always favors representation A: error metrics
are sign-reversed because lower MAE/RMSE is better. The paired table includes
the 95% bootstrap interval, probability that A is better, two-sided bootstrap
p-value, and Holm-adjusted p-value. Change `--model` to another output folder
such as `random_forest`, `xgboost`, or `svm_rbf` to compare that model instead.
