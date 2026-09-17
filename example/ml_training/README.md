# Traditional machine-learning training and prediction

This tutorial shows how to train and use ChemFlow's traditional machine-learning
models for single-task regression and classification. Run all commands from the
ChemFlow repository root.

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
| `descriptor` | Nine RDKit physicochemical descriptors |

For fingerprint features, `fp_bits` sets the fingerprint length:

```yaml
featurization:
  features:
    - ecfp4
    - descriptor
  fp_bits: 2048
```

The saved ChemFlow model package includes the fitted feature preprocessing, so
prediction data should contain SMILES rather than manually generated features.

## 4. Configure data splitting

The supported split methods are:

- `random`: randomly assign molecules;
- `scaffold`: keep each Bemis-Murcko scaffold in only one split;
- `butina`: group similar molecules using Butina clustering;
- `cluster`: group molecules using fingerprint clustering;
- `splitted`: read existing split labels from a dataset column.

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
  split_method: splitted
  split_column: split
  save_split_data: true
  save_dir: ./example/ml_training/my_run/split_data
  split_name: logd_predefined_v1
```

Accepted labels include `train`, `training`, `val`, `valid`, `validation`,
`dev`, `test`, and `testing`. A test split is required; validation is optional.

When `save_split_data` is enabled, ChemFlow may reuse the cached NPZ split on
later runs. If you change the input data, features, or split settings, use a new
`split_name` or remove the old cached split file.

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
  features: [ecfp4, descriptor]
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
n_jobs: -1

scoring_metrics:
  - r2
  - root_mean_squared_error
  - mean_absolute_error
refit_metric: root_mean_squared_error
```

ChemFlow uses its built-in parameter grid when a model has no `param_grid`.
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
<workdir>/<model>_best_model.pkl
<workdir>/<model>_model_package.pkl
```

With regular training, outputs are grouped by model and seed:

```text
<workdir>/<model>/seed_<seed>/<model>_summary.json
<workdir>/<model>/seed_<seed>/<model>_trained_model.pkl
<workdir>/<model>/seed_<seed>/<model>_model_package.pkl
```

Use `status.json` and `training.log` first when diagnosing a failed run.

