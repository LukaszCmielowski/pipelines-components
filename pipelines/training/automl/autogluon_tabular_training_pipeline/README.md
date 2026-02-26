# Autogluon Tabular Training Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

AutoGluon Tabular Training Pipeline.

This pipeline implements an efficient two-stage training approach for AutoGluon tabular models that balances
computational cost with model quality. The pipeline automates the complete machine learning workflow from data loading
to final model evaluation.

**Pipeline Stages:**

1. **Data Loading**: Loads tabular data from an S3-compatible object storage bucket using AWS credentials configured via
   Kubernetes secrets. Produces a `full_dataset` artifact used for splitting and for refitting.

2. **Data Splitting**: Splits the loaded dataset into training and test sets using a configurable test size (default:
   20% test, 80% train). Outputs `sampled_train_dataset` and `sampled_test_dataset` for model selection.

3. **Model Selection**: Trains multiple AutoGluon models on the training data using AutoGluon's ensembling approach
   (stacking with 3 levels and bagging with 2 folds). The component automatically trains various model types including
   neural networks, tree-based models (XGBoost, LightGBM, CatBoost), and linear models. All models are evaluated on the
   test set and ranked by performance. The top N models are selected; the component returns `top_models`, `eval_metric`,
   and `predictor_path` (workspace path where the predictor is saved) for the refitting stage.

4. **Model Refitting**: Refits each of the top N selected models on the full training dataset. Each refit task receives
   `predictor_path` from the selection stage (not a model artifact). This stage runs in parallel (parallelism=2) to
   efficiently retrain multiple models. Each refitted model is saved with a "_FULL" suffix and metrics are written under
   `model_artifact.path / model_name_FULL / metrics`. Outputs are collected for the leaderboard.

5. **Leaderboard Evaluation**: Aggregates evaluation results from all refitted model artifacts by reading pre-computed
   metrics from each model's path (`model.path / model_name / metrics / metrics.json`). Generates an HTML-formatted
   leaderboard ranking models by their performance metrics for comparison and selection.

**Two-Stage Training Benefits:**

- **Efficient Exploration**: Initial model training uses the split training data with efficient ensembling rather than
  expensive hyperparameter optimization.
- **Optimal Performance**: Final models are refitted on the complete original dataset for maximum performance.
- **Parallel Efficiency**: Top models are refitted in parallel to minimize total pipeline execution time.
- **Production-Ready**: Refitted models are AutoGluon Predictors optimized and ready for deployment.

**AutoGluon Ensembling Approach:**

The pipeline leverages AutoGluon's unique ensembling strategy that combines multiple model types using stacking and
bagging rather than traditional hyperparameter optimization. This approach is more efficient and typically produces
better results for tabular data by automatically:

- Training diverse model families (neural networks, tree-based, linear)
- Combining predictions using multi-level stacking
- Using bootstrap aggregation (bagging) for robustness
- Selecting optimal ensemble configurations

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `train_data_secret_name` | `str` | — | Kubernetes secret with S3 credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION). |
| `train_data_bucket_name` | `str` | — | S3-compatible bucket containing the tabular data file. |
| `train_data_file_key` | `str` | — | Key (path) of the data file in the bucket; CSV with feature and label columns. |
| `label_column` | `str` | — | Target/label column name for model training. Must exist in the dataset. |
| `task_type` | `str` | — | Task type: `"binary"` or `"multiclass"` (classification), `"regression"` (continuous). |
| `top_n` | `int` | `3` | Number of top models to select and refit (positive integer). |

## Outputs 📤

| Output | Type | Description |
| ------ | ---- | ----------- |
| `leaderboard_evaluation_task.html_artifact` | `dsl.Output[dsl.HTML]` | HTML leaderboard ranking refitted models by eval metric (e.g. accuracy, RMSE). |
| `refit_full_task.model_artifact` (per top-N) | `dsl.Output[dsl.Model]` | Refitted TabularPredictor per top-N model (full dataset, "_FULL" suffix); N artifacts. |

### Files stored in user storage

Pipeline outputs are written to the artifact store (S3-compatible storage configured for Kubeflow Pipelines). The layout below matches what components write and what downstream consumers expect when loading the leaderboard or a refitted model.

```text
<pipeline_name>/
└── <run_id>/
    ├── leaderboard-evaluation/
    │   └── <task_id>/
    │       └── html_artifact                     # HTML leaderboard (model names + metrics)
    ├── autogluon-models-full-refit/
    │   └── <task_id>/                           # one per top-N model
    │       └── model_artifact/
    │           └── <ModelName>_FULL/            # e.g. LightGBM_BAG_L1_FULL
    │               ├── predictor/               # AutoGluon TabularPredictor files
    │               └── metrics/
    │                   ├── metrics.json         # model evaluation metrics (eval_metric, etc.)
    │                   ├── feature_importance.json
    │                   └── confusion_matrix.json  # for classification tasks only
    ├── automl-data-loader/
    │   └── <task_id>/
    │       └── full_dataset/                    # (optional for debugging) tabular CSV used during run
    ╰── tabular-train-test-split/
        └── <task_id>/
            ├── sampled_train_dataset/           # split of full dataset - used in selection stage
            └── sampled_test_dataset/
```

- **leaderboard-evaluation**: Contains the HTML leaderboard artifact summarizing all model results.
- **autogluon-models-full-refit**: Each top-N model refit task writes its model artifact here, under `<ModelName>_FULL`, including the saved TabularPredictor and associated metrics.
- **automl-data-loader / tabular-train-test-split**: Store original and split datasets for pipeline traceability; useful for debugging or further processing, but not used in deployment.

_Note_: There is one `autogluon-models-full-refit/<task_id>/model_artifact/<ModelName>_FULL` directory for each selected top-N model (parallel execution). Each contains an independently saved and refitted AutoGluon predictor.

For loading:

- Load a refitted model for deployment or notebook exploration using `TabularPredictor.load(<.../model_artifact/<ModelName>_FULL>)`
- Model metrics and feature importances are always at `metrics/` under each model directory.
- The leaderboard HTML is at `leaderboard-evaluation/<task_id>/html_artifact`.

## Usage Examples 💡

### Basic usage (regression)

Run the full two-stage pipeline with data from S3; credentials are provided via a Kubernetes secret:

```python
from kfp import dsl
from kfp_components.pipelines.training.automl.autogluon_tabular_training_pipeline import (
    autogluon_tabular_training_pipeline,
)

# Compile and run the pipeline
pipeline = autogluon_tabular_training_pipeline(
    train_data_secret_name="my-s3-secret",
    train_data_bucket_name="my-data-bucket",
    train_data_file_key="datasets/housing_prices.csv",
    label_column="price",
    task_type="regression",
    top_n=3,
)
```

### Classification (binary or multiclass)

```python
pipeline = autogluon_tabular_training_pipeline(
    train_data_secret_name="my-s3-secret",
    train_data_bucket_name="my-ml-bucket",
    train_data_file_key="data/train.csv",
    label_column="target",
    task_type="multiclass",
    top_n=5,
)
```

### Compile to YAML

```python
from kfp.compiler import Compiler
from kfp_components.pipelines.training.automl.autogluon_tabular_training_pipeline import (
    autogluon_tabular_training_pipeline,
)

Compiler().compile(
    autogluon_tabular_training_pipeline,
    package_path="autogluon_tabular_training_pipeline.yaml",
)
```

### Run pipeline using KFP SDK

Compile and submit a run using the KFP client. Configure the client for your cluster (e.g. `host`, or in-cluster auth). Pipeline parameters are passed as `arguments`:

```python
import kfp
from kfp_components.pipelines.training.automl.autogluon_tabular_training_pipeline import (
    autogluon_tabular_training_pipeline,
)

# Create client (customize host for your KFP instance)
client = kfp.Client(host="https://your-kfp-host/pipeline")

# Run the pipeline with parameters
run = client.create_run_from_pipeline_func(
    autogluon_tabular_training_pipeline,
    arguments={
        "train_data_secret_name": "my-s3-secret",
        "train_data_bucket_name": "my-data-bucket",
        "train_data_file_key": "datasets/housing_prices.csv",
        "label_column": "price",
        "task_type": "regression",
        "top_n": 3,
    },
)
print(f"Submitted run: {run.run_id}")
```

To run from a compiled YAML instead:

```python
from kfp.compiler import Compiler

Compiler().compile(
    autogluon_tabular_training_pipeline,
    package_path="autogluon_tabular_training_pipeline.yaml",
)
run = client.create_run_from_pipeline_package(
    "autogluon_tabular_training_pipeline.yaml",
    arguments={
        "train_data_secret_name": "my-s3-secret",
        "train_data_bucket_name": "my-data-bucket",
        "train_data_file_key": "datasets/housing_prices.csv",
        "label_column": "price",
        "task_type": "regression",
        "top_n": 3,
    },
)
```

## Metadata 🗂️

- **Name**: autogluon_tabular_training_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.14.4
    - Name: Kubernetes, Version: ">=1.28.0"
- **Tags**:
  - training
  - pipeline
- **Last Verified**: 2026-01-22 11:44:56+00:00
- **Owners**:
  - Approvers: None
  - Reviewers: None
