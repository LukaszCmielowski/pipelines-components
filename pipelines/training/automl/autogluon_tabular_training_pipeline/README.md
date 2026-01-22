# AutoGluon Tabular Training Pipeline 🤖

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Automated machine learning pipeline for building and optimizing models for tabular data.

The AutoGluon Tabular Training Pipeline is designed for traditional tabular machine learning tasks:
**classification** (binary and multiclass) and **regression**. It leverages Kubeflow Pipelines to
orchestrate the complete model training workflow, using the AutoGluon library to automatically
build, evaluate, and select optimal models.

The pipeline integrates with Model Registry for model versioning and KServe for model deployment,
producing trained predictors that can be deployed for production machine learning applications.

## Pipeline Workflow 🔄

The optimization process involves the following stages:

1. **Data Loading**: Loads tabular data from data sources (S3, local filesystem)
2. **Data Splitting & Sampling**: Splits data into train/test sets and samples a subset for
   initial model building
3. **Model Building & Selection**: Builds multiple pipeline models using sampled data and
   AutoGluon, promoting best performers
4. **Model Refit**: Refits best candidate models on full training data using AutoGluon
5. **Leaderboard Evaluation**: Evaluates fully trained models and generates a leaderboard
   ranked by performance
6. **Model Registry** (optional): Registers the best model with metadata for deployment
7. **Model Deployment** (optional): Deploys the model using KServe with AutoGluon runtime

## Input Parameters 📥

The pipeline parameters are organized into the following logical groups:

> 📘 **Note:** This documentation uses Kubeflow Pipelines v2 structured types (`Dict`, `List`)
> for complex parameters.

### 1. Experiment Metadata

**Required Parameters:**

- `name: str` - Name of the AutoML experiment run (e.g., "AutoML run")

**Optional Parameters:**

- `description: str` - Description of the experiment (e.g., "Customer churn prediction")

### 2. Input Data Sources

**Required Parameter:**

- `input_data_reference: Dict` - Dictionary defining tabular data source:
  - `connection_id: str` - Connection ID for the data source (e.g., S3 connection ID)
  - `bucket: str` - Bucket name containing the data
  - `path: str` - Path within the bucket/filesystem to the data file or folder

**Optional Parameter:**

- `test_data_reference: Dict` - Dictionary defining test data source (external test data used
  for leaderboard evaluation after training is completed):
  - `connection_id: str` - Connection ID for the test data source
  - `bucket: str` - Bucket name containing the test data file
  - `path: str` - Path within the bucket/filesystem to the test data file (csv)

### 3. Infrastructure Configuration

#### Output Results Storage

**Required Parameter:**

- `results_reference: Dict` - Dictionary defining results storage location:
  - `connection_id: str` - Connection ID for the results storage (e.g., S3 connection ID)
  - `bucket: str` - Bucket name for storing results
  - `path: str` - Path where experiment results will be stored (e.g., "automl/results")

#### MLFlow Integration (Experiment Tracking)

**Optional Parameter:**

- `mlflow_config: Dict` - Dictionary defining MLFlow configuration for experiment tracking:
  - `tracking_uri: str` - MLFlow tracking server URI (e.g., "http://mlflow-server:5000")
  - `experiment_name: str` - MLFlow experiment name (default: uses pipeline `name` parameter)
  - `enabled: bool` - Enable/disable MLFlow tracking (default: `True` if `mlflow_config` is
    provided)

When enabled, AutoML will automatically log:

- Experiment metadata (name, description, run parameters)
- Model training metrics (accuracy, precision, recall, F1, ROC-AUC for classification;
  R², RMSE, MAE for regression)
- Model configuration (task type, label column, preset, eval_metric, time_limit)
- Data preparation settings (sampling method, split configuration)
- Leaderboard rankings and best model information
- Model artifacts references (Predictor models, summary reports)
- Execution timestamps and duration

> 💡 **Note:** If `mlflow_config` is not provided, MLFlow tracking will be disabled. To use
> MLFlow, ensure the MLFlow server is accessible from the pipeline execution environment.

### 4. Data Prep Configuration

**Optional Parameters:**

**Data Sampling:**

- `sampling_config: Dict` - Dictionary defining sampling technique:
  - `n_samples: int` - The number of samples to use for initial model building
    (optional, default: 500)
  - `sampling_method: str` - Sampling method (optional):
    - `"random"` - Random sampling for general use cases
    - `"stratified"` - Stratified sampling for classification tasks to maintain class
      distribution
    - `"truncate"` - Sampling last n records for time-series forecasting tasks

**Data Splitting:**

- `split_config: Dict` - Dictionary defining train/test split configuration:
  - `test_size: float` - Proportion of dataset to include in test split (default: 0.2)

### 5. Model Selection Configuration

**Required Parameters:**

- `task_type: str` - Type of ML task (required):
  - `"regression"` - Regression tasks (default metric: `"R2"`)
  - `"classification"` - Binary & multiclass classification tasks (default metric: `"accuracy"`)
- `label_column: str` - Name of the label/target column in the dataset (required)

**Optional Parameters:**

- `selection_config: Dict` - Dictionary defining model selection configuration:
  - `top_n: int` - Number of top models to promote to refit stage (default: 3)
  - `eval_metric: str` - Evaluation metric (default: AutoGluon chooses based on task_type):
    - Classification: `"accuracy"`, `"f1"`, `"roc_auc"`, `"log_loss"`, etc.
    - Regression: `"R2"`, `"RMSE"`, `"MAE"`, etc.
  - `time_limit: int` - Time limit in seconds for model building (default: 3600)
  - `presets: str` - AutoGluon preset (default: `"best_quality"`):
    - `"best_quality"` - Best model quality, longer training time
    - `"high_quality"` - High quality with faster training
    - `"good_quality_faster_inference"` - Good quality optimized for inference speed
    - `"optimize_for_deployment"` - Optimized for deployment with smaller model size

### 6. Deployment Configuration

**Optional Parameters:**

- `auto_register: bool` - Enable automatic model registration with Model Registry
  (default: `False`)
- `auto_deploy: bool` - Enable automatic model deployment with KServe (default: `False`)

## Required Parameters ✅

The following parameters are required to run the pipeline:

- `name: str` - Experiment name
- `input_data_reference: Dict` - Tabular data source
- `results_reference: Dict` - Results storage location
- `task_type: str` - Type of ML task (`"classification"` or `"regression"`)
- `label_column: str` - Name of the label/target column

## Usage Examples 💡

### Python SDK Example

```python
from kfp import dsl
from kfp_components.pipelines.training.automl.autogluon_tabular_training_pipeline import (
    autogluon_tabular_training_pipeline,
)

# Define data references
input_data_reference = {
    "connection_id": "s3-data-connection",
    "bucket": "my-ml-data-bucket",
    "path": "tabular_data/train.csv"
}

results_reference = {
    "connection_id": "s3-automl-results-connection",
    "bucket": "results",
    "path": "automl/"
}

# Optional MLFlow configuration
mlflow_config = {
    "tracking_uri": "http://mlflow-server.redhat-ods-applications.svc.cluster.local:5000",
    "experiment_name": "AutoML Experiments",
    "enabled": True
}

# Optional sampling and split configuration
sampling_config = {
    "n_samples": 500,
    "sampling_method": "stratified"
}

split_config = {
    "test_size": 0.2
}

# Optional model selection configuration
selection_config = {
    "top_n": 3,
    "eval_metric": "accuracy",
    "time_limit": 3600,
    "presets": "best_quality"
}

# Create pipeline run
run = client.create_run_from_pipeline_func(
    autogluon_tabular_training_pipeline,
    arguments={
        "name": "AutoML Classification Experiment",
        "description": "Customer churn prediction",
        "input_data_reference": input_data_reference,
        "results_reference": results_reference,
        "mlflow_config": mlflow_config,
        "sampling_config": sampling_config,
        "split_config": split_config,
        "task_type": "classification",
        "label_column": "target",
        "selection_config": selection_config,
        "auto_register": True,
        "auto_deploy": False
    }
)
```

### Minimal Example

```python
run = client.create_run_from_pipeline_func(
    autogluon_tabular_training_pipeline,
    arguments={
        "name": "AutoML Run",
        "input_data_reference": {
            "connection_id": "s3-data-connection",
            "bucket": "my-ml-data-bucket",
            "path": "tabular_data/train.csv"
        },
        "results_reference": {
            "connection_id": "s3-automl-results-connection",
            "bucket": "results",
            "path": "automl/"
        },
        "task_type": "classification",
        "label_column": "target"
    }
)
```

## Components Used 🔧

This pipeline orchestrates the following AutoML components:

1. **[Data Loader](../components/training/automl/data-processing/data-loader/README.md)** -
   Reads tabular data from data sources (S3, local filesystem)

2. **[Train Test Split](../components/training/automl/data-processing/train-test-split/README.md)** -
   Splits data into train/test sets and performs sampling

3. **[Model Building Selection](../components/training/automl/model-training/model-building-selection/README.md)** -
   Builds multiple models using sampled data and selects top performers

4. **[Model Refit](../components/training/automl/model-training/model-refit/README.md)** -
   Refits best candidate models on full training dataset

5. **[Leaderboard Evaluation](../components/training/automl/model-evaluation/leaderboard-evaluation/README.md)** -
   Evaluates models and generates leaderboard with metrics

6. **[Model Registry](../components/training/automl/model-deployment/model-registry/README.md)** -
   (Optional) Registers best model with Model Registry

7. **[KServe Deployment](../components/training/automl/model-deployment/kserve-deployment/README.md)** -
   (Optional) Deploys model using KServe with AutoGluon runtime

## Artifacts 📦

For each pipeline run, AutoML generates:

- **Model Artifact(s)**: Trained AutoGluon Predictor models with associated metadata
- **AutoML Run Artifact**: Run status properties and URI to log file with messages
- **Leaderboard Artifact**: Leaderboard with models and eval scores
- **AutoML Experiment Summary Markdown Artifact**: Experiment run report

> 📘 **Details on artifacts:** See artifact documentation for comprehensive information about
> artifact structure, naming conventions, and sample artifacts.

## Optimization Engine: AutoGluon 🚀

The pipeline uses [AutoGluon](https://github.com/autogluon/autogluon), an automated machine
learning library that provides an automated approach to building and optimizing machine learning
models for tabular data.

AutoGluon automatically:

- Handles data preprocessing and feature engineering
- Trains multiple model types (neural networks, tree-based models, etc.)
- Creates ensembles using stacking and bagging techniques
- Selects optimal models based on performance metrics
- Provides production-ready predictors

AutoGluon uses an **ensembling approach** (stacking/bagging) rather than traditional
hyperparameter optimization to achieve high performance.

## Supported Features ✨

### Data Configuration

- **Data Type**: Tabular data (CSV, Parquet, etc.)
- **Data Sources**: S3 (Amazon S3), Local filesystem (FS)
- **Supported Task Types**: Classification (Multiclass, Binary), Regression

### Infrastructure Components

- **Model Training**: AutoGluon library
- **Distributed Computing**: Kubeflow Katib (to be explored for distributed training)
- **Experiment Tracking**: MLFlow (optional) - For experiment tracking, metrics logging, and
  artifact management
- **Model Registry**: RHOAI Model Registry (optional)
- **Model Serving**: KServe with AutoGluon runtime (optional, custom runtime)

### Model Types

- **Ensembling**: Stacking and bagging approaches
- **Model Families**: Neural networks, tree-based models (XGBoost, LightGBM, CatBoost), linear
  models, and more
- **Hyperparameter Optimization**: Not recommended by AutoGluon; ensembling approach preferred
  (HPO to be explored post-MVP)

## Metadata 🗂️

- **Name**: autogluon_tabular_training_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: AutoGluon, Version: >=1.0.0
    - Name: RHOAI Connections API, Version: >=1.0.0
    - Name: RHOAI Model Registry API, Version: >=1.0.0 (optional)
    - Name: KServe, Version: >=0.11.0 (optional)
    - Name: MLFlow, Version: >=2.0.0 (optional)
- **Tags**:
  - training
  - pipeline
  - automl
  - classification
  - regression
  - autogluon
- **Last Verified**: 2026-01-22 11:44:56+00:00

## Additional Resources 📚

- **AutoGluon Documentation**: [AutoGluon GitHub](https://github.com/autogluon/autogluon)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
