# Autogluon Tabular Training Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

AutoGluon Tabular Training Pipeline.

This pipeline implements an efficient two-stage training approach for AutoGluon tabular models that balances
computational cost with model quality. The pipeline automates the complete machine learning workflow from data loading
to final model evaluation.

**Pipeline Stages:**

1. **Data Loading**: Loads tabular data from an S3-compatible object storage bucket using AWS credentials configured via
Kubernetes secrets. The component produces both a tabular_data artifact (for splitting) and a full_dataset artifact (for
model refitting).

2. **Data Splitting**: Splits the loaded tabular data into training and test sets using a configurable test size
(default: 20% test, 80% train). The split is performed on the tabular_data artifact to create separate train and test
datasets for model training and evaluation.

3. **Model Selection**: Trains multiple AutoGluon models on the training data using AutoGluon's ensembling approach
(stacking with 3 levels and bagging with 2 folds). The component automatically trains various model types including
neural networks, tree-based models (XGBoost, LightGBM, CatBoost), and linear models. All models are evaluated on the
test set and ranked by performance. The top N models are selected for the refitting stage.

4. **Model Refitting**: Refits each of the top N selected models on the full dataset (the complete original dataset from
the data loader). This stage runs in parallel (with parallelism of 2) to efficiently retrain multiple models. Each
refitted model is saved with a "_FULL" suffix and optimized for deployment by removing unnecessary models and files.

5. **Leaderboard Evaluation**: Aggregates evaluation results from all refitted model artifacts (each refit component
writes metrics to model_artifact.path / model_name_FULL / metrics). The leaderboard component reads these pre-computed
metrics and generates an HTML-formatted leaderboard ranking models by their performance metrics for comparison and
selection.

**Two-Stage Training Benefits:**

- **Efficient Exploration**: Initial model training uses the split training data with efficient ensembling rather than
expensive hyperparameter optimization - **Optimal Performance**: Final models are refitted on the complete original
dataset for maximum performance - **Parallel Efficiency**: Top models are refitted in parallel to minimize total
pipeline execution time - **Production-Ready**: Refitted models are AutoGluon Predictors optimized and ready for
deployment

**AutoGluon Ensembling Approach:**

The pipeline leverages AutoGluon's unique ensembling strategy that combines multiple model types using stacking and
bagging rather than traditional hyperparameter optimization. This approach is more efficient and typically produces
better results for tabular data by automatically: - Training diverse model families - Combining predictions using
multi-level stacking - Using bootstrap aggregation (bagging) for robustness - Selecting optimal ensemble configurations

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data_secret_name` | `str` | `None` | Kubernetes secret name with S3 credentials; required keys: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION. |
| `train_data_bucket_name` | `str` | `None` | Name of the S3-compatible bucket containing the tabular data file; accessible via train_data_secret_name credentials. |
| `train_data_file_key` | `str` | `None` | Key (path) of the data file in the S3 bucket; file must be CSV with feature columns and target column. |
| `label_column` | `str` | `None` | Name of the target/label column in the dataset; must exist and is used as the prediction target. |
| `task_type` | `str` | `None` | Machine learning task: "binary" or "multiclass" (classification) or "regression"; determines evaluation metrics and model types. |
| `top_n` | `int` | `3` | Number of top-performing models to select and refit (default: 3); must be positive; higher values increase execution time. |

## Metadata 🗂️

- **Name**: autogluon_tabular_training_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - pipeline
- **Last Verified**: 2026-01-22 11:44:56+00:00
- **Owners**:
  - Approvers:
    - Mateusz-Switala
  - Reviewers:
    - Mateusz-Switala
