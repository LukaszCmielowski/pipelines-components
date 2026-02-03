# Autogluon Tabular Training Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

AutoGluon Tabular Training Pipeline.

This pipeline implements an efficient two-stage training approach for AutoGluon tabular models that balances
computational cost with model quality. The pipeline automates the complete machine learning workflow from data loading
to final model evaluation.

**Pipeline Stages:**

1. **Data Loading**: Loads tabular data from an S3-compatible object storage bucket using AWS credentials configured via
Kubernetes secrets.

2. **Data Splitting**: Splits the loaded dataset into training and test sets using a configurable test size (default:
20% test, 80% train).

3. **Model Selection**: Trains multiple AutoGluon models on the training data using AutoGluon's ensembling approach
(stacking with 3 levels and bagging with 2 folds). The component automatically trains various model types including
neural networks, tree-based models (XGBoost, LightGBM, CatBoost), and linear models. All models are evaluated on the
test set and ranked by performance. The top N models are selected for the refitting stage.

4. **Model Refitting**: Refits each of the top N selected models on the full training dataset. This stage runs in
parallel (with configurable parallelism) to efficiently retrain multiple models. Each refitted model is saved with a
"_FULL" suffix and optimized for deployment by removing unnecessary models and files.

5. **Leaderboard Evaluation**: Evaluates all refitted models on the full dataset and generates a markdown-formatted
leaderboard ranking models by their performance metrics. The leaderboard provides comprehensive evaluation results for
model comparison and selection.

**Two-Stage Training Benefits:**

- **Reduced Computational Cost**: Initial model exploration happens on the full dataset but with efficient ensembling
rather than expensive hyperparameter optimization - **Maintained Quality**: Final models are refitted on the complete
training dataset for optimal performance - **Parallel Efficiency**: Top models are refitted in parallel to minimize
total pipeline execution time - **Production-Ready**: Refitted models are AutoGluon Predictors optimized and ready for
deployment

**AutoGluon Ensembling Approach:**

The pipeline leverages AutoGluon's unique ensembling strategy that combines multiple model types using stacking and
bagging rather than traditional hyperparameter optimization. This approach is more efficient and typically produces
better results for tabular data by automatically: - Training diverse model families (neural networks, tree-based,
linear) - Combining predictions using multi-level stacking - Using bootstrap aggregation (bagging) for robustness -
Selecting optimal ensemble configurations

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data_secret_name` | `str` | — | The Kubernetes secret name with S3-compatible credentials for data access. Required keys: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `AWS_REGION`. |
| `train_data_bucket_name` | `str` | — | The name of the S3-compatible bucket containing the tabular data file. The bucket must be accessible using the credentials from `secret_name`. |
| `train_data_file_key` | `str` | — | The key (path) of the data file within the S3 bucket. The file should be in CSV format and contain both feature columns and the label column. |
| `label_column` | `str` | — | The name of the target/label column in the dataset. This column will be used as the prediction target for model training. Must exist in the loaded dataset. |
| `task_type` | `str` | — | The type of machine learning task. Supported values: `"binary"` or `"multiclass"` (classification), `"regression"` (continuous targets). Determines evaluation metrics and model types AutoGluon uses. |
| `top_n` | `int` | `3` | The number of top-performing models to select and refit. Must be a positive integer. Only the top N models from the initial stage are refitted on the full dataset. Higher values increase execution time but provide more final options. |

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `leaderboard_evaluation_task.html_artifact` | `dsl.Output[dsl.HTML]` | HTML-formatted leaderboard ranking all refitted models by the evaluation metric (e.g., accuracy for classification, RMSE for regression). Use this artifact to compare model performance and select the best model for deployment. |
| `refit_full_task.model_artifact` (per top-N model) | `dsl.Output[dsl.Model]` | Refitted AutoGluon TabularPredictor for each of the top N models. Each model is trained on the full dataset, optimized for deployment, and saved with a "_FULL" suffix. There are N such artifacts (one per `top_n`). Use these artifacts to deploy the selected model(s) for inference. |

### Files stored in user storage

Pipeline outputs are written to the artifact store (S3-compatible storage configured for Kubeflow Pipelines). The layout below matches what components write and what downstream consumers expect when loading the leaderboard or a refitted model.

```
<pipeline_name>/
└── <run_id>/
    ├── leaderboard-evaluation/
    │   └── <task_id>/
    │       └── html_artifact                        # HTML leaderboard (model names + metrics)
    │
    └── autogluon-models-full-refit/
        └── <task_id>/                               # one per top-N model
            └── model_artifact/
                └── <ModelName>_FULL/                # e.g. LightGBM_BAG_L1_FULL
                    ├── metrics/
                    │   ├── metrics.json             # evaluation metrics (eval_metric, etc.)
                    │   ├── feature_importance.json
                    │   └── confusion_matrix.json    # classification only
                    └── [AutoGluon predictor files]  # TabularPredictor serialization
```

- **Leaderboard**: Single HTML file
- **Model artifact**: Under each refit task, `model_artifact/<ModelName>_FULL` is the predictor root; load with `TabularPredictor.load(path_to_that_folder)`. The `metrics/` subfolder holds evaluation and feature-importance JSON written by the pipeline.

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
