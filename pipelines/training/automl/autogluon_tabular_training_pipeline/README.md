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
| `train_data_secret_name` | `str` | — | Kubernetes secret with S3 credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION). |
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
    │       └── html_artifact                        # HTML leaderboard (model names + metrics)
    ├── notebook-generation/
    │   └── <task_id>/
    |       └── notebook_artifact/
    |           └── automl_predictor_notebook.ipynb  # jupyter notebook for interaction with TabularPredictor
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
