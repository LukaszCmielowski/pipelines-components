# Autogluon Models Selection ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Build multiple AutoGluon models and select top performers.

This component trains multiple machine learning models using AutoGluon's ensembling approach (stacking and bagging) on
sampled training data, then evaluates them on test data to identify the top N performing models.

The component uses AutoGluon's TabularPredictor which automatically trains various model types (neural networks,
tree-based models, linear models, etc.) and combines them using stacking with multiple levels and bagging. After
training, models are evaluated on the test dataset and ranked by performance. The top N models are selected and their
names are returned for use in subsequent refitting stages. The predictor is saved under workspace_path.

This component is part of a two-stage training pipeline where models are first built and evaluated on sampled data (for
efficiency), then the best candidates are refitted on the full dataset for optimal performance.

## Inputs 📥

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `label_column` | `str` | `None` | The name of the target/label column in the training and test datasets. This column will be used as the prediction target. |
| `task_type` | `str` | `None` | The type of machine learning task. Supported values include "binary", "multiclass" (classification) or "regression". Determines the evaluation metrics and model types AutoGluon will use. |
| `top_n` | `int` | `None` | The number of top-performing models to select from the leaderboard. Only the top N models will be returned and promoted to the refit stage. Must be a positive integer. |
| `train_data` | `dsl.Input[dsl.Dataset]` | `None` | A Dataset artifact containing the training data in CSV format. Used to train the AutoGluon models. The dataset should include the label_column and all feature columns. |
| `test_data` | `dsl.Input[dsl.Dataset]` | `None` | A Dataset artifact containing the test data in CSV format. Used to evaluate model performance and generate the leaderboard. The dataset should match the schema of the training data. |
| `workspace_path` | `str` | `None` | Path (string) to the workspace directory where the trained TabularPredictor will be saved (under workspace_path / autogluon_predictor). This path is also returned as predictor_path for use by downstream components. |

## Outputs 📤

| Name | Type | Description |
| ------ | ------ | ------------- |
| Output | `NamedTuple('outputs', top_models=List[str], eval_metric=str, predictor_path=str, model_config=dict)` | NamedTuple with top_models (list of top N model names), eval_metric (metric name, e.g. accuracy or r2), and predictor_path (path to the saved TabularPredictor for downstream use such as autogluon_models_full_refit). |

## Metadata 🗂️

- **Name**: autogluon_models_selection
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.14.4
- **Tags**:
  - training
- **Last Verified**: 2026-01-22 10:30:08+00:00
- **Owners**:
  - Approvers:
    - Mateusz-Switala
  - Reviewers:
    - Mateusz-Switala
