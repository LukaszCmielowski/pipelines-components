# Autogluon Models Selection ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Build multiple AutoGluon models and select top performers.

This component trains multiple machine learning models using AutoGluon's ensembling approach (stacking and bagging) on
sampled training data, then evaluates them on test data to identify the top N performing models.

The component uses AutoGluon's TabularPredictor which automatically trains various model types (neural networks,
tree-based models, linear models, etc.) and combines them using stacking with multiple levels and bagging. After
training, models are evaluated on the test dataset and ranked by performance. The top N models are selected and their
names are returned (with eval_metric and predictor_path) for use in subsequent refitting stages. The predictor is
saved under the provided workspace_path (workspace_path / autogluon_predictor).

This component is part of a two-stage training pipeline where models are first built and evaluated on sampled data (for
efficiency), then the best candidates are refitted on the full dataset for optimal performance.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `label_column` | `str` | — | Name of the target/label column used as the prediction target. |
| `task_type` | `str` | — | Task type: `"binary"`, `"multiclass"` (classification) or `"regression"`. Determines metrics and model types. |
| `top_n` | `int` | — | Number of top-performing models to select from the leaderboard (positive integer). |
| `train_data` | `dsl.Input[dsl.Dataset]` | — | Dataset artifact (CSV) for training; must include label_column and feature columns. |
| `test_data` | `dsl.Input[dsl.Dataset]` | — | Dataset artifact (CSV) for evaluation; schema should match training data. |
| `workspace_path` | `str` | — | Workspace path; predictor saved under `workspace_path / autogluon_predictor`, returned as predictor_path. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| `top_models` | `List[str]` | Top N model names from the leaderboard, ranked by test performance. |
| `eval_metric` | `str` | Metric used by TabularPredictor (e.g. "accuracy", "r2"), from task_type. |
| `predictor_path` | `str` | Path to saved TabularPredictor (`workspace_path / autogluon_predictor`) for downstream use. |

## Metadata 🗂️

- **Name**: autogluon_models_selection
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - training
- **Last Verified**: 2026-01-22 10:30:08+00:00
- **Owners**:
  - Approvers: None
  - Reviewers: None
