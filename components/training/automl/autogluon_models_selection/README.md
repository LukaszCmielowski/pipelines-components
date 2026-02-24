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
|-----------|------|---------|-------------|
| `label_column` | `str` | `None` | Name of the target/label column in the training and test datasets; used as the prediction target. |
| `task_type` | `str` | `None` | Machine learning task type: "binary", "multiclass" (classification) or "regression"; determines metrics and model types. |
| `top_n` | `int` | `None` | Number of top-performing models to select from the leaderboard; must be a positive integer. |
| `train_data` | `dsl.Input[dsl.Dataset]` | `None` | Dataset artifact (CSV) with training data; must include label_column and all feature columns. |
| `test_data` | `dsl.Input[dsl.Dataset]` | `None` | Dataset artifact (CSV) with test data for evaluation and leaderboard; schema should match training data. |
| `workspace_path` | `str` | `None` | Path to the workspace directory where the TabularPredictor is saved (workspace_path / autogluon_predictor); returned as predictor_path. |

## Outputs 📤

| Name | Type | Description |
|------|------|-------------|
| Output | `NamedTuple('outputs', top_models=List[str], eval_metric=str, predictor_path=str, model_config=dict)` | A NamedTuple with: top_models (List[str]): top N model names from leaderboard; eval_metric (str): metric used by TabularPredictor (e.g. "accuracy", "r2"); predictor_path (str): path to saved predictor (workspace_path / autogluon_predictor); model_config (dict): preset, eval metric, and time limit. |

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
