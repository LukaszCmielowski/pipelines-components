# Autogluon Models Selection ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Build multiple AutoGluon models and select top performers.

This component trains multiple machine learning models using AutoGluon's ensembling approach (stacking and bagging) on
sampled training data, then evaluates them on test data to identify the top N performing models.

The component uses AutoGluon's TabularPredictor which automatically trains various model types (neural networks,
tree-based models, linear models, etc.) and combines them using stacking with multiple levels and bagging. After
training, models are evaluated on the test dataset and ranked by performance. The top N models are selected and their
names are returned for use in subsequent refitting stages.

This component is part of a two-stage training pipeline where models are first built and evaluated on sampled data (for
efficiency), then the best candidates are refitted on the full dataset for optimal performance.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_column` | `str` | `None` | The name of the target/label column in the training
and test datasets. This column will be used as the prediction target. |
| `problem_type` | `str` | `None` | The type of machine learning problem. Supported values
include "classification", "regression", or "time_series". This
determines the evaluation metrics and model types AutoGluon will use. |
| `top_n` | `int` | `None` | The number of top-performing models to select from the leaderboard.
Only the top N models will be returned and promoted to the refit stage.
Must be a positive integer. |
| `train_data_regression` | `dsl.Input[dsl.Dataset]` | `None` | A Dataset artifact containing the training data
in CSV format. This data is used to train the AutoGluon models.
The dataset should include the target_column and all feature columns. |
| `test_data_regression` | `dsl.Input[dsl.Dataset]` | `None` | A Dataset artifact containing the test data in
CSV format. This data is used to evaluate model performance and
generate the leaderboard. The dataset should match the schema of
the training data. |
| `model_artifact` | `dsl.Output[dsl.Model]` | `None` | Output Model artifact where the trained TabularPredictor
will be saved. The artifact metadata will contain a "top_models" key
with the list of selected model names. |

## Outputs 📤

| Name | Type | Description |
|------|------|-------------|
| Output | `NamedTuple('outputs', top_models=List[str])` | A NamedTuple with the following fields:
- top_models (List[str]): A list of model names (strings) representing
  the top N performing models selected from the leaderboard, ranked
  by performance on the test dataset. |

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
