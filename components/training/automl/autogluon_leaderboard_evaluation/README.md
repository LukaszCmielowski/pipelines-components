# Leaderboard Evaluation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate multiple AutoGluon models and generate a leaderboard.

This component evaluates a list of trained AutoGluon TabularPredictor models on a full dataset and generates a
markdown-formatted leaderboard ranking the models by their performance metrics. Each model is loaded, evaluated on the
provided dataset, and the results are compiled into a sorted leaderboard table.

The leaderboard is sorted by root mean squared error (RMSE) in descending order, making it easy to identify the
best-performing models. The output is written as a markdown table that can be used for reporting and model selection
decisions.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `dsl.Input[List[dsl.Model]]` | `None` | A list of Model artifacts containing trained AutoGluon
TabularPredictor models to evaluate. Each model should have
metadata containing a "model_name" field. |
| `full_dataset` | `dsl.Input[dsl.Dataset]` | `None` | A Dataset artifact containing the evaluation dataset
on which all models will be evaluated. The dataset should be
compatible with the models' training data format. |
| `markdown_artifact` | `dsl.Output[dsl.Markdown]` | `None` | Output artifact where the markdown-formatted
leaderboard will be written. The leaderboard contains model names
and their evaluation metrics. |

## Outputs 📤

| Name | Type | Description |
|------|------|-------------|
| Output | `str` | A string message indicating the completion status of the evaluation
process. |

## Metadata 🗂️

- **Name**: leaderboard_evaluation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.14.4
- **Tags**:
  - training
- **Last Verified**: 2026-01-22 10:59:58+00:00
- **Owners**:
  - Approvers: None
  - Reviewers: None
