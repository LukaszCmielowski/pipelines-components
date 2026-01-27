# Leaderboard Evaluation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate multiple AutoGluon models and generate a leaderboard.

This component evaluates a list of trained AutoGluon TabularPredictor models on a full dataset and generates a
markdown-formatted leaderboard ranking the models by their performance metrics. Each model is loaded, evaluated on the
provided dataset, and the results are compiled into a sorted leaderboard table.

The leaderboard is sorted by the specified evaluation metric in descending order, making it easy to identify the
best-performing models. The output is written as a markdown table that can be used for reporting and model selection
decisions.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `List[dsl.Model]` | `None` | A list of Model artifacts containing trained AutoGluon
TabularPredictor models to evaluate. Each model should have
metadata containing a "model_name" field. |
| `eval_metric` | `str` | `None` | The name of the evaluation metric to use for ranking
models in the leaderboard. This should match one of the metrics
returned by the TabularPredictor's evaluate method (e.g., "accuracy"
for classification, "root_mean_squared_error" for regression).
The leaderboard will be sorted by this metric in descending order. |
| `full_dataset` | `dsl.Input[dsl.Dataset]` | `None` | A Dataset artifact containing the evaluation dataset
on which all models will be evaluated. The dataset should be
compatible with the models' training data format. |
| `markdown_artifact` | `dsl.Output[dsl.Markdown]` | `None` | Output artifact where the markdown-formatted
leaderboard will be written. The leaderboard contains model names
and their evaluation metrics. |

## Outputs 📤

This component does not return any outputs. The leaderboard is written directly to the `markdown_artifact` output.

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
