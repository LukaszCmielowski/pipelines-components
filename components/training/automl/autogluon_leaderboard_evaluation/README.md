# Leaderboard Evaluation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate multiple AutoGluon models and generate a leaderboard.

This component aggregates evaluation results from a list of Model artifacts and generates an HTML-formatted leaderboard
ranking the models by their performance metrics. Each model artifact is expected to contain pre-computed metrics at
`model.path / model.metadata["model_name"] / metrics / metrics.json` (e.g. as produced by the autogluon_models_full_refit
component). The component reads these metrics and compiles them into a sorted leaderboard table.

The leaderboard is sorted by the specified evaluation metric in descending order, making it easy to identify the
best-performing models. The output is written as HTML that can be used for reporting and model selection decisions.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `models` | `List[dsl.Model]` | `None` | A list of Model artifacts with metadata "model_name" and metrics at `model.path / model_name / metrics / metrics.json`. |
| `eval_metric` | `str` | `None` | Metric key for ranking (e.g. "accuracy", "root_mean_squared_error"). Leaderboard sorted by this metric descending. |
| `html_artifact` | `dsl.Output[dsl.HTML]` | `None` | Output artifact where the HTML-formatted leaderboard will be written. |

## Outputs 📤

This component does not return any outputs. The leaderboard is written directly to the `html_artifact` output.

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
