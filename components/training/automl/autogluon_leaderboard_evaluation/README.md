# Leaderboard Evaluation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate multiple AutoGluon models and generate a leaderboard.

This component aggregates evaluation results from a list of Model artifacts and generates an HTML-formatted leaderboard
ranking the models by their performance metrics. Each model artifact is expected to contain pre-computed metrics at
`model.path / model.metadata["display_name"] / metrics / metrics.json` (e.g. as produced by the autogluon_models_full_refit
component). The component reads these metrics and compiles them into a sorted leaderboard table.

The leaderboard is sorted by the specified evaluation metric in descending order, making it easy to identify the
best-performing models. The output is written as HTML that can be used for reporting and model selection decisions.

## Inputs 📥

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `models` | `List[dsl.Model]` | `None` | A list of Model artifacts. Each should have metadata containing a "model_name" field and metrics file at model.path / model_name / metrics / metrics.json. |
| `eval_metric` | `str` | `None` | The name of the evaluation metric to use for ranking. Must match a key in the metrics JSON (e.g., "accuracy" for classification, "root_mean_squared_error" for regression). The leaderboard is sorted by this metric in descending order. |
| `html_artifact` | `dsl.Output[dsl.HTML]` | `None` | Output artifact where the HTML-formatted leaderboard will be written. The leaderboard contains model names and their evaluation metrics. |

## Outputs 📤

| Name | Type | Description |
| ------ | ------ | ------------- |
| Output | `NamedTuple('outputs', best_model=str)` | NamedTuple with best_model: the name of the top-ranked model (best on eval_metric). |

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
  - Approvers:
    - Mateusz-Switala
  - Reviewers:
    - Mateusz-Switala
