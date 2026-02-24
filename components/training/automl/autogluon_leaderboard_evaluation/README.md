# Leaderboard Evaluation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate multiple AutoGluon models and generate a leaderboard.

This component aggregates evaluation results from a list of Model artifacts (reading pre-computed metrics from JSON) and
generates an HTML-formatted leaderboard ranking the models by their performance metrics. Each model artifact is expected
to contain metrics at model.path / model.metadata["display_name"] / metrics / metrics.json.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `List[dsl.Model]` | `None` | List of Model artifacts with metadata "display_name" and metrics at model.path / model_name / metrics / metrics.json. |
| `eval_metric` | `str` | `None` | Evaluation metric name for ranking (e.g. "accuracy", "root_mean_squared_error"); leaderboard sorted by this metric descending. |
| `html_artifact` | `dsl.Output[dsl.HTML]` | `None` | Output artifact where the HTML-formatted leaderboard (model names and metrics) will be written. |

## Outputs 📤

| Name | Type | Description |
|------|------|-------------|
| Output | `NamedTuple('outputs', best_model=str)` |  |

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
