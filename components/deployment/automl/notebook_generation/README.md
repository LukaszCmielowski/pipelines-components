# Notebook Generation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Generate a Jupyter notebook for reviewing and running an AutoGluon predictor.

Produces a notebook artifact (automl_predictor_notebook.ipynb) that lets users review the experiment leaderboard, load a
trained AutoGluon model from S3, and run predictions. The notebook is pre-filled with pipeline run details, model name,
and a sample row for prediction.

**Problem types:** Use ``problem_type`` to select the template:

- **regression**: Template uses ``predict(score_df)`` for numeric targets. - **binary** or **multiclass**: Template uses
``predict_proba(score_df)`` and includes a confusion matrix section. Both values share the same classification template.

Invalid values raise ``ValueError``.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem_type` | `str` | `None` | One of ``"regression"``, ``"binary"``, or ``"multiclass"``.
Determines which notebook template is used. |
| `model_name` | `str` | `None` | Name of the trained model to load, matching the leaderboard
model column. |
| `notebook_artifact` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact where the generated notebook file
(automl_predictor_notebook.ipynb) is written. |
| `pipeline_name` | `str` | `None` | Full pipeline run name (e.g. from KFP); used to locate
artifacts in S3. The component strips the last hyphen-separated
segment (run suffix) for the notebook path. |
| `run_id` | `str` | `None` | Pipeline run ID; used with pipeline_name to form the S3 prefix
for leaderboard and model artifacts. |
| `sample_row` | `str` | `None` | JSON string of a single row (object of feature names to
values), used in the notebook's prediction example. The component
parses it, removes the label column, and injects the result.
Expected format: '[{"col1": "val1","col2":"val2"},{"col1":"val3","col2":"val4"}]' |
| `label_column` | `str` | `None` | Key in the parsed sample_row for the target/label column;
this column is omitted from the sample row in the notebook. |

## Metadata 🗂️

- **Name**: notebook_generation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - deployment
- **Last Verified**: 2026-02-03 00:00:00+00:00
- **Owners**:
  - Approvers:
    - Mateusz-Switala
  - Reviewers:
    - Mateusz-Switala
