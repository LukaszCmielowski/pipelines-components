# Notebook Generation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Generates a Jupyter notebook for reviewing and running an AutoGluon predictor after a training run.

The component produces a single notebook artifact (`automl_predictor_notebook.ipynb`) that lets users:

- Review the experiment leaderboard (HTML from S3)
- Load a chosen AutoGluon model from S3
- Run predictions on sample data

The notebook is pre-filled with pipeline run details, model name, and a sample row (features only; label column removed).
The sample row is passed as a JSON string and parsed inside the component.
The exact content depends on **problem type**: regression uses `predict(score_df)`; classification (binary/multiclass)
uses `predict_proba(score_df)` and a confusion matrix section.

## Inputs 📥

|Parameter|Type|Description|
|---------|-----|------------|
|`problem_type`|`str`|One of `"regression"`, `"binary"`, or `"multiclass"`. Invalid values raise `ValueError`.|
|`model_name`|`str`|Name of the trained model to load; must match a model in the leaderboard.|
|`notebook_artifact`|`dsl.Output[dsl.Artifact]`|Output artifact; notebook written to `notebook_artifact.path` as `automl_predictor_notebook.ipynb`.|
|`pipeline_name`|`str`|Full pipeline run name (e.g. from KFP). Last hyphen-separated segment stripped.|
|`run_id`|`str`|Pipeline run ID; with `pipeline_name` forms the S3 prefix for artifacts.|
|`sample_row`|`str`|JSON string of one row (feature names → values). Parsed, label removed, then injected.|
|`label_column`|`str`|Key in parsed `sample_row` for the target; omitted from the sample row in the notebook.|

## Outputs 📤

|Output|Description|
|------|------------|
|**Artifact**|The generated notebook is written to `notebook_artifact.path/automl_predictor_notebook.ipynb`.|
|**Return**|None.|

## Problem types and templates 🎯

|`problem_type`|Template behavior|
|--------------|-----------------|
|`"regression"`|Leaderboard, model load, feature importance, `predict(score_df)` for numeric targets.|
|`"binary"`|Same as regression plus confusion matrix and `predict_proba(score_df)`.|
|`"multiclass"`|Same classification template as `"binary"`.|

Any other value (e.g. `"classification"`) raises `ValueError` with message `Invalid problem type: <value>`.

## Usage examples 💡

### Regression

```python
from kfp import dsl
from kfp_components.components.deployment.automl.notebook_generation import notebook_generation

@dsl.pipeline(name="notebook-regression-pipeline")
def my_pipeline(pipeline_name: str, run_id: str, sample_row: str, label_column: str):
    notebook_task = notebook_generation(
        problem_type="regression",
        model_name="WeightedEnsemble_L2",
        pipeline_name=pipeline_name,
        run_id=run_id,
        sample_row=sample_row,
        label_column=label_column,
    )
    return notebook_task
```

### Classification (binary or multiclass)

```python
notebook_task = notebook_generation(
    problem_type="binary",  # or "multiclass"
    model_name="WeightedEnsemble_L2",
    pipeline_name=pipeline_name,
    run_id=run_id,
    sample_row=sample_row,
    label_column=label_column,
)
```

### In the AutoGluon tabular training pipeline

The pipeline passes `sample_row` from the train-test-split component and uses the same `label_column` and `task_type`-derived problem type:

```python
notebook_generation(
    problem_type=task_type,  # "regression", "binary", or "multiclass"
    model_name=...,
    pipeline_name=...,
    run_id=...,
    sample_row=train_test_split_task.outputs["sample_row"],
    label_column=label_column,
)
```

## Notes 📝

- **Pipeline name:** The component strips the last segment after the final hyphen (e.g. `my-pipeline-abc123` → `my-pipeline`) when filling the notebook, so S3 paths in the notebook match the expected layout.
- **Sample row:** `sample_row` must be a JSON string (e.g. from `train_test_split`'s `sample_row` output, which KFP may serialize). The component parses it, removes the label key, and injects the resulting dict into the notebook.
- **S3:** The notebook assumes the workbench has S3 configured (`AWS_S3_ENDPOINT`, `AWS_S3_BUCKET`) so users can load the leaderboard and model from the pipeline run.

## Metadata 🗂️

- **Name**: notebook_generation
- **Stability**: alpha
- **Dependencies**: Kubeflow Pipelines >=2.14.4
- **Tags**: deployment
- **Last Verified**: 2026-02-03
