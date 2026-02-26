# Autogluon Models Full Refit ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Refit a specific AutoGluon model on the full training dataset.

This component takes a trained AutoGluon TabularPredictor and refits a specific model (identified by `model_name`). By
default AutoGluon `refit_full` uses the predictor's training and validation data; the **test_dataset** input is used for
**evaluation** and for writing metrics. The refitting process retrains the model architecture on the full (train+val)
data, typically improving performance compared to models trained on sampled data.

After refitting, the component creates a cleaned clone of the predictor containing only the original model and its
refitted version (with `_FULL` suffix). The refitted model is set as the best model and the predictor is optimized to
save space by removing unnecessary models and files. Evaluation metrics, feature importance, and (for classification)
confusion matrix are written under `model_artifact.path` / `model_name_FULL` / `metrics`. A Jupyter notebook
(`automl_predictor_notebook.ipynb`) is generated under `model_artifact.path` / `model_name_FULL` / `notebooks` for
inference and exploration, using `pipeline_name`, `run_id`, and `sample_row` for run context and example input.

This component is typically used in a two-stage training pipeline where models are first trained on sampled data for
exploration, then the best candidates are refitted on the full dataset for optimal performance. Supported problem types
are `regression`, `binary`, and `multiclass`; any other type raises `ValueError`.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `model_name` | `str` | — | Name of the model to refit (refitted model saved with `_FULL` suffix). |
| `test_dataset` | `dsl.Input[dsl.Dataset]` | — | Dataset artifact (CSV) used for evaluation and for writing metrics; format should match the data used during initial training. |
| `predictor_path` | `str` | — | Path to a trained AutoGluon TabularPredictor that includes the model specified by `model_name`. |
| `sampling_config` | `dict` | — | Configuration for data sampling (stored in artifact metadata). |
| `split_config` | `dict` | — | Configuration for data splitting (stored in artifact metadata). |
| `model_config` | `dict` | — | Configuration for model training (stored in artifact metadata). |
| `pipeline_name` | `str` | — | Name of the pipeline run (used in generated notebook; last hyphen-separated segment is stripped for display). |
| `run_id` | `str` | — | ID of the pipeline run (used in generated notebook). |
| `sample_row` | `str` | — | JSON string of a list of row objects (e.g. `[{"feature1": 1, "target": 0}]`). Used as example input in the generated notebook; the label column is stripped per row. |
| `model_artifact` | `dsl.Output[dsl.Model]` | — | Output where the refitted predictor, metrics, and generated notebook are saved. |

## Outputs 📤

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `model_name` | `str` | Name of the refitted model (i.e. `model_name` with `_FULL` suffix). |

The refitted predictor, metrics under `model_artifact.path` / `model_name_FULL` / `metrics`, and the notebook under
`model_artifact.path` / `model_name_FULL` / `notebooks` / `automl_predictor_notebook.ipynb` are written to the
`model_artifact` output. Artifact metadata includes `display_name` and `context` (e.g. `data_config`, `task_type`,
`label_column`, `model_config`, `location`, `metrics`). The `context.metrics` dict contains `test_data` with the
evaluation results on the test dataset.

## Usage Examples 💡

### Refit a single model (typical in a ParallelFor)

Usually used after `models_selection`; refit each top model on the full dataset with pipeline placeholders for name and run ID:

```python
from kfp import dsl
from kfp_components.components.training.automl.autogluon_models_full_refit import autogluon_models_full_refit

@dsl.pipeline(name="automl-full-refit-pipeline")
def my_pipeline(selection_task, full_dataset, split_task):
    with dsl.ParallelFor(items=selection_task.outputs["top_models"], parallelism=2) as model_name:
        refit_task = autogluon_models_full_refit(
            model_name=model_name,
            full_dataset=full_dataset,
            predictor_path=selection_task.outputs["predictor_path"],
            sampling_config=selection_task.outputs["sample_config"],
            split_config=split_task.outputs["split_config"],
            model_config=selection_task.outputs["model_config"],
            pipeline_name=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
            run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            sample_row=split_task.outputs["sample_row"],
        )
    return refit_task
```

### Refit with explicit config dicts

```python
refit_task = autogluon_models_full_refit(
    model_name="LightGBM_BAG_L1",
    full_dataset=full_dataset,
    predictor_path="/workspace/autogluon_predictor",
    sampling_config={"n_samples": 10000},
    split_config={"test_size": 0.2, "random_state": 42},
    model_config={"eval_metric": "r2", "time_limit": 300},
    pipeline_name="my-automl-pipeline",
    run_id="run-123",
    sample_row='[{"feature1": 1.0, "target": 0.5}]',
)
```

## Metadata 🗂️

- **Name**: autogluon_models_full_refit
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - training
- **Last Verified**: 2026-01-22 10:31:36+00:00
- **Owners**:
  - Approvers: None
  - Reviewers: None
