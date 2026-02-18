# Autogluon Models Full Refit ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Refit a specific AutoGluon model on the full training dataset.

This component takes a trained AutoGluon TabularPredictor and refits a specific model (identified by `model_name`) on the
complete training dataset. The refitting process retrains the model architecture on the full data, typically improving
performance compared to models trained on sampled data.

After refitting, the component creates a cleaned clone of the predictor containing only the original model and its
refitted version (with `_FULL` suffix). The refitted model is set as the best model and the predictor is optimized to
save space by removing unnecessary models and files. Evaluation metrics, feature importance, and (for classification)
confusion matrix are written under `model_artifact.path` / `model_name_FULL` / `metrics`. A Jupyter notebook
(`automl_predictor_notebook.ipynb`) is generated at the artifact root for inference and exploration, using
`pipeline_name`, `run_id`, and `sample_row` for run context and example input.

This component is typically used in a two-stage training pipeline where models are first trained on sampled data for
exploration, then the best candidates are refitted on the full dataset for optimal performance. Supported problem types
are `regression`, `binary`, and `multiclass`; any other type raises `ValueError`.

## Inputs 📥

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `model_name` | `str` | `None` | The name of the model to refit. Should match a model name in the predictor. The refitted model will be saved with the suffix "_FULL" appended to this name. |
| `full_dataset` | `dsl.Input[dsl.Dataset]` | `None` | A Dataset artifact containing the complete training dataset in CSV format. Used to retrain the specified model. The dataset should match the format and schema of the data used during initial model training. |
| `predictor_path` | `str` | `None` | Path (string) to a trained AutoGluon TabularPredictor that includes the model specified by model_name. The predictor should have been trained previously, potentially on a sampled subset of the data. |
| `model_artifact` | `dsl.Output[dsl.Model]` | `None` | Output Model artifact where the refitted predictor will be saved. The artifact will contain a cleaned predictor with only the original model and its refitted "_FULL" version. Metrics are written under model_artifact.path / model_name_FULL / metrics. The metadata will include the model_name with "_FULL" suffix. |

## Outputs 📤

| Name | Type | Description |
| ------ | ------ | ------------- |
| Output | `NamedTuple('outputs', model_name=str)` | None. The refitted model is saved to the model_artifact output. |

## Metadata 🗂️

- **Name**: autogluon_models_full_refit
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.14.4
- **Tags**:
  - training
- **Last Verified**: 2026-01-22 10:31:36+00:00
- **Owners**:
  - Approvers:
    - Mateusz-Switala
  - Reviewers:
    - Mateusz-Switala
