# Autogluon Models Full Refit ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Refit a specific AutoGluon model on the full training dataset.

This component takes a trained AutoGluon TabularPredictor, loaded from predictor_path, and refits a specific model,
identified by model_name, on the full training data. By default AutoGluon refit_full uses the predictor's training and
validation data; the full_dataset is used for evaluation and for writing metrics. The refitted model is saved with the
suffix "_FULL" appended to model_name.

The component clones the predictor to model_artifact.path / model_name_FULL / predictor, keeps only the specified model
and its refitted version, sets the refitted model as best, and saves space by removing other models. Evaluation metrics,
feature importance, and (for classification) confusion matrix are written under model_artifact.path / model_name_FULL /
metrics. A Jupyter notebook (automl_predictor_notebook.ipynb) is written under model_artifact.path / model_name_FULL /
notebooks for inference and exploration; pipeline_name, run_id, and sample_row are used to fill in run context and
example input (the label column is stripped from sample_row in the notebook). Artifact metadata includes display_name,
context (data_config, task_type, label_column, model_config, location, metrics), and context.location.notebook.
Supported problem types are regression, binary, and multiclass; any other type raises ValueError.

This component is typically used in a two-stage training pipeline where models are first trained on sampled data for
exploration, then the best candidates are refitted on the full dataset for optimal performance.

## Inputs 📥

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `model_name` | `str` | `None` | The name of the model to refit. Should match a model name in the predictor. The refitted model will be saved with the suffix "_FULL" appended to this name. |
| `full_dataset` | `dsl.Input[dsl.Dataset]` | `None` | A Dataset artifact containing the complete training dataset in CSV format. Used to retrain the specified model. The dataset should match the format and schema of the data used during initial model training. |
| `predictor_path` | `str` | `None` | Path (string) to a trained AutoGluon TabularPredictor that includes the model specified by model_name. The predictor should have been trained previously, potentially on a sampled subset of the data. |
| `sampling_config` | `dict` | `None` | Configuration dictionary for data sampling (stored in artifact metadata). |
| `split_config` | `dict` | `None` | Configuration dictionary for data splitting (stored in artifact metadata). |
| `model_config` | `dict` | `None` | Configuration dictionary for model training (stored in artifact metadata). |
| `pipeline_name` | `str` | `None` | Name of the pipeline run. The last hyphen-separated segment is stripped for use in the generated notebook. |
| `run_id` | `str` | `None` | ID of the pipeline run (used in the generated notebook). |
| `sample_row` | `str` | `None` | JSON string of a list of row objects (e.g. '[{"feature1": 1, "target": 0}]'). Used as example input in the generated notebook; the label column is removed from each row. |
| `model_artifact` | `dsl.Output[dsl.Model]` | `None` | Output Model artifact where the refitted predictor will be saved. The artifact will contain a cleaned predictor with only the original model and its refitted "_FULL" version. Metrics are written under model_artifact.path / model_name_FULL / metrics. The metadata will include the model_name with "_FULL" suffix. |

## Outputs 📤

| Name | Type | Description |
| ------ | ------ | ------------- |
| Output | `NamedTuple('outputs', model_name=str)` | NamedTuple with field model_name: the refitted model name (model_name with "_FULL" suffix). The refitted predictor and artifacts are also written to model_artifact. |

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
