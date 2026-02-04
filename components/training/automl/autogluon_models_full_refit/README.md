# Autogluon Models Full Refit ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Refit a specific AutoGluon model on the full training dataset.

This component takes a trained AutoGluon TabularPredictor and refits a specific model (identified by model_name) on the
complete training dataset. The refitting process retrains the model architecture on the full data, typically improving
performance compared to models trained on sampled data.

After refitting, the component creates a cleaned clone of the predictor containing only the original model and its
refitted version (with "_FULL" suffix). The refitted model is set as the best model and the predictor is optimized to
save space by removing unnecessary models and files.

This component is typically used in a two-stage training pipeline where models are first trained on sampled data for
exploration, then the best candidates are refitted on the full dataset for optimal performance.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `model_name` | `str` | `None` | Name of the model to refit (refitted model saved with "_FULL" suffix). |
| `full_dataset` | `dsl.Input[dsl.Dataset]` | `None` | Dataset artifact (CSV) with complete training data; format must match initial training. |
| `predictor_path` | `str` | `None` | Path to a trained AutoGluon TabularPredictor that includes the model specified by model_name. |
| `model_artifact` | `dsl.Output[dsl.Model]` | `None` | Output where the refitted predictor and metrics (under model_artifact.path / model_name_FULL / metrics) are saved. |

## Outputs 📤

This component does not return any outputs. The refitted predictor and metrics are written to the `model_artifact` output.

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
