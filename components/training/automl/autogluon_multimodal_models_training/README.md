# Autogluon Multimodal Models Training ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train one ResNet-18 AutoMM predictor and evaluate the exported model on the holdout.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `split_manifest_path` | `str` | `None` | Staged dataset split manifest in the run workspace. |
| `model` | `dsl.Output[dsl.Model]` | `None` | Standalone predictor, versioned model metadata, and evaluation JSON files. |
| `metrics` | `dsl.Output[dsl.Metrics]` | `None` | Holdout metric scalars for the pipeline run. |
| `use_gpu` | `bool` | `True` | Require one GPU. Disable only for development smoke tests. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | URI of the exported model directory. |

## Metadata 🗂️

- **Name**: autogluon_multimodal_models_training
- **Description**: Train and export one AutoMM image classification predictor.
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.16.1
- **Tags**:
  - training
  - automl
  - multimodal
- **Last Verified**: 2026-11-26 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - mateusz-szewczyk-dev
    - LukaszCmielowski
    - Mateusz-Switala
    - jakub-walaszczyk
  - Reviewers:
    - mateusz-szewczyk-dev
    - Mateusz-Switala

<!-- custom-content -->

## Use in a pipeline

Set the task's image to the prebuilt AutoMM runtime from this directory's
Containerfile. Pass the split manifest returned by the image loader and use
the same run-scoped workspace. Training uses one ResNet-18 backbone and a
600-second fit budget. The component reloads the exported predictor and
computes accuracy, balanced accuracy, and a confusion matrix on the holdout.

The model artifact contains `predictor/`, `model.json`, and `metrics/`.
See the [AutoMM prototype pipeline](../../../../pipelines/training/automl/autogluon_multimodal_training_pipeline/README.md)
for image build instructions and the local prediction notebook.
