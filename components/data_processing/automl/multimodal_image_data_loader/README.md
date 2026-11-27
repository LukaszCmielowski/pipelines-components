# Multimodal Image Data Loader ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Validate an image manifest, stage images, and write fixed train, tuning, and holdout splits.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `bucket_name` | `str` | `None` | S3 bucket containing the manifest and images. |
| `file_key` | `str` | `None` | CSV manifest object key. Image keys are relative to its directory. |
| `image_columns` | `list[str]` | `None` | One manifest column holding JPEG or PNG object keys. |
| `label_column` | `str` | `None` | Manifest column holding the class label. |
| `task_type` | `str` | `None` | binary or multiclass. |
| `task_family` | `str` | `None` | image_prediction. |
| `workspace_path` | `str` | `None` | Run-scoped workspace shared with the trainer. |
| `data_quality` | `dsl.Output[dsl.Artifact]` | `None` | Aggregate validation and split information. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | Path to the split manifest in the shared workspace. |

## Metadata 🗂️

- **Name**: multimodal_image_data_loader
- **Description**: Validate, stage, and split an S3 image classification dataset.
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.16.1
- **Tags**:
  - data-processing
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

Set the task's image to the prebuilt AutoMM prototype runtime. The loader
expects an S3 Connection Secret using the AutoML keys
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT`, and optional
`AWS_DEFAULT_REGION`. Pass a writable run-scoped workspace shared with the
trainer. The loader keeps the labeled splits and staged images in that workspace
and publishes aggregate counts and a manifest digest as an artifact.

See the [AutoMM prototype pipeline](../../../../pipelines/training/automl/autogluon_multimodal_training_pipeline/README.md)
for the manifest format, supported inputs, and runtime image.
