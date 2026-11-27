# Autogluon Multimodal Training Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train a single AutoMM image classifier and publish an offline-loadable predictor.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `train_data_secret_name` | `str` | `None` | Name of the S3 Connection Secret in the run namespace. |
| `train_data_bucket_name` | `str` | `None` | Bucket containing the manifest and images. |
| `train_data_file_key` | `str` | `None` | S3 key of the CSV manifest. Image keys are relative to its directory. |
| `label_column` | `str` | `None` | Label column in the manifest. |
| `image_columns` | `list[str]` | `None` | Exactly one column containing relative JPEG or PNG image keys. |
| `task_type` | `str` | `None` | binary or multiclass. |
| `task_family` | `str` | `image_prediction` | image_prediction, the only task family supported by this prototype. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `MultimodalOutputs` | Standalone model artifact, metrics, aggregate data quality, and model URI. |

## Metadata 🗂️

- **Name**: autogluon-multimodal-training-pipeline
- **Description**: Prototype AutoMM image classification from S3 to an offline-loadable predictor.
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

## Prototype workflow

This pipeline trains and packages one image classifier for offline notebook use.
It supports image-only binary and multiclass classification. Compile and submit
it through the Kubeflow Pipelines SDK or upload the compiled YAML manually.

### Prepare the data and Connection

Upload a UTF-8 CSV manifest and JPEG or PNG images to the same S3-compatible
bucket. Image keys are relative to the manifest directory. For a manifest at
`products/train.csv`, the first example below resolves to
`products/images/shoe-01.jpg`.

```csv
image_key,category
images/shoe-01.jpg,shoe
images/bag-01.png,bag
```

Use at least three distinct images per class so each class can appear in the
train, tuning, and holdout splits. Use a larger dataset for a meaningful quality
check. The loader requires two classes for binary classification or at least
three for multiclass. It deduplicates byte-identical images before creating the
splits, rejects conflicting labels for duplicates, and fails on unreadable images.

Create or select an S3 Connection Secret in the run namespace. Use the existing
AutoML keys `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_S3_ENDPOINT`.
`AWS_DEFAULT_REGION` is optional. The task uses TLS trust configured for the
pipeline server. The loader stages images in an 8 GiB run workspace and limits
the manifest to 16 MiB and 10,000 rows, each image to 8 MiB and 16 million pixels,
and the total staged image data to 4 GiB.

### Build and compile

Build the dedicated image for Linux x86-64 from the repository root and publish
it to a registry the cluster can pull from. The default pipeline requests one
NVIDIA GPU for training. Use `--cpu` at compile time for a small development run.

```sh
AUTOMM_IMAGE=quay.io/YOUR_ORG/automm-prototype:dev
podman build --platform linux/amd64 \
  -f components/training/automl/autogluon_multimodal_models_training/Containerfile \
  -t "$AUTOMM_IMAGE" .
podman push "$AUTOMM_IMAGE"
python pipelines/training/automl/autogluon_multimodal_training_pipeline/pipeline.py \
  --image "$AUTOMM_IMAGE"
```

The image pins AutoMM 1.5.0, TIMM 1.0.3, and PyTorch 2.8.0 with CUDA 12.8.
Training may download the ResNet-18 pretrained weights on first use. The
prototype's offline check applies to loading and scoring the exported predictor.

Upload `autogluon-multimodal-training-pipeline.yaml` and supply the Secret name,
bucket, manifest key, and the column names. For the example manifest, set
`image_columns=["image_key"]`, `label_column=category`, and `task_type=binary`.
Both tasks disable caching so a fresh run reads the current objects.

To submit through the SDK, use an authenticated KFP client for your pipeline server.

```python
run = client.create_run_from_pipeline_package(
    "autogluon-multimodal-training-pipeline.yaml",
    arguments={
        "train_data_secret_name": "my-s3-connection",
        "train_data_bucket_name": "training",
        "train_data_file_key": "products/train.csv",
        "label_column": "category",
        "image_columns": ["image_key"],
        "task_type": "binary",
    },
    experiment_name="automm-prototype",
)
```

### Score an image in a notebook

Copy the full model directory from a trusted pipeline run's `model_uri` into an
environment with the same AutoMM and PyTorch versions and open the
[prediction notebook](predict_local_image.ipynb).
Provide a local JPEG or PNG and execute the notebook in a fresh kernel. To verify
offline inference, make the model directory and image available locally before
blocking network access. The notebook loads the saved predictor and displays
its prediction and class probabilities.

The model contains the standalone predictor under `predictor/`, its class order
and input schema in `model.json`, and holdout metrics under `metrics/`. The run
also publishes scalar KFP metrics, a data-quality summary, and the model URI for
downstream tooling.
