# Tabular Data Loader ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Downloads a dataset from S3, samples 50% of the rows, and saves the sample to the output artifact.

## Inputs 📥

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `file_key` | `str` | `None` | The S3 object key (path) for the dataset file. |
| `bucket_name` | `str` | `None` | The S3 bucket containing the dataset file. |
| `full_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output artifact where the sampled dataset (CSV) will be written. |

## Outputs 📤

| Name | Type | Description |
| ------ | ------ | ------------- |
| Output | `NamedTuple('outputs', sample_config=dict)` | Contains a sample configuration dictionary. |

## Metadata 🗂️

- **Name**: tabular_data_loader
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - data-processing
- **Last Verified**: 2026-01-22 10:26:27+00:00
- **Owners**:
  - Approvers:
    - Mateusz-Switala
  - Reviewers:
    - Mateusz-Switala
