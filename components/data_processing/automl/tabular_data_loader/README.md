# Tabular Data Loader ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Loads a tabular CSV dataset from an S3-compatible bucket using AWS credentials.

## Inputs 📥

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `file_key` | `str` | `None` | Path to the CSV file in the S3 bucket. |
| `bucket_name` | `str` | `None` | Name of the S3 bucket. |
| `full_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output artifact for the downloaded dataset. |

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
