# Tabular Train Test Split ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Splits a tabular dataset into train and test sets and writes them to output artifacts.

## Inputs 📥

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `dataset` | `dsl.Input[dsl.Dataset]` | `None` | Input CSV dataset to split. |
| `sampled_train_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset artifact for the train split. |
| `sampled_test_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset artifact for the test split. |
| `test_size` | `float` | `0.3` | Proportion of the data to include in the test split. |

## Outputs 📤

| Name | Type | Description |
| ------ | ------ | ------------- |
| Output | `NamedTuple('outputs', sample_row=str, split_config=dict)` | Contains a sample row and a split configuration dictionary. |

## Metadata 🗂️

- **Name**: tabular_train_test_split
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - data-processing
- **Last Verified**: 2026-01-22 10:28:49+00:00
- **Owners**:
  - Approvers:
    - Mateusz-Switala
  - Reviewers:
    - Mateusz-Switala
