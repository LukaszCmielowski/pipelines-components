# Train Test Split ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train Test Split component.

TODO: Add a detailed description of what this component does.

Args: input_param: Description of the component parameter. # Add descriptions for other parameters

Returns: Description of what the component returns.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `dsl.Input[dsl.Dataset]` | `None` |  |
| `sampled_train_dataset` | `dsl.Output[dsl.Dataset]` | `None` |  |
| `sampled_test_dataset` | `dsl.Output[dsl.Dataset]` | `None` |  |
| `test_size` | `float` | `0.3` |  |

## Outputs 📤

| Name | Type | Description |
|------|------|-------------|
| Output | `NamedTuple('outputs', sample_row=str)` |  |

## Metadata 🗂️

- **Name**: train_test_split
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
