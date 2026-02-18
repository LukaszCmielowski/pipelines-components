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

| Output | Type | Description |
| ------ | ---- | ----------- |
| `train_data` | `dsl.Dataset` | Full training dataset artifact. |
| `test_data` | `dsl.Dataset` | Test dataset artifact. |
| `sampled_train_data` | `dsl.Dataset` | Sampled training dataset (default: 500 samples) for initial model building. |
| Return value | `str` | Message indicating the completion status of data splitting. |

## Usage Examples 💡

### Classification Task

```python
from kfp import dsl
from kfp_components.components.training.automl.data_processing.tabular_train_test_split import tabular_train_test_split

@dsl.pipeline(name="train-test-split-classification-pipeline")
def my_pipeline(tabular_data):
    """Example pipeline for classification task."""
    split_task = tabular_train_test_split(
        tabular_data=tabular_data,
        task_type="classification",
        label_column="target",
        sampling_config={
            "n_samples": 500,
            "sampling_method": "stratified"
        },
        split_config={
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True
        }
    )
    return split_task
```

### Regression Task

```python
@dsl.pipeline(name="train-test-split-regression-pipeline")
def my_pipeline(tabular_data):
    """Example pipeline for regression task."""
    split_task = tabular_train_test_split(
        tabular_data=tabular_data,
        task_type="regression",
        label_column="price",
        sampling_config={
            "n_samples": 500,
            "sampling_method": "random"
        },
        split_config={
            "test_size": 0.2,
            "random_state": 42
        }
    )
    return split_task
```

### Time-Series Task

```python
@dsl.pipeline(name="train-test-split-timeseries-pipeline")
def my_pipeline(tabular_data):
    """Example pipeline for time-series task."""
    split_task = tabular_train_test_split(
        tabular_data=tabular_data,
        task_type="time_series",
        label_column="value",
        sampling_config={
            "n_samples": 500,
            "sampling_method": "truncate"
        },
        split_config={
            "n_last_rows": 100  # Last 100 rows as test set
        }
    )
    return split_task
```

## Sampling Strategy 🎯

The component uses different sampling strategies based on the task type:

1. **Classification**: Uses stratified sampling by default to maintain class distribution in the sampled dataset
2. **Regression**: Uses random sampling
3. **Time-Series**: Uses truncate method to sample the last n records, preserving temporal order

The sampled dataset (default: 500 samples) is used in the model-building-selection stage to reduce computational cost during initial model exploration.

## Notes 📝

- **Default Sample Size**: 500 samples are used for initial model building to balance exploration speed and model quality
- **Stratified Splitting**: Automatically enabled for classification tasks to maintain class distribution
- **Time-Series Handling**: Uses chronological splitting to preserve temporal order (no shuffling)
- **Reproducibility**: Uses random_state parameter for reproducible splits

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
