# Train Test Split ✂️

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Splits data into train/test sets and performs sampling for AutoML workflows.

The Train Test Split component is a critical step in the AutoML pipeline that prepares data for model
training and evaluation. It splits the tabular data into training and test sets using appropriate
sampling techniques (random, stratified, or time-series driven). Additionally, it samples a subset
of the training data (default: 500 samples) for initial model building to reduce computational
cost during the exploration phase.

This component supports multiple task types (classification, regression, time-series) and adapts its splitting strategy accordingly. For classification tasks, it supports stratified sampling to maintain class distribution. For time-series tasks, it uses chronological splitting to preserve temporal order.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset artifact containing the full training data. |
| `test_data` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset artifact containing the test data. |
| `sampled_train_data` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset artifact containing the sampled training data for initial model building. |
| `tabular_data` | `dsl.Input[dsl.Dataset]` | `None` | Input dataset artifact containing the raw tabular data from data-loader. |
| `task_type` | `str` | `None` | Type of ML task. Required: `"classification"`, `"regression"`, or `"time_series"`. |
| `label_column` | `str` | `None` | Name of the label/target column in the dataset. Required. |
| `sampling_config` | `dict` | `None` | Optional dictionary with sampling configuration. See [Sampling Configuration](#sampling-configuration) below. |
| `split_config` | `dict` | `None` | Optional dictionary with train/test split configuration. See [Split Configuration](#split-configuration) below. |

### Sampling Configuration

The `sampling_config` dictionary supports:

```python
{
    "n_samples": 500,              # Number of samples for initial model building (default: 500)
    "sampling_method": "random"   # Sampling method: "random", "stratified", or "truncate"
}
```

**Sampling Methods:**

- `"random"` - Random sampling for general use cases
- `"stratified"` - Stratified sampling for classification tasks to maintain class distribution
- `"truncate"` - Sampling last n records for time-series forecasting tasks

### Split Configuration

The `split_config` dictionary supports:

```python
{
    "test_size": 0.2,        # Proportion of dataset for test split (default: 0.2)
    "random_state": 42,      # Random seed for reproducibility (default: 42)
    "stratify": True         # Enable stratified splitting for classification (default: True for classification)
}
```

For time-series tasks:

```python
{
    "n_last_rows": 100       # Number of last rows to use as test set for time-series
}
```

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `train_data` | `dsl.Dataset` | The full training dataset artifact. |
| `test_data` | `dsl.Dataset` | The test dataset artifact. |
| `sampled_train_data` | `dsl.Dataset` | The sampled training dataset (default: 500 samples) for initial model building. |
| Return value | `str` | A message indicating the completion status of data splitting. |

## Usage Examples 💡

### Classification Task

```python
from kfp import dsl
from kfp_components.components.training.automl.data_processing.train_test_split import train_test_split

@dsl.pipeline(name="train-test-split-classification-pipeline")
def my_pipeline(tabular_data):
    """Example pipeline for classification task."""
    split_task = train_test_split(
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
    split_task = train_test_split(
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
    split_task = train_test_split(
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

- **Name**: train-test-split
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: pandas, Version: >=2.0.0
    - Name: scikit-learn, Version: >=1.0.0
- **Tags**:
  - automl
  - data-processing
  - train-test-split
  - data-sampling
- **Last Verified**: 2025-01-27 00:00:00+00:00

## Additional Resources 📚

- **AutoML Documentation**: [AutoML README](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/README.md)
- **Components Documentation**: [Components Structure](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/components.md)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
