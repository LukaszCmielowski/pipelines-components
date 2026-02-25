# Train Test Split ✂️

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Splits a tabular (CSV) dataset into train and test sets for AutoML workflows.

The Train Test Split component takes a single CSV dataset and splits it into training and test sets using scikit-learn's `train_test_split`.
For **regression** tasks the split is random; for **binary** and **multiclass** tasks the split is **stratified** by the label column by default, so that class proportions are preserved in both splits.
The component writes the train and test CSVs to the output artifacts and returns a sample row (from the test set) and the split configuration.

## Inputs 📥

| Parameter              | Type                     | Default   | Description |
| ---------------------- | ------------------------ | -------- | ----------- |
| `dataset`              | `dsl.Input[dsl.Dataset]` | *required* | Input CSV dataset to split. |
| `task_type`            | `str`                    | *required* | Machine learning task type: `"binary"`, `"multiclass"`, or `"regression"`. |
| `label_column`         | `str`                    | *required* | Name of the label/target column. |
| `split_config`         | `dict`                   | *required* | Split configuration dictionary. Available keys: "test_size" (float), "random_state" (int), "stratify" (bool). |
| `sampled_train_dataset` | `dsl.Output[dsl.Dataset]` | *required* | Output dataset artifact for the train split. |
| `sampled_test_dataset` | `dsl.Output[dsl.Dataset]` | *required* | Output dataset artifact for the test split. |

### Split Configuration

The `split_config` dictionary supports:

```python
{
    "test_size": 0.3,       # Proportion of dataset for test split (default: 0.3)
    "random_state": 42,     # Random seed for reproducibility (default: 42)
    "stratify": True        # Use stratified split for binary/multiclass (default: True)
}
```

- **Regression**: `stratify` is ignored; the split is always random.
- **Binary / multiclass**: If `stratify` is `True` (default), the split is stratified by `label_column`; if `False`, the split is random.

## Outputs 📤

| Output                 | Type           | Description |
| ---------------------- | -------------- | ----------- |
| `sampled_train_dataset` | `dsl.Dataset`  | Training split (CSV). |
| `sampled_test_dataset`  | `dsl.Dataset`  | Test split (CSV). |
| Return value            | `NamedTuple`   | `sample_row`: JSON string of one row from the test set; `split_config`: dict with `test_size`. |

## Usage Examples 💡

### Regression (random split)

```python
from kfp import dsl
from kfp_components.components.data_processing.automl.tabular_train_test_split import tabular_train_test_split

@dsl.pipeline(name="train-test-split-regression-pipeline")
def my_pipeline(dataset):
    split_task = tabular_train_test_split(
        dataset=dataset,
        task_type="regression",
        label_column="price",
        split_config={"test_size": 0.3, "random_state": 42},
    )
    return split_task
```

### Classification (stratified split)

```python
@dsl.pipeline(name="train-test-split-classification-pipeline")
def my_pipeline(dataset):
    split_task = tabular_train_test_split(
        dataset=dataset,
        task_type="multiclass",
        label_column="target",
        split_config={"test_size": 0.2, "random_state": 42, "stratify": True},
    )
    return split_task
```

### Binary classification with custom test size

```python
@dsl.pipeline(name="train-test-split-binary-pipeline")
def my_pipeline(dataset):
    split_task = tabular_train_test_split(
        dataset=dataset,
        task_type="binary",
        label_column="label",
        split_config={"test_size": 0.25, "random_state": 42},
    )
    return split_task
```

### Classification with random (non-stratified) split

```python
@dsl.pipeline(name="train-test-split-random-classification-pipeline")
def my_pipeline(dataset):
    split_task = tabular_train_test_split(
        dataset=dataset,
        task_type="multiclass",
        label_column="target",
        split_config={"test_size": 0.2, "stratify": False},
    )
    return split_task
```

## Notes 📝

- **Stratified split**: Used by default for `task_type="binary"` and `"multiclass"` when `split_config["stratify"]` is `True` (default) to preserve class distribution in train and test sets.
- **Reproducibility**: Pass `random_state` in `split_config` (default: 42) for consistent splits.
- **Output format**: Train and test artifacts are written as CSV files; the component appends `.csv` to the output artifact URIs.

## Metadata 🗂️

- **Name**: tabular_train_test_split
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow Pipelines >= 2.15.2
  - pandas
  - scikit-learn
- **Tags**:
  - automl
  - data-processing
  - train-test-split
- **Last Verified**: 2025-01-27 00:00:00+00:00

## Additional Resources 📚

- **AutoML Documentation**: [AutoML README](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/README.md)
- **Components Documentation**: [Components Structure](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/components.md)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
