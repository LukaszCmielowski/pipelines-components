# Tabular Data Loader 📊

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Loads tabular (CSV) data from S3 for AutoML processing. The component reads data in chunks (up to 1GB in memory) and supports configurable sampling strategies.

The Tabular Data Loader is typically the first step in the AutoML pipeline. It streams CSV data from an S3 bucket, optionally samples it using one of the supported strategies, and writes the result to an output dataset artifact.
Authentication uses AWS-style credentials provided via environment variables (e.g. from a Kubernetes secret).

## Inputs 📥

| Parameter        | Type                     | Default     | Description |
| --------------- | ------------------------ | ----------- | ----------- |
| `file_key`      | `str`                    | *required*  | Path to the CSV file within the S3 bucket. |
| `bucket_name`  | `str`                    | *required*  | Name of the S3 bucket containing the file. |
| `full_dataset` | `dsl.Output[dsl.Dataset]` | *required*  | Output artifact where the sampled CSV will be written. |
| `sampling_method` | `Optional[str]`       | `None`      | Sampling strategy: `"first_n_rows"`, `"stratified"`, or `"random"`. If `None`, derived from `task_type`: `"stratified"` for binary/multiclass, `"random"` for regression. |
| `label_column`  | `Optional[str]`          | `None`      | Name of the target/label column. Required when `sampling_method="stratified"`. |
| `task_type`     | `str`                    | `"regression"` | Machine learning task: `"binary"`, `"multiclass"`, or `"regression"`. Used when `sampling_method` is `None` to choose the strategy. |

### Sampling strategies

- **first_n_rows** — Read rows from the start of the file until the 1GB size limit.
- **stratified** — Iterate over all batches, merge with accumulated data, and subsample proportionally by `label_column` when over the limit. Requires `label_column`.
- **random** — Iterate over all batches, merge with accumulated data, and randomly subsample when over the limit.

### Credentials

S3 access uses environment variables (e.g. from a Kubernetes secret):

- `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` — required for S3.

## Outputs 📤

| Output          | Type     | Description |
| --------------- | -------- | ----------- |
| `full_dataset`  | `dsl.Dataset` | CSV artifact with the loaded (and optionally sampled) data. |
| Return value    | `NamedTuple`  | `sample_config`: dict with `n_samples` (number of rows written). |

## Usage Examples 💡

### Basic usage (default: sampling from task_type)

With default parameters, `sampling_method` is derived from `task_type` (e.g. regression → random sampling):

```python
from kfp import dsl
from kfp_components.components.data_processing.automl.tabular_data_loader import automl_data_loader

@dsl.pipeline(name="automl-training-pipeline")
def my_pipeline():
    load_task = automl_data_loader(
        bucket_name="my-ml-bucket",
        file_key="data/train.csv",
        label_column="target",
        task_type="regression",
    )
    return load_task
```

### Explicit sampling method

```python
load_task = automl_data_loader(
    bucket_name="my-ml-bucket",
    file_key="data/train.csv",
    full_dataset=...,
    sampling_method="first_n_rows",
)
```

### Stratified sampling (classification)

```python
load_task = automl_data_loader(
    bucket_name="my-ml-bucket",
    file_key="data/train.csv",
    full_dataset=...,
    sampling_method="stratified",
    label_column="target",
)
```

## Supported formats and limits 📋

- **Format**: CSV only.
- **Size limit**: Up to 1GB of data in memory (sampled if larger).
- **Streaming**: Data is read in batches (10k rows per chunk) to handle large files.

## Logging 📝

The component logs at INFO level:

- Which sampling method is used (including when derived from `task_type`).
- Number of rows read and the S3 location (`bucket_name`, `file_key`).

## Metadata 🗂️

- **Name**: tabular_data_loader
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow Pipelines >= 2.15.2
- **Tags**:
  - data-processing
  - automl
- **Last Verified**: 2026-02-23 14:30:00+00:00

## Additional resources 📚

- **AutoML Documentation**: [AutoML README](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/README.md)
- **Components Documentation**: [Components Structure](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/components.md)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
