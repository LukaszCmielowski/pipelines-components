# Data Loader 📊

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Reads tabular data from data sources (S3, local filesystem) for AutoML processing.

The Data Loader component is the first step in the AutoML pipeline workflow. It loads tabular data from
various sources including S3 (via RHOAI Connections API) or local filesystem. The component supports
multiple data formats including CSV, Parquet, and XLSX. It returns tabular data artifacts that are
used by subsequent components in the AutoML pipeline.

This component integrates with RHOAI Connections API for accessing data from S3 or other cloud storage systems. The component handles authentication and data retrieval transparently, allowing users to specify data sources through connection IDs.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `tabular_data` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset artifact containing the loaded training data. |
| `input_data_reference` | `dict` | `None` | Dict with `connection_id`, `bucket`, `path`; optional `format`. |
| `test_data` | `dsl.Output[dsl.Dataset]` | `None` | Optional output dataset artifact for test data. |
| `test_data_reference` | `dict` | `None` | Optional dict defining test data source; same structure as `input_data_reference`. |

### Input Data Reference Structure

The `input_data_reference` dictionary should contain:

```python
{
    "connection_id": "s3-data-connection",  # RHOAI Connection ID for S3 access
    "bucket": "my-ml-data-bucket",          # Bucket name containing the data
    "path": "tabular_data/train.csv"        # Path within bucket/filesystem to data file
}
```

For local filesystem:

```python
{
    "connection_id": None,                  # Not required for local filesystem
    "bucket": None,                         # Not required for local filesystem
    "path": "/local/path/to/data.csv"       # Local filesystem path
}
```

## Outputs 📤

| Output | Type | Description |
| ------ | ---- | ----------- |
| `tabular_data` | `dsl.Dataset` | Loaded training dataset artifact (data file and metadata). |
| `test_data` | `dsl.Dataset` | Optional test dataset artifact if `test_data_reference` is provided. |
| Return value | `str` | Message indicating the completion status of data loading. |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.training.automl.data_processing.data_loader import data_loader

@dsl.pipeline(name="data-loading-pipeline")
def my_pipeline():
    """Example pipeline demonstrating data loading."""
    load_task = data_loader(
        input_data_reference={
            "connection_id": "s3-data-connection",
            "bucket": "my-ml-data-bucket",
            "path": "tabular_data/train.csv"
        }
    )
    return load_task
```

### With Test Data

```python
@dsl.pipeline(name="data-loading-with-test-pipeline")
def my_pipeline():
    """Example pipeline with both training and test data."""
    load_task = data_loader(
        input_data_reference={
            "connection_id": "s3-data-connection",
            "bucket": "my-ml-data-bucket",
            "path": "tabular_data/train.csv"
        },
        test_data_reference={
            "connection_id": "s3-data-connection",
            "bucket": "my-ml-data-bucket",
            "path": "tabular_data/test.csv"
        }
    )
    return load_task
```

### Local Filesystem

```python
@dsl.pipeline(name="local-data-loading-pipeline")
def my_pipeline():
    """Example pipeline loading from local filesystem."""
    load_task = data_loader(
        input_data_reference={
            "connection_id": None,
            "bucket": None,
            "path": "/local/path/to/data.csv"
        }
    )
    return load_task
```

## Supported Formats 📋

- **CSV** (`.csv`) - Comma-separated values
- **Parquet** (`.parquet`) - Columnar storage format
- **XLSX** (`.xlsx`) - Excel format

## Notes 📝

- **Torch Compatible Data Loader**: Torch compatible data loader to be explored in future versions
- **Data Cache Support**: Data cache support to be explored in next stage
- **Connection Management**: Uses RHOAI Connections API for secure access to S3 and other cloud storage systems
- **Format Detection**: Data format is automatically detected from file extension if not explicitly specified

## Metadata 🗂️

- **Name**: data-loader
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: RHOAI Connections API, Version: >=1.0.0
    - Name: pandas, Version: >=2.0.0
    - Name: pyarrow, Version: >=14.0.0
- **Tags**:
  - automl
  - data-processing
  - tabular-data
  - data-loading
- **Last Verified**: 2025-01-27 00:00:00+00:00

## Additional Resources 📚

- **AutoML Documentation**: [AutoML README](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/README.md)
- **Components Documentation**: [Components Structure](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/components.md)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
