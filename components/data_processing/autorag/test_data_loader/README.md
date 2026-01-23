# Test Data Loader 📊

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Reads test data from JSON files using a DataLoader component.

The Test Data Loader component loads test data from JSON files for AutoRAG evaluation. It reads
test data from various sources including S3 (via RHOAI Connections API) or local filesystem and
returns the data as a pandas DataFrame. This test data is used for evaluating RAG configurations
during the optimization process.

The component supports JSON format only and is typically the first component executed in the
AutoRAG pipeline, as the test data is needed for document sampling in subsequent steps.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_data` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the loaded test data as a pandas DataFrame. |
| `test_data_reference` | `dict` | `None` | Dictionary defining test data source. Required keys: `connection_id` (str), `bucket` (str), `path` (str). |

### Test Data Reference Structure

The `test_data_reference` dictionary should contain:

```python
{
    "connection_id": "s3-benchmarks-connection",  # RHOAI Connection ID for S3 access
    "bucket": "autorag_benchmarks",              # Bucket name containing the test data
    "path": "my-folder/test_data.json"           # Path within bucket/filesystem to JSON file
}
```

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `test_data` | `dsl.Artifact` | The loaded test data as a pandas DataFrame artifact. |
| Return value | `str` | A message indicating the completion status of test data loading. |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.data_processing.autorag.test_data_loader import test_data_loader

@dsl.pipeline(name="test-data-loading-pipeline")
def my_pipeline():
    """Example pipeline demonstrating test data loading."""
    load_task = test_data_loader(
        test_data_reference={
            "connection_id": "s3-benchmarks-connection",
            "bucket": "autorag_benchmarks",
            "path": "my-folder/test_data.json"
        }
    )
    return load_task
```

### Local Filesystem

```python
@dsl.pipeline(name="local-test-data-loading-pipeline")
def my_pipeline():
    """Example pipeline loading from local filesystem."""
    load_task = test_data_loader(
        test_data_reference={
            "connection_id": None,
            "bucket": None,
            "path": "/local/path/to/test_data.json"
        }
    )
    return load_task
```

## Test Data Format 📋

The component expects test data in JSON format. The JSON file should contain test questions and
expected answers for RAG evaluation.

## Notes 📝

- **JSON Format Only**: Only JSON format is supported for test data files
- **DataFrame Output**: Returns data as a pandas DataFrame for easy processing
- **Early Execution**: Typically executed first in the pipeline as test data is needed for document
  sampling
- **Connection Management**: Uses RHOAI Connections API for secure access to S3 and other cloud
  storage systems

## Metadata 🗂️

- **Name**: test_data_loader
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: pandas, Version: >=2.0.0
- **Tags**:
  - data-processing
  - autorag
  - test-data
- **Last Verified**: 2026-01-23 00:00:00+00:00

## Additional Resources 📚

- **AutoRAG Documentation**: See AutoRAG pipeline documentation for comprehensive information
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
