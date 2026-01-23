# Document Loader 📄

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Reads unstructured data from data sources (S3, local filesystem) and performs document sampling.

The Document Loader component is the first step in the AutoRAG pipeline workflow. It loads documents
from various sources including S3 (via RHOAI Connections API) or local filesystem. The component
supports multiple document formats and performs document sampling based on test data to prepare a
subset of documents for processing. Document sampling functionality is integrated within this
component.

This component integrates with RHOAI Connections API for accessing documents from S3 or other
cloud storage systems. The component handles authentication and data retrieval transparently,
allowing users to specify data sources through connection IDs.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampled_documents` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the sampled documents. |
| `input_data_reference` | `dict` | `None` | Dictionary defining document data source. Required keys: `connection_id` (str), `bucket` (str), `path` (str). |
| `test_data` | `dsl.Input[dsl.Artifact]` | `None` | Optional input artifact containing test data for document sampling. |
| `sampling_config` | `dict` | `None` | Optional dictionary with sampling configuration. |

### Input Data Reference Structure

The `input_data_reference` dictionary should contain:

```python
{
    "connection_id": "s3-documents-connection",  # RHOAI Connection ID for S3 access
    "bucket": "my-documents-bucket",              # Bucket name containing the documents
    "path": "rh_documents/"                       # Path within bucket/filesystem to documents
}
```

### Sampling Configuration

The `sampling_config` dictionary supports test data driven sampling:

- Sample documents referenced in test data
- Add noise documents up to 1GB limit (in-memory)

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `sampled_documents` | `dsl.Artifact` | The sampled documents artifact ready for text extraction. |
| Return value | `str` | A message indicating the completion status of document loading. |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.data_processing.autorag.document_loader import document_loader

@dsl.pipeline(name="document-loading-pipeline")
def my_pipeline():
    """Example pipeline demonstrating document loading."""
    load_task = document_loader(
        input_data_reference={
            "connection_id": "s3-documents-connection",
            "bucket": "my-documents-bucket",
            "path": "rh_documents/"
        }
    )
    return load_task
```

### With Test Data Sampling

```python
@dsl.pipeline(name="document-loading-with-sampling-pipeline")
def my_pipeline(test_data):
    """Example pipeline with document sampling."""
    load_task = document_loader(
        input_data_reference={
            "connection_id": "s3-documents-connection",
            "bucket": "my-documents-bucket",
            "path": "rh_documents/"
        },
        test_data=test_data,
        sampling_config={
            "method": "test_data_driven",
            "max_size_gb": 1.0
        }
    )
    return load_task
```

## Supported Document Types 📋

- **PDF** (`.pdf`) - Portable Document Format
- **DOCX** (`.docx`) - Microsoft Word documents
- **PPTX** (`.pptx`) - Microsoft PowerPoint presentations
- **Markdown** (`.md`) - Markdown files
- **HTML** (`.html`) - HTML documents
- **Plain text** (`.txt`) - Text files

## Notes 📝

- **Document Sampling**: Sampling functionality is integrated within this component
- **Test Data Driven Sampling**: Samples documents referenced in test data and adds noise documents
  up to 1GB limit (in-memory)
- **Connection Management**: Uses RHOAI Connections API for secure access to S3 and other cloud
  storage systems
- **Format Support**: Automatically detects and handles multiple document formats

## Metadata 🗂️

- **Name**: document_loader
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: RHOAI Connections API, Version: >=1.0.0
    - Name: ai4rag, Version: >=1.0.0
- **Tags**:
  - data-processing
  - autorag
  - document-loading
- **Last Verified**: 2026-01-23 10:29:35+00:00

## Additional Resources 📚

- **AutoRAG Documentation**: See AutoRAG pipeline documentation for comprehensive information
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
