# Text Extraction 📝

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Extracts text from provided documents using the docling library.

The Text Extraction component processes documents loaded by the document-loader component and
extracts text content using the `docling` library. It handles multiple document formats including
PDF, DOCX, PPTX, Markdown, HTML, and plain text files. The extracted text is prepared for
subsequent processing steps in the AutoRAG pipeline, including chunking and embedding generation.

This component is a critical step in the AutoRAG workflow as it converts unstructured documents
into structured text that can be processed by downstream components.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extracted_text` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the extracted text content. |
| `documents` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact containing the documents from document-loader. |

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `extracted_text` | `dsl.Artifact` | The extracted text content artifact ready for chunking and embedding. |
| Return value | `str` | A message indicating the completion status of text extraction. |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.data_processing.autorag.text_extraction import text_extraction

@dsl.pipeline(name="text-extraction-pipeline")
def my_pipeline(sampled_documents):
    """Example pipeline demonstrating text extraction."""
    extract_task = text_extraction(
        documents=sampled_documents
    )
    return extract_task
```

### In AutoRAG Pipeline Context

```python
@dsl.pipeline(name="autorag-pipeline")
def autorag_pipeline(sampled_documents):
    """Example AutoRAG pipeline with text extraction."""
    # Extract text from sampled documents
    extracted_text = text_extraction(
        documents=sampled_documents
    )
    
    # Continue with search space preparation and optimization
    # ...
    
    return extracted_text
```

## Supported Document Formats 📋

The component supports extraction from the following document formats:

- **PDF** (`.pdf`) - Portable Document Format
- **DOCX** (`.docx`) - Microsoft Word documents
- **PPTX** (`.pptx`) - Microsoft PowerPoint presentations
- **Markdown** (`.md`) - Markdown files
- **HTML** (`.html`) - HTML documents
- **Plain text** (`.txt`) - Text files

## Docling Library 🔧

The component uses the `docling` library for text extraction. Docling provides:

- High-quality text extraction from various document formats
- Preservation of document structure and formatting where applicable
- Support for complex document layouts
- Robust handling of different document types

## Notes 📝

- **Format Support**: Handles multiple document formats automatically
- **Text Quality**: Uses docling library for high-quality text extraction
- **Pipeline Integration**: Works seamlessly with document-loader output
- **Structured Output**: Extracted text is formatted for downstream processing (chunking, embedding)

## Metadata 🗂️

- **Name**: text_extraction
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: docling, Version: >=1.0.0
- **Tags**:
  - data-processing
  - autorag
  - text-extraction
- **Last Verified**: 2026-01-23 00:00:00+00:00

## Additional Resources 📚

- **AutoRAG Documentation**: See AutoRAG pipeline documentation for comprehensive information
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
