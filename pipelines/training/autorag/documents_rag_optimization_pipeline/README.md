# Documents Rag Optimization Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications.

The Documents RAG Optimization Pipeline is an automated system for building and optimizing Retrieval-Augmented
Generation (RAG) applications within Red Hat OpenShift AI. It leverages Kubeflow Pipelines to orchestrate the
optimization workflow, using the ai4rag optimization engine to systematically explore RAG configurations and identify
the best performing parameter settings based on an upfront-specified quality metric.

The system integrates with llama-stack API for inference and vector database operations, producing optimized RAG
Patterns as artifacts that can be deployed and used for production RAG applications. It can also communicate with
externally provided MLFlow server to support advanced experiment tracking features.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `None` | Name of the AutoRAG experiment run (e.g., "AutoRAG run"). |
| `input_data_reference` | `Dict` | `None` | Dictionary defining document data source with keys: connection_id,
bucket, path. |
| `test_data_reference` | `Dict` | `None` | Dictionary defining test data source with keys: connection_id, bucket,
path. Test data JSON file is supported only. |
| `results_reference` | `Dict` | `None` | Dictionary defining results storage location with keys: connection_id,
bucket, path. |
| `description` | `Optional[str]` | `None` | Optional description of the experiment (e.g., "RHOAI Kubeflow Pipelines Docs"). |
| `vector_database_id` | `Optional[str]` | `None` | Optional vector database id (e.g., registered in llama-stack Milvus
database). If not provided, an in-memory database will be used. |
| `mlflow_config` | `Optional[Dict]` | `None` | Optional dictionary defining MLFlow configuration for experiment tracking with
keys: tracking_uri, experiment_name, enabled. |
| `optimization` | `Optional[Dict]` | `None` | Optional dictionary defining optimization settings with keys:
max_number_of_rag_patterns (int), metric (str). Supported metrics: faithfulness,
answer_correctness. |
| `chunking_constraints` | `Optional[List[Dict]]` | `None` | Optional list of dictionaries defining chunking configurations. Each
dictionary contains: method (str), chunk_overlap (int), chunk_size (int). |
| `embeddings_constraints` | `Optional[List[Dict]]` | `None` | Optional list of dictionaries defining embedding models. Each
dictionary contains: model (str). |
| `generation_constraints` | `Optional[List[Dict]]` | `None` | Optional list of dictionaries defining generation models. Each
dictionary contains: model (str), optional context_template_text (str), optional
messages (list[dict]). |
| `retrieval_constraints` | `Optional[List[Dict]]` | `None` | Optional list of dictionaries defining retrieval method
configurations. Each dictionary contains: method (str), number_of_chunks (int),
optional hybrid_ranker (dict). |

## Stored artifacts (S3 / results storage) 📁

After pipeline execution, outputs are stored under the location defined by `results_reference` (bucket and path). Typical layout:

```
<results_reference.bucket> / <results_reference.path>/
├── leaderboard                    # Leaderboard HtML artifact (RAG patterns ranked by metric)
├── autorag_run                    # Run artifact (log and experiment status)
└── rag_patterns/                  # Directory of generated RAG patterns
    ├── <pattern_name_0>/
    │   ├── pattern.json           # Pattern config, params, and evaluation metrics
    │   ├── indexing_notebook.ipynb # Notebook to build/populate the vector index
    │   └── inference_notebook.ipynb # Notebook for retrieval and generation
    ├── <pattern_name_1>/
    │   ├── pattern.json
    │   ├── indexing_notebook.ipynb
    │   └── inference_notebook.ipynb
    └── ...
```

Each RAG pattern folder corresponds to one optimized configuration; pattern names and count depend on the run (e.g. `max_number_of_rag_patterns`).


## Usage Examples 🧪

```python
"""Example usage of the documents_rag_optimization_pipeline."""

from kfp import dsl
from kfp_components.pipelines.training.documents_rag_optimization_pipeline import (
    documents_rag_optimization_pipeline,
)


def example_minimal_usage():
    """Minimal example with only required parameters."""
    # Define data references
    input_data_reference = {
        "connection_id": "s3-documents-connection",
        "bucket": "my-documents-bucket",
        "path": "rh_documents/",
    }

    test_data_reference = {
        "connection_id": "s3-benchmarks-connection",
        "bucket": "autorag_benchmarks",
        "path": "my-folder/test_data.json",
    }

    results_reference = {
        "connection_id": "s3-autorag-results-connection",
        "bucket": "results",
        "path": "autorag/",
    }

    # Create pipeline run
    run = documents_rag_optimization_pipeline(
        name="AutoRAG Experiment 2",
        input_data_reference=input_data_reference,
        test_data_reference=test_data_reference,
        results_reference=results_reference,
    )
    return run


def example_full_usage():
    """Full example with all optional parameters."""
    # Define data references
    input_data_reference = {
        "connection_id": "s3-documents-connection",
        "bucket": "my-documents-bucket",
        "path": "rh_documents/",
    }

    test_data_reference = {
        "connection_id": "s3-benchmarks-connection",
        "bucket": "autorag_benchmarks",
        "path": "my-folder/test_data.json",
    }

    results_reference = {
        "connection_id": "s3-autorag-results-connection",
        "bucket": "results",
        "path": "autorag/",
    }

    # Optional MLFlow configuration
    mlflow_config = {
        "tracking_uri": "http://mlflow-server.redhat-ods-applications.svc.cluster.local:5000",
        "experiment_name": "AutoRAG Experiments",
        "enabled": True,
    }

    # Optional optimization settings
    optimization = {
        "max_number_of_rag_patterns": 4,
        "metric": "answer_correctness",
    }

    # Optional constraints
    chunking_constraints = [
        {
            "method": "recursive",
            "chunk_overlap": 256,
            "chunk_size": 2048,
        }
    ]

    embeddings_constraints = [
        {"model": "ibm/slate-125m-english-rtrvr-v2"},
        {"model": "intfloat/multilingual-e5-large"},
    ]

    generation_constraints = [
        {"model": "mistralai/mixtral-8x7b-instruct-v01"},
        {"model": "ibm/granite-13b-instruct-v2"},
    ]

    retrieval_constraints = [
        {
            "method": "simple",
            "number_of_chunks": 2,
            "hybrid_ranker": {
                "strategy": "weighted",
                "alpha": 0.6,
            },
        }
    ]

    # Create pipeline run
    run = documents_rag_optimization_pipeline(
        name="AutoRAG Experiment 1",
        description="RHOAI Kubeflow Pipelines Docs",
        input_data_reference=input_data_reference,
        test_data_reference=test_data_reference,
        vector_database_id="milvus-database",
        results_reference=results_reference,
        mlflow_config=mlflow_config,
        optimization=optimization,
        chunking_constraints=chunking_constraints,
        embeddings_constraints=embeddings_constraints,
        generation_constraints=generation_constraints,
        retrieval_constraints=retrieval_constraints,
    )
    return run

```

## Metadata 🗂️

- **Name**: documents_rag_optimization_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: >=1.0.0
    - Name: llama-stack API, Version: >=1.0.0
    - Name: RHOAI Connections API, Version: >=1.0.0
    - Name: Milvus, Version: >=2.0.0
    - Name: Milvus Lite, Version: >=2.0.0
    - Name: MLFlow, Version: >=2.0.0
    - Name: docling, Version: >=1.0.0
- **Tags**:
  - training
  - pipeline
  - autorag
  - rag-optimization
- **Last Verified**: 2026-01-23 14:57:32+00:00
- **Owners**:
  - Approvers: None
  - Reviewers: None


<!-- custom-content -->

## Pipeline Workflow 🔄

The optimization process involves the following stages:

1. **Test Data Loading**: Loads test data from JSON files for evaluation
2. **Document Loading & Sampling**: Loads documents from data sources and samples a subset based
   on test data
3. **Text Extraction**: Extracts text from sampled documents using the docling library
4. **Search Space Preparation**: Builds and validates the search space of RAG configurations,
   including model preselection and validation using in-memory vector databases
5. **RAG Templates Optimization**: Systematically tests different RAG configurations from the
   defined search space using GAM-based prediction
6. **Evaluation**: Assesses each configuration's performance using test data
7. **Pattern Generation**: Produces artifacts including RAG Patterns, associated metrics, logs and
   notebooks
8. **Leaderboard**: Maintains a leaderboard of RAG Patterns ranked by performance

## Input Parameters Organization 📋

The pipeline parameters are organized into the following logical groups:

> 📘 **Note:** This documentation uses Kubeflow Pipelines v2 structured types (`Dict`, `List`)
> for complex parameters.

### 1. Experiment Metadata

**Required Parameters:**

- `name: str` - Name of the AutoRAG experiment run (e.g., "AutoRAG run")

**Optional Parameters:**

- `description: str` - Description of the experiment (e.g., "RHOAI Kubeflow Pipelines Docs")

### 2. Input Data Sources

#### Document Data

**Required Parameter:**

- `input_data_reference: Dict` - Dictionary defining document data source:
  - `connection_id: str` - Connection ID for the data source (e.g., S3 connection ID)
  - `bucket: str` - Bucket name containing the documents
  - `path: str` - Path within the bucket/filesystem to the documents folder or single file

#### Test Data

Test data JSON file is supported only.

**Required Parameter:**

- `test_data_reference: Dict` - Dictionary defining test data source:
  - `connection_id: str` - Connection ID for the test data source (e.g., S3 connection ID)
  - `bucket: str` - Bucket name containing the test data file
  - `path: str` - Path within the bucket/filesystem to the test data file

### 3. Infrastructure Configuration

#### Vector Database

**Optional Parameter:**

- `vector_database_id: str` - Vector database id (e.g., registered in llama-stack Milvus database).
  If not provided, an in-memory database will be used.

#### Output Results Storage

**Required Parameter:**

- `results_reference: Dict` - Dictionary defining results storage location:
  - `connection_id: str` - Connection ID for the results storage (e.g., S3 connection ID)
  - `bucket: str` - Bucket name for storing results
  - `path: str` - Path where experiment results will be stored (e.g., "autorag/results")

#### MLFlow Integration (Experiment Tracking)

**Optional Parameter:**

- `mlflow_config: Dict` - Dictionary defining MLFlow configuration for experiment tracking:
  - `tracking_uri: str` - MLFlow tracking server URI (e.g., "http://mlflow-server:5000")
  - `experiment_name: str` - MLFlow experiment name (default: uses pipeline `name` parameter)
  - `enabled: bool` - Enable/disable MLFlow tracking (default: `True` if `mlflow_config` is
    provided)

When enabled, AutoRAG will automatically log:

- Experiment metadata (name, description, run parameters)
- Optimization metrics (answer_correctness, faithfulness, context_correctness)
- Configuration parameters (chunking, embeddings, generation, retrieval settings)
- Leaderboard rankings and best pattern information
- Artifact references (RAG Pattern artifacts, summary reports)
- Execution timestamps and duration

> 💡 **Note:** If `mlflow_config` is not provided, MLFlow tracking will be disabled. To use
> MLFlow, ensure the MLFlow server is accessible from the pipeline execution environment.

### 4. Optimization Configuration

#### Optimization Settings

**Optional Parameter:**

- `optimization: Dict` - Dictionary defining optimization settings:
  - `max_number_of_rag_patterns: int` - Maximum number of RAG patterns to generate (default: 4)
  - `metric: str` - Metric to optimize (e.g., `"answer_correctness"` or `"faithfulness"`)

Supported metrics are: `faithfulness` and `answer_correctness`. On top of those the
`context_correctness` is automatically calculated measuring the retrieved chunks quality.

#### Search Space Constraints

Constraints define the search space for RAG optimization. Each constraint section is provided as a
list parameter:

**Optional Parameters:**

**Chunking Constraints:**

- `chunking_constraints: List[Dict]` - List of dictionaries defining chunking configurations:
  - Each dictionary contains: `method: str`, `chunk_overlap: int`, `chunk_size: int`

**Embeddings Constraints:**

- `embeddings_constraints: List[Dict]` - List of dictionaries defining embedding models:
  - Each dictionary contains: `model: str`

**Generation Constraints:**

- `generation_constraints: List[Dict]` - List of dictionaries defining generation models:
  - Each dictionary contains: `model: str`, optional `context_template_text: str`, optional
    `messages: List[Dict]` array

**Retrieval Constraints:**

- `retrieval_constraints: List[Dict]` - List of dictionaries defining retrieval method
  configurations:
  - Each dictionary contains: `method: str`, `number_of_chunks: int`, optional `hybrid_ranker:
    Dict` (with `strategy: str`, `sparse_vectors: str`, `alpha: float`, `k: int`)

## Required Parameters ✅

The following parameters are required to run the pipeline:

- `name: str` - Experiment name
- `input_data_reference: Dict` - Document data source
- `test_data_reference: Dict` - Test data source
- `results_reference: Dict` - Results storage location

> 💡 **Note:** When optional parameters are omitted, AutoRAG uses default values or explores the
> full available search space.

## Components Used 🔧

This pipeline orchestrates the following AutoRAG components:

1. **[Test Data Loader](../components/data_processing/autorag/test_data_loader/README.md)** -
   Loads test data from JSON files

2. **[Document Loader](../components/data_processing/autorag/document_loader/README.md)** -
   Loads documents from data sources and performs document sampling

3. **[Text Extraction](../components/data_processing/autorag/text_extraction/README.md)** -
   Extracts text from documents using docling library

4. **[Search Space Preparation](../components/training/autorag/search_space_preparation/README.md)** -
   Builds and validates RAG configuration search space

5. **[RAG Templates Optimization](../components/training/autorag/rag_templates_optimization/README.md)** -
   Core optimization component using GAM-based prediction

## Artifacts 📦

For each pipeline run, AutoRAG generates:

- **RAG Pattern Artifact(s)**: Multiple RAG Patterns, each consisting of properties and URI to
  tar archive with notebooks:
  - **Index building notebook**: For building the vector index/collection (in-memory database) or
    populating existing index/collection (persistent database) with all user documents
  - **Retrieval/generation notebook**: For performing retrieval and generation operations
- **AutoRAG Run Artifact**: Run status properties and URI to log file with messages
- **AutoRAG Experiment Summary Markdown Artifact**: Experiment run report including:
  - Data preparation details
  - Search space and explored configurations
  - Leaderboard of RAG Patterns ranked by performance
  - Links to remaining Artifacts

## Optimization Engine: ai4rag 🚀

The pipeline uses [ai4rag](https://github.com/IBM/ai4rag), a RAG Templates Optimization Engine that
provides an automated approach to optimizing Retrieval-Augmented Generation (RAG) systems. The
engine is designed to be LLM and Vector Database provider agnostic, making it flexible and
adaptable to various RAG implementations.

ai4rag accepts a variety of RAG templates and search space definitions, then systematically
explores different parameter configurations to find optimal settings. The engine returns initialized
RAG templates with optimal parameter values, which are referred to as RAG Patterns.

## Supported Features ✨

**Status**: Tech Preview - MVP (May 2026)

### RAG Configuration

- **RAG Type**: Documents (documents provided as input)
- **Supported Languages**: English
- **Supported Document Types**: PDF, DOCX, PPTX, Markdown, HTML, Plain text
- **Document Data Sources**: S3 (Amazon S3), Local filesystem (FS)

### Infrastructure Components

- **Vector Databases**: Milvus, Milvus Lite
- **LLM Provider**: Llama-stack-supported models and vendors
- **Experiment Tracking**: MLFlow (optional) - For experiment tracking, metrics logging, and
  artifact management

### Processing Methods

- **Chunking Method**: Recursive
- **Retrieval Methods**: Simple, Simple with hybrid ranker

### Interfaces

- **API**: Programmatic access to AutoRAG functionality
- **UI**: User interface for interacting with AutoRAG

## Glossary 📚

### RAG Configuration Definition

A **RAG Configuration** is a specific set of parameter values that define how a
Retrieval-Augmented Generation system operates. It includes settings for:

- **Chunking**: Method and parameters for splitting documents (e.g., recursive method with
  chunk_size=2048, chunk_overlap=256)
- **Embeddings**: The embedding model used (e.g., `intfloat/multilingual-e5-large`)
- **Generation**: The language model used (e.g., `ibm/granite-13b-instruct-v2`) along with its
  parameters
- **Retrieval**: The method for retrieving relevant document chunks (e.g., simple retrieval or
  hybrid ranker)

### RAG Pattern

A **RAG Pattern** is an optimized RAG configuration that has been evaluated and ranked by
AutoRAG. It represents a complete, deployable RAG system with:

- Validated parameter settings that have been tested and evaluated
- Performance metrics (e.g., answer_correctness, faithfulness, context_correctness)
- Executable notebooks for indexing and inference operations
- A position in the leaderboard based on performance

### RAG Template

A **RAG Template** is a reusable blueprint that defines the structure and workflow of a RAG
system. Templates are parameterized and AutoRAG uses templates as the foundation, optimizing the
parameter values to create RAG Patterns.

## Additional Resources 📚

- **ai4rag Documentation**: [ai4rag GitHub](https://github.com/IBM/ai4rag)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
