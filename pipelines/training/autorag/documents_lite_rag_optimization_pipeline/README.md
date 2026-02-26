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
| `test_data_secret_name` | `str` | — | Kubernetes secret name for S3-compatible credentials (test data). Must provide: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION. |
| `test_data_bucket_name` | `str` | — | S3 (or compatible) bucket name for the test data JSON file. |
| `test_data_key` | `str` | — | Object key (path) of the test data JSON file in the test data bucket. |
| `input_data_secret_name` | `str` | — | Kubernetes secret name for S3-compatible credentials (input documents). Same env vars as above. |
| `input_data_bucket_name` | `str` | — | S3 (or compatible) bucket name for the input documents. |
| `input_data_key` | `str` | — | Object key (path) of the input documents in the input data bucket. |
| `llama_stack_secret_name` | `str` | — | Kubernetes secret name for llama-stack API connection. The secret must define: `LLAMA_STACK_CLIENT_API_KEY`, `LLAMA_STACK_CLIENT_BASE_URL`. |
| `embeddings_models` | `Optional[List]` | `None` | Optional list of embedding model identifiers for the search space. |
| `generation_models` | `Optional[List]` | `None` | Optional list of foundation/generation model identifiers for the search space. |
| `optimization_metric` | `str` | `"faithfulness"` | Metric to optimize. Supported: `faithfulness`, `answer_correctness`, `context_correctness`. |
| `vector_database_id` | `Optional[str]` | `None` | Optional vector database id (e.g. llama-stack Milvus). If not set, in-memory database may be used. |

## Stored artifacts (S3 / results storage) 📁

After pipeline execution, outputs are stored in the pipeline run's artifact location. Layout follows pipeline and component structure:

```text
<pipeline_name>/
└── <run_id>/
    ├── documents-sampling/
    │   └── <task_id>/
    │       └── sampled_documents_atrifact                     # YAML artifact containing sampled documents metadata
    ├── text-extraction/
    │   └── <task_id>/
    │       └── extracted_text_artifact           # Folder containing markdown files with extracted text
    ├── leaderboard-evaluation/
    │   └── <task_id>/
    │       └── html_artifact                     # HTML leaderboard (RAG pattern names + metrics); single file at path
    ├── autorag-run/
    │   └── <task_id>/
    │       └── run_artifact                      # Log and experiment status
    └── rag-templates-optimization/
        └── <task_id>/
            └── rag_patterns_artifact/
                ├── <pattern_name_0>/             # one per top-N RAG pattern
                │   ├── pattern.json              # Flat schema: name, iteration, settings, scores, final_score
                │   ├── evaluation_results.json   # Per-question evaluation (question, answer, correct_answers, scores, etc.)
                │   ├── indexing_notebook.ipynb   # Notebook to build/populate the vector index
                │   └── inference_notebook.ipynb  # Notebook for retrieval and generation
                ├── <pattern_name_1>/
                │   ├── pattern.json
                │   ├── evaluation_results.json
                │   ├── indexing_notebook.ipynb
                │   └── inference_notebook.ipynb
                └── ...
```

- `pipeline_name`: pipeline identifier (e.g. `documents-rag-optimization-pipeline`).
- `run_id`: Kubeflow Pipelines run ID.
- Component folders (`leaderboard-evaluation`, `rag-pattern-generation`, etc.) align with pipeline steps; `<task_id>` is the KFP task ID for that step.
- Pattern count and names depend on the run (e.g. `max_number_of_rag_patterns`).

### RAG pattern artifact schema (pattern.json and evaluation_results.json)

Each pattern directory under `rag_patterns_artifact/` contains:

- **pattern.json**:
  - **name**: pattern identifier.
  - **iteration**, **max_combinations**, **duration_seconds**: optimization run metadata.
  - **settings**: single object with **vector_store** (`datasource_type`, `collection_name`),
    **chunking** (`method`, `chunk_size`, `chunk_overlap`), **embedding** (`model_id`, `distance_metric`),
    **retrieval** (`method`, `number_of_chunks`), **generation** (`model_id`, `context_template_text`,
    `user_message_text`, `system_message_text`).
  - **scores**: object whose keys are metric names (e.g. `answer_correctness`, `faithfulness`,
    `context_correctness`); each value is `{ "mean", "ci_low", "ci_high" }`.
  - **final_score**: scalar optimization metric value.
- **evaluation_results.json** — List of per-question evaluation entries. Each entry has `question`,
  `correct_answers`, `answer`, `answer_contexts` (list of `{text, document_id}`), and `scores`
  (per-metric score for that question). Structure matches ai4rag
  `ExperimentResults.create_evaluation_results_json()`; a fallback is used when `question_scores` is
  missing or incomplete so the file is always valid.

**Sample pattern.json:**

```json
{
  "name": "Pattern_0",
  "iteration": 0,
  "max_combinations": 390,
  "duration_seconds": 100,
  "settings": {
    "vector_store": {
      "datasource_type": "ls_milvus",
      "collection_name": "collection0"
    },
    "chunking": {
      "method": "recursive",
      "chunk_size": 256,
      "chunk_overlap": 128
    },
    "embedding": {
      "model_id": "mock-embed-a",
      "distance_metric": "cosine"
    },
    "retrieval": {
      "method": "recursive",
      "number_of_chunks": 5
    },
    "generation": {
      "model_id": "mock-llm-1",
      "context_template_text": "{document}",
      "user_message_text": "<prompt template: context + question; answer in question language>",
      "system_message_text": "<system prompt: answer from context only; say if unanswerable>"
    }
  },
  "scores": {
    "answer_correctness": {
      "mean": 0.5,
      "ci_low": 0.4,
      "ci_high": 0.7
    },
    "faithfulness": {
      "mean": 0.2,
      "ci_low": 0.1,
      "ci_high": 0.5
    },
    "context_correctness": {
      "mean": 1.0,
      "ci_low": 0.9,
      "ci_high": 1.0
    }
  },
  "final_score": 0.5
}
```

**Sample evaluation_results.json:**

```json
[
  {
    "question": "What foundation models are available in watsonx.ai?",
    "correct_answers": ["The following models are available in watsonx.ai: flan-t5-xl-3b, ..."],
    "answer": "Watsonx.ai provides foundation models such as flan-t5-xl-3b, granite-13b-instruct-v2, and others.",
    "answer_contexts": [
      { "text": "Model architecture influences how the model behaves.", "document_id": "120CAE8361AE4E0B6FE4D6F0D32EEE9517F11190_1.txt" },
      { "text": "Learn more about governing assets in AI use cases.", "document_id": "0ECEAC44DA213D067B5B5EA66694E6283457A441_9.txt" }
    ],
    "scores": {
      "answer_correctness": 0.72,
      "faithfulness": 0.85,
      "context_correctness": 1.0
    }
  },
  {
    "question": "How can I ensure generated answers are accurate and factual?",
    "correct_answers": ["Utilize RAG, prompt engineering, and validate output."],
    "answer": "Use retrieval-augmented generation and validate the model output against your data.",
    "answer_contexts": [
      { "text": "Retrieval-augmented generation in IBM watsonx.ai.", "document_id": "752D982C2F694FFEE2A312CEA6ADF22C2384D4B2_0.txt" }
    ],
    "scores": {
      "answer_correctness": 0.65,
      "faithfulness": 0.91,
      "context_correctness": 0.8
    }
  }
]
```

## Example usage

```python

from kfp_components.pipelines.training.autorag.documents_rag_optimization_pipeline import (
    documents_rag_optimization_pipeline,
)


def example_minimal_usage():
    """Minimal example with only required parameters."""
    return documents_rag_optimization_pipeline(
        test_data_secret_name="s3-test-data-secret",
        test_data_bucket_name="autorag-benchmarks",
        test_data_key="test_data.json",
        input_data_secret_name="s3-input-secret",
        input_data_bucket_name="my-documents-bucket",
        input_data_key="rh_documents/",
        llama_stack_secret_name="llama-stack-secret",
    )


def example_full_usage():
    """Full example with optional parameters."""
    return documents_rag_optimization_pipeline(
        test_data_secret_name="s3-test-data-secret",
        test_data_bucket_name="autorag-benchmarks",
        test_data_key="my-folder/test_data.json",
        input_data_secret_name="s3-input-secret",
        input_data_bucket_name="my-documents-bucket",
        input_data_key="rh_documents/",
        llama_stack_secret_name="llama-stack-secret",
        embeddings_models=["ibm/slate-125m-english-rtrvr-v2", "intfloat/multilingual-e5-large"],
        generation_models=["mistralai/mixtral-8x7b-instruct-v01", "ibm/granite-13b-instruct-v2"],
        optimization_metric="answer_correctness",
        vector_database_id="milvus-database",
    )
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

## Required Parameters ✅

The following parameters are required to run the pipeline:

- `test_data_secret_name` - Kubernetes secret for S3 credentials (test data)
- `test_data_bucket_name` - Bucket containing the test data JSON file
- `test_data_key` - Object key to the test data JSON file
- `input_data_secret_name` - Kubernetes secret for S3 credentials (input documents)
- `input_data_bucket_name` - Bucket containing the input documents
- `input_data_key` - Object key to the input documents (folder or file)
- `llama_stack_secret_name` - Kubernetes secret for llama-stack API connection (must define `LLAMA_STACK_CLIENT_API_KEY`, `LLAMA_STACK_CLIENT_BASE_URL`)

Optional parameters (`embeddings_models`, `generation_models`, `optimization_metric`, `vector_database_id`) use defaults or search-space defaults when omitted.

## Components Used 🔧

This pipeline orchestrates the following AutoRAG components:

1. **[Test Data Loader](../components/data_processing/autorag/test_data_loader/README.md)** -
   Loads test data from JSON files

2. **[Documents sampling](../components/data_processing/autorag/documents_sampling/README.md)** -
   Loads documents from data sources and performs document sampling

3. **[Text Extraction](../components/data_processing/autorag/text_extraction/README.md)** -
   Extracts text from documents using docling library

4. **[Search Space Preparation](../components/training/autorag/search_space_preparation/README.md)** -
   Builds and validates RAG configuration search space

5. **[RAG Templates Optimization](../components/training/autorag/rag_templates_optimization/README.md)** -
   Core optimization component using GAM-based prediction

6. **[Leaderboard Evaluation](../components/training/autorag/leaderboard_evaluation/README.md)** -
   Builds an HTML leaderboard artifact from RAG pattern results (pattern names, settings, metrics)

## Artifacts 📦

For each pipeline run, the following artifacts are produced (see [Stored artifacts (S3 / results storage)](#stored-artifacts-s3--results-storage-) for the exact layout):

- **RAG Patterns Artifact** (`rag_patterns_artifact`): A directory containing one subdirectory per
  top-N RAG pattern (named by pattern). Each subdirectory includes:
  - **pattern.json** — Flat schema with `name`, `iteration`, `max_combinations`, `duration_seconds`,
    `settings` (vector_store, chunking, embedding, retrieval, generation), `scores`
    (per-metric mean/ci_low/ci_high), and `final_score`
  - **evaluation_results.json** — Per-question evaluation (question, answer, correct_answers,
    answer_contexts, scores)
  - **indexing_notebook.ipynb** — Notebook for building or populating the vector index/collection
  - **inference_notebook.ipynb** — Notebook for retrieval and generation
- **Leaderboard HTML Artifact** (`html_artifact`): HTML leaderboard table of RAG patterns ranked by
  `final_score`, with pattern name, metrics and config columns (chunking, embedding, retrieval,
  generation)

`rag_patterns_artifact` metadata:

```json
{
   "name":"rag_patterns_artifact",
   "uri":"<pipeline_name>/<run_id>/rag-templates-optimization/<task_id>/rag_patterns/",
   "metadata":{
      "patterns":[
         {
            "name":"pattern0",
            "iteration":0,
            "max_combinations":3,
            "duration_seconds":20,
            "location": {
               "evaluation_results": "pattern0/evaluation_results.json",
               "indexing_notebook": "pattern0/indexing_notebook.ipynb",  
               "inference_notebook": "pattern0/inference_notebook.ipynb",
               "pattern_descriptor": "pattern0/pattern.json"
            },  
            "settings":{
               "vector_store":{
                  "datasource_type":"ls_milvus",
                  "collection_name":"collection0"
               },
               "chunking":{
                  "method":"recursive",
                  "chunk_size":256,
                  "chunk_overlap":128
               },
               "embedding":{
                  "model_id":"mock-embed-a",
                  "distance_metric":"cosine"
               },
               "retrieval":{
                  "method":"window",
                  "number_of_chunks":5
               },
               "generation":{
                  "model_id":"mock-llm-1",
                  "context_template_text":"{document}",
                  "user_message_text":"<prompt: context + question; answer in question language>",
                  "system_message_text":"<system: answer from context only; say if unanswerable>"
               }
            },
            "scores":{
               "answer_correctness":{
                  "mean":0.5,
                  "ci_low":0.4,
                  "ci_high":0.7
               },
               "faithfulness":{
                  "mean":0.2,
                  "ci_low":0.1,
                  "ci_high":0.5
               },
               "context_correctness":{
                  "mean":1.0,
                  "ci_low":0.9,
                  "ci_high":1.0
               }
            },
            "final_score":0.5
         },
         {
            "name":"pattern1",
            "iteration":2,
            "max_combinations":3,
            "duration_seconds":10,
            "location": {
               "evaluation_results": "pattern1/evaluation_results.json",
               "indexing_notebook": "pattern1/indexing_notebook.ipynb",  
               "inference_notebook": "pattern1/inference_notebook.ipynb",
               "pattern_descriptor": "pattern1/pattern.json"
            },  
            "settings":{
               "vector_store":{
                  "datasource_type":"ls_milvus",
                  "collection_name":"collection0"
               },
               "chunking":{
                  "method":"recursive",
                  "chunk_size":256,
                  "chunk_overlap":128
               },
               "embedding":{
                  "model_id":"mock-embed-a",
                  "distance_metric":"cosine"
               },
               "retrieval":{
                  "method":"window",
                  "number_of_chunks":5
               },
               "generation":{
                  "model_id":"mock-llm-1",
                  "context_template_text":"{document}",
                  "user_message_text":"<prompt: context + question; answer in question language>",
                  "system_message_text":"<system: answer from context only; say if unanswerable>"
               }
            },
            "scores":{
               "answer_correctness":{
                  "mean":0.5,
                  "ci_low":0.4,
                  "ci_high":0.7
               },
               "faithfulness":{
                  "mean":0.2,
                  "ci_low":0.1,
                  "ci_high":0.5
               },
               "context_correctness":{
                  "mean":1.0,
                  "ci_low":0.9,
                  "ci_high":1.0
               }
            },
            "final_score":0.5
         }
      ]
   }
}
```

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
