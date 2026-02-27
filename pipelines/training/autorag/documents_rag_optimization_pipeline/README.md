# Documents Rag Optimization Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications as part of Red Hat OpenShift AI.

Documents Rag Optimization Pipeline leverages Kubeflow Pipelines to orchestrate the experiment workflow, 
using the ai4rag optimization engine to systematically explore RAG configurations and 
identify the best performing parameter settings based on an upfront-specified quality metric.

The system integrates with llama-stack API for model inference and vector database operations, producing optimized RAG
Patterns as artifacts that can be deployed and used for production-grade RAG applications. It can also communicate with
externally provided MLFlow server to support advanced experiment tracking features.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_data_secret_name` | `str` | — | Kubernetes secret's name for S3-compatible storage credentials holding test data. Must include: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION. |
| `test_data_bucket_name` | `str` | — | Name of the S3-like bucket holding test data. |
| `test_data_key` | `str` | — | S3-like object key (a path) identifying the test data (a JSON file)|
| `input_data_secret_name` | `str` | — | Kubernetes secret's name for S3-compatible storage credentials holding input documents. Must include the same set of keys as `test_data_secret_name`. |
| `input_data_bucket_name` | `str` | — | Name of the S3-like bucket holding input documents. |
| `input_data_key` | `str` | — | S3-like object key (a path) identifying the input data (file or a folder of thereof). |
| `llama_stack_secret_name` | `str` | — | Kubernetes secret's name for llama-stack server connection info. The secret must include: `LLAMA_STACK_CLIENT_API_KEY`, `LLAMA_STACK_CLIENT_BASE_URL`. |
| `embeddings_models` | `List` | - | List of embedding model identifiers to try out in the experiment process. |
| `generation_models` | `List` | - | List of foundation/generation model identifiers to try out in the experiment process. |
| `optimization_metric` | `str` | `"faithfulness"` | Metric to optimize for. Supported: `faithfulness`, `answer_correctness`, `context_correctness`. |
| `llama_stack_vector_database_id` | `Optional[str]` | `ls_milvus` | vector database identifier as registered in llama-stack (e.g. llama-stack Milvus). |

## Stored artifacts (S3 / results storage) 📁

Any outputs from the pipeline execution are stored in the predefined artifact location (can set for whole KP server or for each pipeline separately). Below layout presents the pipeline and component structure:

```
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
    └── rag-templates-optimization/
        └── <task_id>/
            └── autorag_run_artifact              # Log and experiment status
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

- **pattern.json**: Collection of pattern details and related metadata.  
- **evaluation_results.json** — Structure matches ai4rag `ExperimentResults.create_evaluation_results_json()`; a fallback is used when `question_scores` is missing or incomplete so the file is always valid.

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
      "user_message_text": "\n\nContext:\n{reference_documents}:\n\nQuestion: {question}. \nAgain, please answer the question based on the context provided only. If the context is not related to the question, just say you cannot answer. Respond exclusively in the language of the question, regardless of any other language used in the provided context. Ensure that your entire response is in the same language as the question.",
      "system_message_text": "Please answer the question I provide in the Question section below, based solely on the information I provide in the Context section. If the question is unanswerable, please say you cannot answer."
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


def my_pipeline():
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
        llama_stack_vector_database_id="milvus-database",
    )
```

<!-- custom-content -->

## Pipeline Workflow 🔄

The optimization process involves the following stages:

1. **Test Data Loading**: Loads test data from JSON files for evaluation
2. **Document Loading & Sampling**: Loads documents from data sources and chooses a subset based
   on the test data
3. **Text Extraction**: Extracts text from sampled documents using the docling library
4. **Search Space Preparation**: Builds the search space defining RAG configurations to try out in the experiment process. Limits the foundation and embedding models to 3 and 2 (respectively)
5. **RAG Templates Optimization**: Systematically tests different RAG configurations from the
   defined search space using GAM-based prediction
6. **Evaluation**: Assesses each configuration's performance using test data
7. **Pattern Generation**: Produces artifacts including RAG Patterns, associated metrics, logs and
   notebooks
8. **Leaderboard**: Builds a leaderboard of RAG Patterns ranked by performance


## Components of the pipeline 🔧

This pipeline orchestrates the following AutoRAG components:

1. **[Test Data Loader](../components/data_processing/autorag/test_data_loader/README.md)** -
   Loads test data from specified S3-compatible data sources

2. **[Documents sampling](../components/data_processing/autorag/documents_sampling/README.md)** -
   Loads and samples documents from specified S3-compatible data sources

3. **[Text Extraction](../components/data_processing/autorag/text_extraction/README.md)** -
   Extracts text from documents using docling library

4. **[Search Space Preparation](../components/training/autorag/search_space_preparation/README.md)** -
   Builds RAG configuration search space

5. **[RAG Templates Optimization](../components/training/autorag/rag_templates_optimization/README.md)** -
   Core optimization component using GAM-based prediction

6. **[Leaderboard Evaluation](../components/training/autorag/leaderboard_evaluation/README.md)** -
   Builds a HTML leaderboard artifact from RAG pattern results (pattern names, settings, metrics)

## Artifacts 📦

For each pipeline run, the following artifacts are produced (see [Stored artifacts (S3 / results storage)](#stored-artifacts-s3--results-storage-) for the exact layout):

- **RAG Patterns Artifact** (`rag_patterns_artifact`): A directory containing one subdirectory per top-N RAG pattern (named by pattern). Each subdirectory includes:
  - **pattern.json** — Flat schema with `name`, `iteration`, `max_combinations`, `duration_seconds`, `settings` (vector_store, chunking, embedding, retrieval, generation), `scores` (per-metric mean/ci_low/ci_high), and `final_score`
  - **evaluation_results.json** — Per-question evaluation (question, answer, correct_answers, answer_contexts, scores)
  - **indexing_notebook.ipynb** — Notebook for building or populating the vector index/collection
  - **inference_notebook.ipynb** — Notebook for retrieval and generation
- **Leaderboard HTML Artifact** (`html_artifact`): HTML leaderboard table of RAG patterns ranked by `final_score`, with pattern name, metrics (e.g. mean_answer_correctness, mean_faithfulness, mean_context_correctness), and config columns (chunking, embedding, retrieval, generation)

`rag_patterns_artifact` metadata:
```json
{
   "name":"rag_patterns_artifact",
   "uri":"documents-rag-optimization-pipeline/b888dca4-11de-49f5-8b60-e820613a623d/rag-templates-optimization/29a02588-7bed-4bf2-a595-66ae503440d3/rag_patterns/",
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
                  "user_message_text":"\n\nContext:\n{reference_documents}:\n\nQuestion: {question}. \nAgain, please answer the question based on the context provided only. If the context is not related to the question, just say you cannot answer. Respond exclusively in the language of the question, regardless of any other language used in the provided context. Ensure that your entire response is in the same language as the question.",
                  "system_message_text":"Please answer the question I provide in the Question section below, based solely on the information I provide in the Context section. If the question is unanswerable, please say you cannot answer."
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

- **Vector Databases**: Milvus, Milvus Lite, ChromaDB
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
- **Last Verified**: 2026-01-23 14:57:32+00:00  Feb 27 15:28:30 CET 2026
- **Owners**:
  - Approvers: None
  - Reviewers: None