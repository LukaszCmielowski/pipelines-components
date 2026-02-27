# Search Space Preparation 🔍

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Builds the search space defining RAG configurations. 
Performs the so called model preselection that limits the number of foundation and embedding models for the further optimization stage to 3 and 2 (respectively) of the best performing ones.
The output of this step is a .yml formatted report file containg the search space definition.
It allows for easy recreation at any time making the eventual re-runs faster and reproducible to greate extent.

The preselection phase, similarly to the optimization stage, also makes use of the `ai4rag` library to explore some RAG configurations allowing to grade the models' performance and choose a couple of best ones from the start.
The usage of this compoent ensures that only valid and performant configurations are passed to the further optimization stage greatly reducing computational cost and improving optimization efficiency.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_data` | dsl.Input[dsl.Artifact] | - | The test data for an experiment in form of a JSON file. |
| `extracted_text` | dsl.Input[dsl.Artifact] | - | A folder of files with text extracted from the input documents. |
| `embeddings_models` | dsl.Input[dsl.Artifact] | - | List of embedding model identifiers to try out in the experiment process. |
| `generation_models` | dsl.Input[dsl.Artifact] | - | List of foundation model identifiers to try out in the experiment process. |
| `metric` | `str` | `faithfulness` | A RAG metric to optimise the experiment for. |


## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `search_space_prep_report` | `dsl.Output[dsl.Artifact]` | Path to a .yml-formatted file containing the search space definition. |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.training.autorag.search_space_preparation import (
    search_space_preparation,
)

@dsl.pipeline(name="search-space-preparation-pipeline")
def my_pipeline():
    """Example pipeline for search space preparation."""
    constraints = {
        "chunking": [
            {
                "method": "recursive",
                "chunk_overlap": 256,
                "chunk_size": 2048
            }
        ],
        "embeddings": [
            {"model": "ibm/slate-125m-english-rtrvr-v2"}
        ],
        "generation": [
            {"model": "mistralai/mixtral-8x7b-instruct-v01"}
        ],
        "retrieval": [
            {
                "method": "simple",
                "number_of_chunks": 2
            }
        ]
    }
    
    prep_task = search_space_preparation(
        test_data=,
        extracted_text=,
        embedding_models=,
        generation_models=,
    )
    return prep_task
```


## Search space creation Process 🔧

The component performs the following checks in order to ensure a valid search space:

1. **Model Availability**: Validates that specified models are available and accessible
2. **Configuration Compatibility**: Ensures configuration combinations are compatible
3. **Performance Testing**: Tests configurations using an in-memory vector database
4. **Search Space Adjustment**: Adjusts the search space based on validation results
5. **Output Generation**: Produces serialized search space in a form of a .yml file

## In-Memory Vector Database 🗄️

The component uses an in-memory vector database for:

- Model performance validation
- Configuration compatibility testing
- Search space optimization
- Early filtering of invalid configurations

## Notes 📝

- **Model Preselection**: Validates and preselects models based on performance
- **Search Space Reduction**: Filters out invalid configurations before optimization
- **ai4rag Integration**: Uses ai4rag library for systematic validation
- **Optimization Efficiency**: Reduces computational cost by validating configurations early

## Metadata 🗂️

- **Name**: search_space_preparation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: >=1.0.0
- **Tags**:
  - training
  - autorag
  - search-space
  - optimization
- **Last Verified**: 2026-01-23 00:00:00+00:00

## Additional Resources 📚

- **AutoRAG Documentation**: See AutoRAG pipeline documentation for comprehensive information
- **ai4rag Documentation**: [ai4rag GitHub](https://github.com/IBM/ai4rag)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
