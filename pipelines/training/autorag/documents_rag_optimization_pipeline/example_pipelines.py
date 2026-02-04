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
