"""Example usage of the documents_rag_optimization_pipeline."""

from kfp import dsl
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
