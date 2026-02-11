from typing import List, Optional

from kfp import dsl
from kfp.kubernetes import use_secret_as_env

from components.data_processing.autorag.document_loader import document_loader
from components.data_processing.autorag.test_data_loader import test_data_loader
from components.data_processing.autorag.text_extraction import text_extraction
from components.training.autorag.rag_templates_optimization.component import rag_templates_optimization
from components.training.autorag.search_space_preparation.component import search_space_preparation


@dsl.pipeline(
    name="documents-rag-optimization-pipeline",
    description="Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications",
)
def documents_rag_optimization_pipeline(
    test_data_secret_name: str,
    test_data_bucket_name: str,
    test_data_key: str,
    input_data_secret_name: str,
    input_data_bucket_name: str,
    input_data_key: str,
    llama_stack_secret_name: str,
    embeddings_models: Optional[List] = None,
    generation_models: Optional[List] = None,
    optimization_metrics: str = "faithfulness",
    vector_database_id: Optional[str] = None,
):
    """Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications.

    The Documents RAG Optimization Pipeline is an automated system for building and optimizing
    Retrieval-Augmented Generation (RAG) applications within Red Hat OpenShift AI. It leverages
    Kubeflow Pipelines to orchestrate the optimization workflow, using the ai4rag optimization
    engine to systematically explore RAG configurations and identify the best performing parameter
    settings based on an upfront-specified quality metric.

    The system integrates with llama-stack API for inference and vector database operations,
    producing optimized RAG patterns as artifacts that can be deployed and used for production
    RAG applications.

    Args:
        test_data_secret_name: Name of the Kubernetes secret holding S3-compatible credentials for
            test data access. The following environment variables are required:
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION.
        test_data_bucket_name: S3 (or compatible) bucket name for the test data file.
        test_data_key: Object key (path) of the test data JSON file in the test data bucket.
        input_data_secret_name: Name of the Kubernetes secret holding S3-compatible credentials
            for input document data access. The following environment variables are required:
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION.
        input_data_bucket_name: S3 (or compatible) bucket name for the input documents.
        input_data_key: Object key (path) of the input documents in the input data bucket.
        llama_stack_secret_name: Name of the Kubernetes secret for llama-stack API connection.
            The secret is expected to provide the LLAMASTACK_CLIENT_CONNECTION environment variable.
        embeddings_models: Optional list of embedding model identifiers to use in the search space.
        generation_models: Optional list of foundation/generation model identifiers to use in the
            search space.
        optimization_metrics: Quality metric used to optimize RAG patterns (e.g., "faithfulness").
        vector_database_id: Optional vector database id (e.g., registered in llama-stack Milvus).
            If not provided, an in-memory database may be used.
    """

    test_data_loader_task = test_data_loader(
        test_data_bucket_name=test_data_bucket_name,
        test_data_path=test_data_key,
    )

    document_loader_task = document_loader(
        input_data_bucket_name=input_data_bucket_name,
        input_data_path=input_data_key,
        test_data=test_data_loader_task.outputs["test_data"],
        sampling_config={},
    )

    for task, secret_name in zip(
        [test_data_loader_task, document_loader_task],
        [test_data_secret_name, input_data_secret_name],
    ):
        use_secret_as_env(
            task,
            secret_name=secret_name,
            secret_key_to_env={
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
                "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
                "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
            },
        )

    text_extraction_task = text_extraction(documents=document_loader_task.outputs["sampled_documents"])

    mps_task = search_space_preparation(
        test_data=test_data_loader_task.outputs["test_data"],
        extracted_text=text_extraction_task.outputs["extracted_text"],
        embeddings_models=embeddings_models,
        generation_models=generation_models,
    )

    hpo_task = rag_templates_optimization(
        extracted_text=text_extraction_task.outputs["extracted_text"],
        test_data=test_data_loader_task.outputs["test_data"],
        search_space_prep_report=mps_task.outputs["search_space_prep_report"],
        vector_database_id="ls_milvus",
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        documents_rag_optimization_pipeline,
        package_path=__file__.replace(".py", "_pipeline.yaml"),
    )
