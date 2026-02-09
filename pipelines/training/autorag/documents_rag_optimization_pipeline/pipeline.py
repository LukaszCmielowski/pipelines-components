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
    input_data_secret_name: str,
    test_data_bucket_name: str,
    test_data_key: str,
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
    producing optimized RAG Patterns as artifacts that can be deployed and used for production
    RAG applications. It can also communicate with externally provided MLFlow server to support
    advanced experiment tracking features.

    Args:
        name: Name of the AutoRAG experiment run (e.g., "AutoRAG run").
        input_data_reference: Dictionary defining document data source with keys: connection_id,
            bucket, path.
        test_data_reference: Dictionary defining test data source with keys: connection_id, bucket,
            path. Test data JSON file is supported only.
        results_reference: Dictionary defining results storage location with keys: connection_id,
            bucket, path.
        description: Optional description of the experiment (e.g., "RHOAI Kubeflow Pipelines Docs").
        vector_database_id: Optional vector database id (e.g., registered in llama-stack Milvus
            database). If not provided, an in-memory database will be used.
        mlflow_config: Optional dictionary defining MLFlow configuration for experiment tracking with
            keys: tracking_uri, experiment_name, enabled.
        optimization: Optional dictionary defining optimization settings with keys:
            max_number_of_rag_patterns (int), metric (str). Supported metrics: faithfulness,
            answer_correctness.
        chunking_constraints: Optional list of dictionaries defining chunking configurations. Each
            dictionary contains: method (str), chunk_overlap (int), chunk_size (int).
        embeddings_constraints: Optional list of dictionaries defining embedding models. Each
            dictionary contains: model (str).
        generation_constraints: Optional list of dictionaries defining generation models. Each
            dictionary contains: model (str), optional context_template_text (str), optional
            messages (list[dict]).
        retrieval_constraints: Optional list of dictionaries defining retrieval method
            configurations. Each dictionary contains: method (str), number_of_chunks (int),
            optional hybrid_ranker (dict).
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
                "aws_access_key_id": "AWS_ACCESS_KEY_ID",
                "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
                "endpoint_url": "AWS_ENDPOINT_URL",
                "aws_region_name": "AWS_REGION",
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
