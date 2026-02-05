from typing import Dict, List, Optional

from kfp import dsl

# from pipelines.data_processing.autorag.data_loading_pipeline import data_loading_pipeline
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
    # extracted_text: dsl.InputPath(dsl.Artifact),
    # test_data: dsl.InputPath(dsl.Artifact),
    secret_name: str = "kubeflow-aws-secrets",
    test_data_bucket_name: str = "wnowogorski-test-bucket",
    test_data_path: str = "benchmark.json",
    input_data_bucket_name: str = "wnowogorski-test-bucket",
    input_data_path: str = "",
    sampling_config: dict = {},
    vector_database_id: Optional[str] = None,
    mlflow_config: Optional[dict] = None,
    optimization: Optional[dict] = None,
    chunking_constraints: Optional[list[dict]] = None,
    embeddings_constraints: Optional[list[dict]] = None,
    generation_constraints: Optional[list[dict]] = None,
    retrieval_constraints: Optional[list[dict]] = None,
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
        test_data_path=test_data_path,
    )

    document_loader_task = document_loader(
        input_data_bucket_name=input_data_bucket_name,
        input_data_path=input_data_path,
        test_data=test_data_loader_task.outputs["test_data"],
        sampling_config=sampling_config,
    )

    text_extraction_task = text_extraction(documents=document_loader_task.outputs["sampled_documents"])

    mps_task = search_space_preparation(
        test_data=test_data_loader_task.outputs["test_data"],
        extracted_text=text_extraction_task.outputs["extracted_text"],
        constraints=constraints,
    )

    hpo_task = rag_templates_optimization(
        extracted_text=text_extraction_task.outputs["extracted_text"],
        test_data=test_data_loader_task.outputs["test_data"],
        search_space_prep_report=mps_task.outputs["search_space_prep_report"],
        vector_database_id="milvus",
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        documents_rag_optimization_pipeline,
        package_path=__file__.replace(".py", "_pipeline.yaml"),
    )
