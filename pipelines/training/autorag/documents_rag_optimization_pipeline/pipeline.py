from typing import Dict, List, Optional

from kfp import dsl


@dsl.pipeline(
    name="documents-rag-optimization-pipeline",
    description="Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications",
)
def documents_rag_optimization_pipeline(
    name: str,
    input_data_reference: Dict,
    test_data_reference: Dict,
    results_reference: Dict,
    description: Optional[str] = None,
    vector_database_id: Optional[str] = None,
    mlflow_config: Optional[Dict] = None,
    optimization: Optional[Dict] = None,
    chunking_constraints: Optional[List[Dict]] = None,
    embeddings_constraints: Optional[List[Dict]] = None,
    generation_constraints: Optional[List[Dict]] = None,
    retrieval_constraints: Optional[List[Dict]] = None,
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
    # TODO: Implement your pipeline logic here


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        documents_rag_optimization_pipeline,
        package_path=__file__.replace(".py", "_pipeline.yaml"),
    )
