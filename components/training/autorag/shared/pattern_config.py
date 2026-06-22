"""Map AutoRAG pattern.json settings to documents indexing pipeline parameters."""

from __future__ import annotations

from typing import Any


def pattern_json_to_indexing_params(pattern: dict[str, Any], *, input_data_key: str = "") -> dict[str, Any]:
    """Translate a pattern.json document into ``documents_indexing_pipeline`` parameter names.

    Secret names and ``input_data_bucket_name`` are intentionally omitted so operators
    can supply them at deploy time.

    Args:
        pattern: Parsed pattern.json content (flat schema with ``name`` and ``settings``).
        input_data_key: S3 prefix for the full document corpus (from optimization context).

    Returns:
        Dict of pipeline parameter names to default values for YAML compilation.
    """
    settings = pattern.get("settings", {})
    chunking = settings.get("chunking", {})
    embedding = settings.get("embedding", {})
    vs_binding = settings.get("vector_store_binding", {})
    vs_legacy = settings.get("vector_store", {})

    provider_id = vs_binding.get("provider_id") or vs_legacy.get("datasource_type") or ""
    collection_name = vs_binding.get("vector_store_id") or vs_legacy.get("collection_name")
    provider_type = vs_binding.get("provider_type") or vs_legacy.get("datasource_type") or ""

    embedding_params = embedding.get("embedding_params")
    if embedding_params is None:
        embedding_params = {}

    params: dict[str, Any] = {
        "pattern_name": pattern.get("name", ""),
        "embedding_model_id": embedding.get("model_id", ""),
        "vector_io_provider_id": provider_id,
        "embedding_params": embedding_params,
        "distance_metric": embedding.get("distance_metric", "cosine"),
        "chunking_method": chunking.get("method", "recursive"),
        "chunk_size": chunking.get("chunk_size", 1024),
        "chunk_overlap": chunking.get("chunk_overlap", 0),
        "input_data_key": input_data_key or "",
        "vector_store_type": provider_type,
    }

    if collection_name is not None:
        params["collection_name"] = collection_name

    return params
