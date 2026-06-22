"""Tests for pattern.json to indexing pipeline parameter mapping."""

from kfp_components.components.training.autorag.shared.pattern_config import pattern_json_to_indexing_params


def test_pattern_json_to_indexing_params_maps_vector_store_binding():
    pattern = {
        "name": "pattern_alpha",
        "settings": {
            "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
            "embedding": {
                "model_id": "granite-embedding",
                "distance_metric": "euclidean",
                "embedding_params": {"embedding_dimension": 384},
            },
            "vector_store_binding": {
                "provider_id": "milvus-prod",
                "provider_type": "milvus",
                "vector_store_id": "coll_pattern_alpha",
            },
        },
    }

    params = pattern_json_to_indexing_params(pattern, input_data_key="docs/corpus/")

    assert params["pattern_name"] == "pattern_alpha"
    assert params["embedding_model_id"] == "granite-embedding"
    assert params["vector_io_provider_id"] == "milvus-prod"
    assert params["collection_name"] == "coll_pattern_alpha"
    assert params["vector_store_type"] == "milvus"
    assert params["chunk_size"] == 512
    assert params["chunk_overlap"] == 64
    assert params["distance_metric"] == "euclidean"
    assert params["embedding_params"] == {"embedding_dimension": 384}
    assert params["input_data_key"] == "docs/corpus/"


def test_pattern_json_to_indexing_params_falls_back_to_legacy_vector_store():
    pattern = {
        "name": "legacy_pattern",
        "settings": {
            "chunking": {},
            "embedding": {"model_id": "embed-model"},
            "vector_store": {
                "datasource_type": "pgvector",
                "collection_name": "legacy_coll",
            },
        },
    }

    params = pattern_json_to_indexing_params(pattern)

    assert params["vector_io_provider_id"] == "pgvector"
    assert params["collection_name"] == "legacy_coll"
    assert params["vector_store_type"] == "pgvector"
