"""Tests for per-pattern indexing pipeline YAML compilation."""

from pathlib import Path

import yaml

from kfp_components.components.training.autorag.shared.pipeline_compile import compile_indexing_pipeline_yaml


def _minimal_pattern() -> dict:
    return {
        "name": "Pattern 01",
        "settings": {
            "chunking": {"method": "recursive", "chunk_size": 768, "chunk_overlap": 128},
            "embedding": {
                "model_id": "text-embedding-model",
                "distance_metric": "cosine",
                "embedding_params": {"embedding_dimension": 768},
            },
            "vector_store_binding": {
                "provider_id": "milvus",
                "provider_type": "milvus",
                "vector_store_id": "coll_pattern_01",
            },
        },
    }


def test_compile_indexing_pipeline_yaml_sets_defaults_and_image(tmp_path: Path):
    shared_root = Path(__file__).resolve().parents[1]
    base_template = shared_root / "pipeline_templates" / "documents_indexing_pipeline.yaml"
    assert base_template.is_file(), "Run pipeline compile to refresh pipeline_templates/documents_indexing_pipeline.yaml"

    output_path = tmp_path / "indexing_pipeline.yaml"
    compile_indexing_pipeline_yaml(
        pattern=_minimal_pattern(),
        output_path=output_path,
        base_template_path=base_template,
        input_data_key="production/docs/",
        image="quay.io/example/autorag@sha256:abc123",
    )

    with output_path.open(encoding="utf-8") as handle:
        docs = list(yaml.safe_load_all(handle))

    spec = docs[0]

    params = spec["root"]["inputDefinitions"]["parameters"]
    assert params["embedding_model_id"]["defaultValue"] == "text-embedding-model"
    assert params["vector_io_provider_id"]["defaultValue"] == "milvus"
    assert params["collection_name"]["defaultValue"] == "coll_pattern_01"
    assert params["chunk_size"]["defaultValue"] == 768.0
    assert params["chunk_overlap"]["defaultValue"] == 128.0
    assert params["input_data_key"]["defaultValue"] == "production/docs/"
    assert params["vector_store_type"]["defaultValue"] == "milvus"
    assert params["pattern_name"]["defaultValue"] == "Pattern 01"
    assert "ogx_secret_name" in params
    assert "defaultValue" not in params["ogx_secret_name"]

    assert spec["pipelineInfo"]["name"] == "autorag-indexing-pattern-01"
    assert spec["deploymentSpec"]["executors"]["exec-documents-indexing"]["container"]["image"] == (
        "quay.io/example/autorag@sha256:abc123"
    )
