from typing import Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def documents_indexing(
    embedding_model_id: str,
    extracted_text: dsl.Input[dsl.Artifact],
    vector_io_provider_id: str,
    indexing_stats: dsl.Output[dsl.Artifact],
    embedding_params: Optional[dict] = None,
    distance_metric: str = "cosine",
    chunking_method: str = "recursive",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    batch_size: int = 20,
    collection_name: Optional[str] = None,
    pattern_name: Optional[str] = None,
    vector_store_type: Optional[str] = None,
):
    """Index extracted text into a vector store with optional batch processing.

    Reads markdown files from extracted_text, chunks them, embeds via OGX,
    and adds them to the vector store. When batch_size > 0, processes documents
    in batches to limit memory use and allow progress on large inputs.

    Args:
        embedding_model_id: Embedding model ID used for the vector store.
        extracted_text: Input artifact (folder) containing .md files from text extraction.
        vector_io_provider_id: OGX provider ID for the vector database.
        embedding_params: Optional embedding parameters.
        distance_metric: Vector distance metric (e.g. "cosine").
        chunking_method: Chunking method.
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap in characters.
        batch_size: Number of documents per batch; 0 means process all in one batch.
        collection_name: Optional name of the collection to reuse; omit to create a new one.
        indexing_stats: Output artifact directory containing vector_store_stats.json and
            indexing_run_metadata.json.
        pattern_name: Optional RAG pattern identifier for run metadata.
        vector_store_type: Optional vector store provider type (e.g. milvus) for statistics.
    """
    import json
    import logging
    import os
    import ssl
    import sys
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import httpx
    from ai4rag.rag.chunking import LangChainChunker
    from ai4rag.rag.embedding.ogx import OGXEmbeddingModel, OGXEmbeddingParams
    from ai4rag.rag.vector_store.ogx import OGXVectorStore
    from langchain_core.documents import Document
    from ogx_client import APIConnectionError as OGXAPIConnectionError
    from ogx_client import OgxClient

    def _is_ssl_error(exc: BaseException) -> bool:
        """Check whether an exception (or its cause/context chain) is an SSL verification failure."""
        seen = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            msg = str(current).upper()
            if "CERTIFICATE_VERIFY_FAILED" in msg or "SSL" in msg:
                return True
            current = current.__cause__ or current.__context__
        return False

    def _create_ogx_client(**kwargs) -> OgxClient:
        """Create OgxClient, falling back to SSL-unverified if self-signed cert detected."""
        client = OgxClient(**kwargs)
        try:
            client.models.list()
        except (ssl.SSLCertVerificationError, httpx.ConnectError, OGXAPIConnectionError) as exc:
            if _is_ssl_error(exc):
                logger.warning(
                    "SSL verification failed for OgxClient — retrying with verify=False. ",
                )
                client = OgxClient(
                    **kwargs,
                    http_client=httpx.Client(verify=False),
                )
            else:
                raise
        return client

    logger = logging.getLogger("Document Loader component logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    supported_distance_metrics = ("cosine", "euclidean")
    supported_chunking_methods = ("recursive",)
    supported_chunks_sizes_range = (128, 2048)

    if not vector_io_provider_id or not vector_io_provider_id.strip():
        raise ValueError("vector_io_provider_id must be a non-empty string.")

    if not embedding_model_id:
        raise ValueError("embedding_model_id must be a non-empty string.")

    if distance_metric not in supported_distance_metrics:
        raise ValueError(
            f"distance metric {distance_metric} is not supported, supported types are {supported_distance_metrics}."
        )

    if chunking_method not in supported_chunking_methods:
        raise ValueError(f"chunking_method is not supported, supported methods are {supported_chunking_methods}.")

    if not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer.")
    else:
        if not (supported_chunks_sizes_range[0] <= chunk_size <= supported_chunks_sizes_range[1]):
            raise ValueError(
                f"chunk_size must be an integer in the range"
                f" {supported_chunks_sizes_range[0]} to {supported_chunks_sizes_range[1]}."
            )

    if not isinstance(chunk_overlap, (int, float)):
        raise TypeError("chunk_overlap must be a numerical value.")

    if embedding_params is None:
        embedding_params = {}
    else:
        if not isinstance(embedding_params, dict):
            raise TypeError("embedding_params must be a dictionary.")

    params = OGXEmbeddingParams(**embedding_params)

    ogx_base_url = os.getenv("OGX_CLIENT_BASE_URL")
    ogx_api_key = os.getenv("OGX_CLIENT_API_KEY")
    missing = [
        name for name, val in (("OGX_CLIENT_BASE_URL", ogx_base_url), ("OGX_CLIENT_API_KEY", ogx_api_key)) if not val
    ]
    if missing:
        raise RuntimeError(f"Required environment variable(s) not set: {', '.join(missing)}")

    def _write_indexing_artifacts(
        *,
        indexing_stats_path: Path,
        started_at: datetime,
        duration_seconds: float,
        pattern_name: str | None,
        vector_store_type: str | None,
        collection_name: str | None,
        vector_io_provider_id: str,
        embedding_model_id: str,
        embedding_params: dict,
        distance_metric: str,
        document_count: int,
        chunk_count: int,
    ) -> None:
        indexing_stats_path.mkdir(parents=True, exist_ok=True)
        completed_at = datetime.now(timezone.utc)
        embedding_dimension = (
            embedding_params.get("embedding_dimension") if isinstance(embedding_params, dict) else None
        )

        vector_store_stats = {
            "collection_name": collection_name,
            "vector_store_type": vector_store_type or vector_io_provider_id,
            "document_count": document_count,
            "chunk_count": chunk_count,
            "embedding_model": embedding_model_id,
            "embedding_dimension": embedding_dimension,
            "distance_metric": distance_metric,
            "indexing_duration_seconds": round(duration_seconds, 3),
            "created_at": completed_at.isoformat().replace("+00:00", "Z"),
            "pattern_id": pattern_name or "",
        }
        run_metadata = {
            "pattern_id": pattern_name or "",
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
            "duration_seconds": round(duration_seconds, 3),
            "document_count": document_count,
            "chunk_count": chunk_count,
            "embedding_model_id": embedding_model_id,
            "vector_io_provider_id": vector_io_provider_id,
            "collection_name": collection_name,
        }

        (indexing_stats_path / "vector_store_stats.json").write_text(
            json.dumps(vector_store_stats, indent=2),
            encoding="utf-8",
        )
        (indexing_stats_path / "indexing_run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2),
            encoding="utf-8",
        )

    client = _create_ogx_client(
        base_url=ogx_base_url,
        api_key=ogx_api_key,
    )

    started_at = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()

    base = Path(extracted_text.path)
    paths = sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".md")
    total_documents = len(paths)
    logger.info("Found %s documents to index", total_documents)

    resolved_collection_name = collection_name
    total_chunks = 0

    if total_documents == 0:
        logger.warning("No documents found in %s", extracted_text.path)
        _write_indexing_artifacts(
            indexing_stats_path=Path(indexing_stats.path),
            started_at=started_at,
            duration_seconds=time.monotonic() - start_monotonic,
            pattern_name=pattern_name,
            vector_store_type=vector_store_type,
            collection_name=resolved_collection_name,
            vector_io_provider_id=vector_io_provider_id,
            embedding_model_id=embedding_model_id,
            embedding_params=embedding_params,
            distance_metric=distance_metric,
            document_count=0,
            chunk_count=0,
        )
        return

    chunker = LangChainChunker(method=chunking_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedding_model = OGXEmbeddingModel(client=client, model_id=embedding_model_id, params=params)

    collection_name_param = {"reuse_collection_name": collection_name} if collection_name is not None else {}
    ogx_vectorstore = OGXVectorStore(
        embedding_model=embedding_model,
        client=client,
        provider_id=vector_io_provider_id,
        distance_metric=distance_metric,
        **collection_name_param,
    )

    effective_batch_size = batch_size if batch_size > 0 else total_documents

    for start in range(0, total_documents, effective_batch_size):
        batch_paths = paths[start : start + effective_batch_size]
        batch_documents = [
            Document(
                page_content=p.read_text(encoding="utf-8", errors="replace"),
                metadata={"document_id": p.stem},
            )
            for p in batch_paths
        ]
        batch_chunks = chunker.split_documents(batch_documents)
        ogx_vectorstore.add_documents(batch_chunks)
        total_chunks += len(batch_chunks)
        batch_num = start // effective_batch_size + 1
        num_batches = (total_documents + effective_batch_size - 1) // effective_batch_size
        logger.info(
            "Batch %s/%s: indexed %s documents (%s chunks), total chunks so far: %s",
            batch_num,
            num_batches,
            len(batch_documents),
            len(batch_chunks),
            total_chunks,
        )

    logger.info(
        "Documents indexing finished: %s documents, %s chunks",
        total_documents,
        total_chunks,
    )

    if resolved_collection_name is None:
        vs_collection = getattr(ogx_vectorstore, "collection_name", None)
        if isinstance(vs_collection, str) and vs_collection:
            resolved_collection_name = vs_collection
            logger.info("Resolved collection name from vector store: %s", vs_collection)
        elif vs_collection is not None:
            logger.warning(
                "Vector store collection_name has unexpected type %s; omitting from indexing stats.",
                type(vs_collection).__name__,
            )
        else:
            logger.warning(
                "Collection name was not provided and could not be resolved from the vector store; "
                "indexing stats will omit collection_name."
            )

    _write_indexing_artifacts(
        indexing_stats_path=Path(indexing_stats.path),
        started_at=started_at,
        duration_seconds=time.monotonic() - start_monotonic,
        pattern_name=pattern_name,
        vector_store_type=vector_store_type,
        collection_name=resolved_collection_name,
        vector_io_provider_id=vector_io_provider_id,
        embedding_model_id=embedding_model_id,
        embedding_params=embedding_params,
        distance_metric=distance_metric,
        document_count=total_documents,
        chunk_count=total_chunks,
    )
