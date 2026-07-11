from unittest.mock import Mock

from app.schemas.document import Chunk
from app.schemas.retrieval import (
    RetrievalResult,
    RetrievedChunk,
    RetrieverType,
    SearchQuery,
)
from app.services.search import SearchService


def test_search_service_returns_reranked_results() -> None:
    pipeline = Mock()
    reranker = Mock()

    query = SearchQuery(
        query="What is diabetes?",
        top_k=5,
    )

    chunk = Chunk(
        document_id="00000000-0000-0000-0000-000000000001",
        chunk_index=0,
        content="Diabetes is a chronic disease.",
        token_count=5,
    )

    retrieved = RetrievedChunk(
        chunk=chunk,
        retrieval_score=0.7,
        rank=1,
        retriever=RetrieverType.HYBRID,
    )

    pipeline.retrieve.return_value = RetrievalResult(
        query=query,
        retrieved_chunks=[retrieved],
        retriever=RetrieverType.HYBRID,
        latency_ms=12.5,
    )

    reranked = retrieved.model_copy(
        update={
            "retrieval_score": 0.95,
            "rank": 1,
        }
    )

    reranker.rerank.return_value = [reranked]

    service = SearchService(
        retrieval_pipeline=pipeline,
        reranker=reranker,
    )

    result = service.search(query)

    assert len(result.retrieved_chunks) == 1
    assert result.retrieved_chunks[0].retrieval_score == 0.95
    assert result.query == query
    assert result.retriever == RetrieverType.HYBRID
    assert result.latency_ms == 12.5


def test_pipeline_called_once() -> None:
    pipeline = Mock()
    reranker = Mock()

    query = SearchQuery(query="diabetes")

    pipeline.retrieve.return_value = RetrievalResult(
        query=query,
        retrieved_chunks=[],
        retriever=RetrieverType.HYBRID,
        latency_ms=1.0,
    )

    reranker.rerank.return_value = []

    service = SearchService(
        retrieval_pipeline=pipeline,
        reranker=reranker,
    )

    service.search(query)

    pipeline.retrieve.assert_called_once_with(query)


def test_reranker_called_with_pipeline_results() -> None:
    pipeline = Mock()
    reranker = Mock()

    query = SearchQuery(
        query="diabetes",
        top_k=3,
    )

    chunk = Chunk(
        document_id="00000000-0000-0000-0000-000000000001",
        chunk_index=0,
        content="Diabetes is a chronic disease.",
        token_count=5,
    )

    retrieved = RetrievedChunk(
        chunk=chunk,
        retrieval_score=0.8,
        rank=1,
        retriever=RetrieverType.HYBRID,
    )

    pipeline.retrieve.return_value = RetrievalResult(
        query=query,
        retrieved_chunks=[retrieved],
        retriever=RetrieverType.HYBRID,
        latency_ms=5.0,
    )

    reranker.rerank.return_value = [retrieved]

    service = SearchService(
        retrieval_pipeline=pipeline,
        reranker=reranker,
    )

    service.search(query)

    reranker.rerank.assert_called_once_with(
        query="diabetes",
        chunks=[retrieved],
        top_k=3,
    )


def test_metadata_preserved_after_reranking() -> None:
    pipeline = Mock()
    reranker = Mock()

    query = SearchQuery(query="covid")

    chunk = Chunk(
        document_id="00000000-0000-0000-0000-000000000001",
        chunk_index=0,
        content="COVID-19 is caused by SARS-CoV-2.",
        token_count=6,
    )

    retrieved = RetrievedChunk(
        chunk=chunk,
        retrieval_score=0.5,
        rank=1,
        retriever=RetrieverType.HYBRID,
    )

    pipeline.retrieve.return_value = RetrievalResult(
        query=query,
        retrieved_chunks=[retrieved],
        retriever=RetrieverType.HYBRID,
        latency_ms=10.0,
    )

    reranker.rerank.return_value = [retrieved]

    service = SearchService(
        retrieval_pipeline=pipeline,
        reranker=reranker,
    )

    result = service.search(query)

    assert result.query == query
    assert result.retriever == RetrieverType.HYBRID
    assert result.latency_ms == 10.0