from unittest.mock import Mock

from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.document import Chunk
from app.schemas.retrieval import (
    RetrievalResult,
    RetrievedChunk,
    RetrieverType,
    SearchQuery,
)


def test_pipeline_retrieves_results() -> None:
    embedder = Mock()
    hybrid = Mock()

    embedder.embed_query.return_value = [0.1, 0.2, 0.3]

    chunk = Chunk(
        document_id="00000000-0000-0000-0000-000000000001",
        chunk_index=0,
        content="Diabetes is a chronic disease.",
        token_count=5,
    )

    hybrid.search.return_value = [
        RetrievedChunk(
            chunk=chunk,
            retrieval_score=0.95,
            rank=1,
            retriever=RetrieverType.DENSE,
        )
    ]

    pipeline = RetrievalPipeline(
        embedder=embedder,
        hybrid_retriever=hybrid,
    )

    query = SearchQuery(
        query="What is diabetes?",
        top_k=5,
    )

    result = pipeline.retrieve(query)

    assert isinstance(result, RetrievalResult)
    assert result.query == query
    assert len(result.retrieved_chunks) == 1
    assert result.retriever == RetrieverType.HYBRID


def test_embedder_called_once() -> None:
    embedder = Mock()
    hybrid = Mock()

    embedder.embed_query.return_value = [0.1, 0.2]

    hybrid.search.return_value = []

    pipeline = RetrievalPipeline(embedder, hybrid)

    query = SearchQuery(query="diabetes")

    pipeline.retrieve(query)

    embedder.embed_query.assert_called_once_with(
        "diabetes"
    )


def test_hybrid_called_with_correct_arguments() -> None:
    embedder = Mock()
    hybrid = Mock()

    embedder.embed_query.return_value = [0.3, 0.4]

    hybrid.search.return_value = []

    pipeline = RetrievalPipeline(embedder, hybrid)

    query = SearchQuery(
        query="insulin",
        top_k=7,
    )

    pipeline.retrieve(query)

    hybrid.search.assert_called_once_with(
        query="insulin",
        query_embedding=[0.3, 0.4],
        top_k=7,
    )


def test_latency_is_non_negative() -> None:
    embedder = Mock()
    hybrid = Mock()

    embedder.embed_query.return_value = [0.1]
    hybrid.search.return_value = []

    pipeline = RetrievalPipeline(
        embedder=embedder,
        hybrid_retriever=hybrid,
    )

    result = pipeline.retrieve(
        SearchQuery(query="covid")
    )

    assert result.latency_ms >= 0.0