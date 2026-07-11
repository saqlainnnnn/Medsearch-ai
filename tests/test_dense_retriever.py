from unittest.mock import Mock

import pytest

from app.retrieval.dense.exceptions import DenseRetrievalError
from app.retrieval.dense.faiss import FAISSDenseRetriever
from app.schemas.document import (
    Chunk,
    Document,
    DocumentMetadata,
    EmbeddedChunk,
)
from app.schemas.retrieval import RetrieverType


@pytest.fixture
def embedded_chunks() -> list[EmbeddedChunk]:
    document = Document(
        metadata=DocumentMetadata(
            title="Test Document",
            source="test.pdf",
        ),
        content="Sample document.",
        word_count=2,
    )

    chunks = [
        Chunk(
            document_id=document.document_id,
            chunk_index=0,
            content="Diabetes is a chronic disease.",
            token_count=5,
        ),
        Chunk(
            document_id=document.document_id,
            chunk_index=1,
            content="Insulin regulates blood glucose.",
            token_count=5,
        ),
    ]

    return [
        EmbeddedChunk(
            chunk=chunks[0],
            embedding=[0.1, 0.2, 0.3, 0.4],
        ),
        EmbeddedChunk(
            chunk=chunks[1],
            embedding=[0.2, 0.3, 0.4, 0.5],
        ),
    ]


def test_build_index(embedded_chunks: list[EmbeddedChunk]) -> None:
    retriever = FAISSDenseRetriever()

    retriever.build(embedded_chunks)

    assert retriever._index is not None
    assert len(retriever._embedded_chunks) == 2


def test_build_empty_index() -> None:
    retriever = FAISSDenseRetriever()

    with pytest.raises(DenseRetrievalError):
        retriever.build([])


def test_search_returns_results(
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    retriever = FAISSDenseRetriever()

    retriever.build(embedded_chunks)

    results = retriever.search(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        top_k=2,
    )

    assert len(results) == 2


def test_search_before_build_raises_error() -> None:
    retriever = FAISSDenseRetriever()

    with pytest.raises(DenseRetrievalError):
        retriever.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
        )


def test_top_k_respected(
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    retriever = FAISSDenseRetriever()

    retriever.build(embedded_chunks)

    results = retriever.search(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        top_k=1,
    )

    assert len(results) == 1


def test_retriever_type_is_dense(
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    retriever = FAISSDenseRetriever()

    retriever.build(embedded_chunks)

    results = retriever.search(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
    )

    assert all(
        result.retriever == RetrieverType.DENSE
        for result in results
    )


def test_metadata_preserved(
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    retriever = FAISSDenseRetriever()

    retriever.build(embedded_chunks)

    results = retriever.search(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
    )

    assert (
        results[0].chunk.document_id
        == embedded_chunks[0].chunk.document_id
    )


def test_search_failure(
    monkeypatch,
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    retriever = FAISSDenseRetriever()

    retriever.build(embedded_chunks)

    broken_index = Mock()
    broken_index.search.side_effect = RuntimeError("boom")

    monkeypatch.setattr(
        retriever,
        "_index",
        broken_index,
    )

    with pytest.raises(DenseRetrievalError):
        retriever.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
        )