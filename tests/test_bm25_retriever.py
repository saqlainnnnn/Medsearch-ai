from unittest.mock import Mock

import pytest

from app.retrieval.sparse.bm25 import BM25Retriever
from app.retrieval.sparse.exceptions import SparseRetrievalError
from app.schemas.document import (
    Chunk,
    Document,
    DocumentMetadata,
)
from app.schemas.retrieval import RetrieverType


@pytest.fixture
def chunks() -> list[Chunk]:
    document = Document(
        metadata=DocumentMetadata(
            title="Test Document",
            source="test.pdf",
        ),
        content="Sample document.",
        word_count=2,
    )

    return [
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


def test_build_index(chunks: list[Chunk]) -> None:
    retriever = BM25Retriever()

    retriever.build(chunks)

    assert retriever._bm25 is not None
    assert len(retriever._chunks) == 2


def test_build_empty_index() -> None:
    retriever = BM25Retriever()

    with pytest.raises(SparseRetrievalError):
        retriever.build([])


def test_search_returns_results(
    chunks: list[Chunk],
) -> None:
    retriever = BM25Retriever()

    retriever.build(chunks)

    results = retriever.search(
        query="diabetes",
        top_k=2,
    )

    assert len(results) == 2


def test_search_before_build_raises_error() -> None:
    retriever = BM25Retriever()

    with pytest.raises(SparseRetrievalError):
        retriever.search("diabetes")


def test_top_k_respected(
    chunks: list[Chunk],
) -> None:
    retriever = BM25Retriever()

    retriever.build(chunks)

    results = retriever.search(
        query="diabetes",
        top_k=1,
    )

    assert len(results) == 1


def test_retriever_type_is_bm25(
    chunks: list[Chunk],
) -> None:
    retriever = BM25Retriever()

    retriever.build(chunks)

    results = retriever.search("diabetes")

    assert all(
        result.retriever == RetrieverType.BM25
        for result in results
    )


def test_metadata_preserved(
    chunks: list[Chunk],
) -> None:
    retriever = BM25Retriever()

    retriever.build(chunks)

    results = retriever.search("diabetes")

    assert (
        results[0].chunk.document_id
        == chunks[0].document_id
    )


def test_search_failure(
    monkeypatch,
    chunks: list[Chunk],
) -> None:
    retriever = BM25Retriever()

    retriever.build(chunks)

    broken_bm25 = Mock()
    broken_bm25.get_scores.side_effect = RuntimeError("boom")

    monkeypatch.setattr(
        retriever,
        "_bm25",
        broken_bm25,
    )

    with pytest.raises(SparseRetrievalError):
        retriever.search("diabetes")