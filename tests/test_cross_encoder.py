from unittest.mock import Mock

import pytest

from app.retrieval.reranker.cross_encoder import CrossEncoderReranker
from app.retrieval.reranker.exceptions import RerankingError
from app.schemas.document import Chunk, Document, DocumentMetadata
from app.schemas.retrieval import RetrievedChunk, RetrieverType


@pytest.fixture
def retrieved_chunks() -> list[RetrievedChunk]:
    document = Document(
        metadata=DocumentMetadata(
            title="Test Document",
            source="test.pdf",
        ),
        content="Sample document.",
        word_count=2,
    )

    chunk1 = Chunk(
        document_id=document.document_id,
        chunk_index=0,
        content="Diabetes is a chronic disease.",
        token_count=5,
    )

    chunk2 = Chunk(
        document_id=document.document_id,
        chunk_index=1,
        content="Insulin regulates blood glucose.",
        token_count=5,
    )

    return [
        RetrievedChunk(
            chunk=chunk1,
            retrieval_score=0.10,
            rank=2,
            retriever=RetrieverType.HYBRID,
        ),
        RetrievedChunk(
            chunk=chunk2,
            retrieval_score=0.20,
            rank=1,
            retriever=RetrieverType.HYBRID,
        ),
    ]


def test_rerank_returns_results(
    monkeypatch,
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    model = Mock()
    model.predict.return_value = [0.9, 0.2]

    reranker._model = model

    results = reranker.rerank(
        query="diabetes",
        chunks=retrieved_chunks,
    )

    assert len(results) == 2


def test_empty_input_returns_empty_list() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    reranker._model = Mock()

    assert reranker.rerank(
        query="diabetes",
        chunks=[],
    ) == []


def test_top_k_respected(
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    model = Mock()
    model.predict.return_value = [0.9, 0.2]

    reranker._model = model

    results = reranker.rerank(
        query="diabetes",
        chunks=retrieved_chunks,
        top_k=1,
    )

    assert len(results) == 1


def test_scores_are_updated(
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    model = Mock()
    model.predict.return_value = [0.9, 0.2]

    reranker._model = model

    results = reranker.rerank(
        query="diabetes",
        chunks=retrieved_chunks,
    )

    assert results[0].retrieval_score == 0.9
    assert results[1].retrieval_score == 0.2


def test_ranks_are_reassigned(
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    model = Mock()
    model.predict.return_value = [0.9, 0.2]

    reranker._model = model

    results = reranker.rerank(
        query="diabetes",
        chunks=retrieved_chunks,
    )

    assert results[0].rank == 1
    assert results[1].rank == 2


def test_metadata_preserved(
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    model = Mock()
    model.predict.return_value = [0.9, 0.2]

    reranker._model = model

    results = reranker.rerank(
        query="diabetes",
        chunks=retrieved_chunks,
    )

    assert (
        results[0].chunk.document_id
        == retrieved_chunks[0].chunk.document_id
    )


def test_predict_called_once(
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    model = Mock()
    model.predict.return_value = [0.9, 0.2]

    reranker._model = model

    reranker.rerank(
        query="diabetes",
        chunks=retrieved_chunks,
    )

    model.predict.assert_called_once()


def test_prediction_failure(
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    model = Mock()
    model.predict.side_effect = RuntimeError("boom")

    reranker._model = model

    with pytest.raises(RerankingError):
        reranker.rerank(
            query="diabetes",
            chunks=retrieved_chunks,
        )