from uuid import UUID

import pytest
from pydantic import ValidationError

from app.schemas.retrieval import (
    RetrieverType,
    RetrievalResult,
    RetrievedChunk,
    SearchQuery,
)


class TestSearchQuery:
    def test_creation(self):
        query = SearchQuery(
            query="What is diabetes?"
        )

        assert isinstance(query.query_id, UUID)
        assert query.query == "What is diabetes?"
        assert query.top_k == 10

    def test_query_rewrite(self):
        query = SearchQuery(
            query="diabetes",
            rewritten_query="What is diabetes mellitus?"
        )

        assert query.rewritten_query == "What is diabetes mellitus?"

    def test_empty_query(self):
        with pytest.raises(ValidationError):
            SearchQuery(
                query=""
            )


class TestRetrievedChunk:
    def test_creation(self, sample_chunk):
        retrieved = RetrievedChunk(
            chunk=sample_chunk,
            retrieval_score=0.92,
            rank=1,
            retriever=RetrieverType.DENSE,
        )

        assert retrieved.retrieval_score == 0.92
        assert retrieved.rank == 1
        assert retrieved.chunk == sample_chunk

    def test_negative_score(self, sample_chunk):
        with pytest.raises(ValidationError):
            RetrievedChunk(
                chunk=sample_chunk,
                retrieval_score=-1,
                rank=1,
                retriever=RetrieverType.DENSE,
            )

    def test_invalid_rank(self, sample_chunk):
        with pytest.raises(ValidationError):
            RetrievedChunk(
                chunk=sample_chunk,
                retrieval_score=0.8,
                rank=0,
                retriever=RetrieverType.DENSE,
            )


class TestRetrievalResult:
    def test_creation(self, sample_chunk):
        retrieved = RetrievedChunk(
            chunk=sample_chunk,
            retrieval_score=0.95,
            rank=1,
            retriever=RetrieverType.DENSE,
        )

        query = SearchQuery(
            query="What is diabetes?"
        )

        result = RetrievalResult(
            query=query,
            retrieved_chunks=[retrieved],
            retriever=RetrieverType.DENSE,
            latency_ms=12.4,
        )

        assert result.retriever == RetrieverType.DENSE
        assert result.total_chunks == 1
        assert len(result.retrieved_chunks) == 1

    def test_invalid_latency(self):
        query = SearchQuery(
            query="Hello"
        )

        with pytest.raises(ValidationError):
            RetrievalResult(
                query=query,
                retriever=RetrieverType.DENSE,
                latency_ms=-1,
            )