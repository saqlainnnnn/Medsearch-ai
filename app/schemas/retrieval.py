from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.document import Chunk


class RetrieverType(str, Enum):
    """
    Supported retrieval strategies.
    """

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class SearchQuery(BaseModel):
    """
    Represents a user query throughout the retrieval pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: UUID = Field(
        default_factory=uuid4,
        description="Unique query identifier.",
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Original user query.",
    )

    rewritten_query: str | None = Field(
        default=None,
        description="Query after rewriting.",
    )

    top_k: int = Field(
        default=10,
        ge=1,
        description="Number of chunks to retrieve.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the query was created.",
    )


class RetrievedChunk(BaseModel):
    """
    A chunk returned by a retriever.
    """

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk = Field(
        ...,
        description="Retrieved document chunk.",
    )

    retrieval_score: float = Field(
        ...,
        ge=0.0,
        description="Retrieval relevance score.",
    )

    rank: int = Field(
        ...,
        ge=1,
        description="Rank assigned by the retriever.",
    )

    retriever: RetrieverType = Field(
        ...,
        description="Retriever that produced this result.",
    )


class RetrievalResult(BaseModel):
    """
    Complete retrieval output for a query.
    """

    model_config = ConfigDict(extra="forbid")

    query: SearchQuery = Field(
        ...,
        description="Search query used for retrieval.",
    )

    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved chunks ordered by rank.",
    )

    retriever: RetrieverType = Field(
        ...,
        description="Retriever used to generate the result.",
    )

    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Retrieval latency in milliseconds.",
    )

    @property
    def total_chunks(self) -> int:
        """
        Returns the number of retrieved chunks.
        """
        return len(self.retrieved_chunks)