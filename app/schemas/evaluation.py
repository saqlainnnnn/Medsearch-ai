from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class RetrievalMetrics(BaseModel):
    """
    Metrics evaluating retrieval quality.
    """

    model_config = ConfigDict(extra="forbid")

    recall_at_k: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Recall@K score.",
    )

    precision_at_k: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Precision@K score.",
    )

    mrr: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean Reciprocal Rank.",
    )

    hit_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Hit Rate.",
    )

    ndcg: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized Discounted Cumulative Gain.",
    )


class GenerationMetrics(BaseModel):
    """
    Metrics evaluating answer quality.
    """

    model_config = ConfigDict(extra="forbid")

    faithfulness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    answer_relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    context_precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    context_recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class EvaluationResult(BaseModel):
    """
    Complete evaluation results for a query.
    """

    model_config = ConfigDict(extra="forbid")

    evaluation_id: UUID = Field(
        default_factory=uuid4,
    )

    query_id: UUID = Field(
        ...,
        description="Query being evaluated.",
    )

    retrieval: RetrievalMetrics

    generation: GenerationMetrics

    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )