from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.schemas.evaluation import (
    EvaluationResult,
    GenerationMetrics,
    RetrievalMetrics,
)


class TestRetrievalMetrics:
    def test_creation(self):
        metrics = RetrievalMetrics(
            recall_at_k=0.85,
            precision_at_k=0.76,
            mrr=0.81,
            hit_rate=1.0,
            ndcg=0.79,
        )

        assert metrics.recall_at_k == 0.85
        assert metrics.precision_at_k == 0.76
        assert metrics.mrr == 0.81
        assert metrics.hit_rate == 1.0
        assert metrics.ndcg == 0.79

    @pytest.mark.parametrize(
        "field,value",
        [
            ("recall_at_k", -0.1),
            ("precision_at_k", 1.5),
            ("mrr", -1.0),
            ("hit_rate", 2.0),
            ("ndcg", -0.5),
        ],
    )
    def test_invalid_metric_values(self, field, value):
        valid = {
            "recall_at_k": 0.8,
            "precision_at_k": 0.8,
            "mrr": 0.8,
            "hit_rate": 0.8,
            "ndcg": 0.8,
        }

        valid[field] = value

        with pytest.raises(ValidationError):
            RetrievalMetrics(**valid)


class TestGenerationMetrics:
    def test_creation(self):
        metrics = GenerationMetrics(
            faithfulness=0.92,
            answer_relevance=0.89,
            context_precision=0.91,
            context_recall=0.87,
        )

        assert metrics.faithfulness == 0.92
        assert metrics.answer_relevance == 0.89
        assert metrics.context_precision == 0.91
        assert metrics.context_recall == 0.87

    @pytest.mark.parametrize(
        "field,value",
        [
            ("faithfulness", -0.1),
            ("answer_relevance", 1.2),
            ("context_precision", -1.0),
            ("context_recall", 2.0),
        ],
    )
    def test_invalid_metric_values(self, field, value):
        valid = {
            "faithfulness": 0.9,
            "answer_relevance": 0.9,
            "context_precision": 0.9,
            "context_recall": 0.9,
        }

        valid[field] = value

        with pytest.raises(ValidationError):
            GenerationMetrics(**valid)


class TestEvaluationResult:
    def test_creation(self):
        retrieval = RetrievalMetrics(
            recall_at_k=0.8,
            precision_at_k=0.75,
            mrr=0.82,
            hit_rate=1.0,
            ndcg=0.79,
        )

        generation = GenerationMetrics(
            faithfulness=0.93,
            answer_relevance=0.90,
            context_precision=0.91,
            context_recall=0.88,
        )

        result = EvaluationResult(
            query_id=uuid4(),
            retrieval=retrieval,
            generation=generation,
        )

        assert isinstance(result.evaluation_id, UUID)
        assert isinstance(result.query_id, UUID)
        assert result.retrieval.mrr == 0.82
        assert result.generation.faithfulness == 0.93
        assert result.evaluated_at is not None

    def test_unique_evaluation_ids(self):
        retrieval = RetrievalMetrics(
            recall_at_k=1.0,
            precision_at_k=1.0,
            mrr=1.0,
            hit_rate=1.0,
            ndcg=1.0,
        )

        generation = GenerationMetrics(
            faithfulness=1.0,
            answer_relevance=1.0,
            context_precision=1.0,
            context_recall=1.0,
        )

        result1 = EvaluationResult(
            query_id=uuid4(),
            retrieval=retrieval,
            generation=generation,
        )

        result2 = EvaluationResult(
            query_id=uuid4(),
            retrieval=retrieval,
            generation=generation,
        )

        assert result1.evaluation_id != result2.evaluation_id