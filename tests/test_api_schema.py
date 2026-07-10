from uuid import UUID

import pytest
from pydantic import ValidationError

from app.schemas.api import (
    Citation,
    MedicalAnswer,
    QueryRequest,
    QueryResponse,
)
from app.schemas.retrieval import (
    RetrievalResult,
    RetrieverType,
    RetrievedChunk,
    SearchQuery,
)


class TestCitation:
    def test_creation(self):
        citation = Citation(
            source_title="ADA Guidelines",
            page_number=12,
            section="Treatment",
        )

        assert citation.source_title == "ADA Guidelines"
        assert citation.page_number == 12
        assert citation.section == "Treatment"

    def test_optional_fields(self):
        citation = Citation(
            source_title="WHO Guidelines",
        )

        assert citation.page_number is None
        assert citation.section is None


class TestQueryRequest:
    def test_creation(self):
        request = QueryRequest(
            question="What is diabetes?"
        )

        assert request.question == "What is diabetes?"
        assert request.top_k == 5
        assert request.include_sources is True

    def test_custom_top_k(self):
        request = QueryRequest(
            question="What is diabetes?",
            top_k=10,
        )

        assert request.top_k == 10

    def test_empty_question(self):
        with pytest.raises(ValidationError):
            QueryRequest(
                question=""
            )

    def test_invalid_top_k(self):
        with pytest.raises(ValidationError):
            QueryRequest(
                question="Hello",
                top_k=0,
            )


class TestMedicalAnswer:
    def test_creation(self):
        citation = Citation(
            source_title="ADA Guidelines",
            page_number=8,
        )

        answer = MedicalAnswer(
            answer="Metformin is the recommended first-line therapy.",
            citations=[citation],
        )

        assert answer.grounded is True
        assert len(answer.citations) == 1

    def test_no_citations(self):
        answer = MedicalAnswer(
            answer="No supporting evidence found.",
        )

        assert answer.citations == []


class TestQueryResponse:
    def test_creation(self, sample_chunk):
        query = SearchQuery(
            query="What is diabetes?"
        )

        retrieved_chunk = RetrievedChunk(
            chunk=sample_chunk,
            retrieval_score=0.95,
            rank=1,
            retriever=RetrieverType.DENSE,
        )

        retrieval_result = RetrievalResult(
            query=query,
            retrieved_chunks=[retrieved_chunk],
            retriever=RetrieverType.DENSE,
            latency_ms=15.2,
        )

        medical_answer = MedicalAnswer(
            answer="Diabetes is a chronic metabolic disorder.",
            citations=[
                Citation(
                    source_title="ADA Guidelines",
                    page_number=3,
                )
            ],
        )

        response = QueryResponse(
            question=query.query,
            answer=medical_answer,
            retrieval_result=retrieval_result,
            latency_ms=82.4,
        )

        assert isinstance(response.query_id, UUID)
        assert response.question == query.query
        assert response.latency_ms == 82.4
        assert response.retrieval_result.total_chunks == 1

    def test_negative_latency(self, sample_chunk):
        query = SearchQuery(
            query="Hello"
        )

        retrieval_result = RetrievalResult(
            query=query,
            retriever=RetrieverType.DENSE,
        )

        medical_answer = MedicalAnswer(
            answer="Hello"
        )

        with pytest.raises(ValidationError):
            QueryResponse(
                question="Hello",
                medical_answer=medical_answer,
                retrieval_result=retrieval_result,
                latency_ms=-1,
            )