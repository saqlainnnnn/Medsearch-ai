from unittest.mock import Mock

from app.schemas.api import MedicalAnswer
from app.schemas.document import Chunk
from app.schemas.retrieval import (
    RetrievalResult,
    RetrievedChunk,
    RetrieverType,
    SearchQuery,
)
from app.services.generation import GenerationService


def create_retrieval_result() -> RetrievalResult:
    query = SearchQuery(
        query="What is diabetes?"
    )

    chunk = RetrievedChunk(
        chunk=Chunk(
            document_id="00000000-0000-0000-0000-000000000001",
            chunk_index=0,
            content="Diabetes is a chronic metabolic disease.",
            token_count=7,
        ),
        retrieval_score=0.95,
        rank=1,
        retriever=RetrieverType.HYBRID,
    )

    return RetrievalResult(
        query=query,
        retrieved_chunks=[chunk],
        retriever=RetrieverType.HYBRID,
        latency_ms=5.0,
    )


def test_generation_returns_medical_answer() -> None:
    llm = Mock()

    llm.generate.return_value = "Generated answer."

    service = GenerationService(llm)

    result = service.generate(
        create_retrieval_result()
    )

    assert isinstance(result, MedicalAnswer)


def test_answer_matches_llm_output() -> None:
    llm = Mock()

    llm.generate.return_value = "Generated answer."

    service = GenerationService(llm)

    result = service.generate(
        create_retrieval_result()
    )

    assert result.answer == "Generated answer."


def test_llm_called_once() -> None:
    llm = Mock()

    llm.generate.return_value = "Answer"

    service = GenerationService(llm)

    service.generate(
        create_retrieval_result()
    )

    llm.generate.assert_called_once()


def test_citations_created() -> None:
    llm = Mock()

    llm.generate.return_value = "Answer"

    service = GenerationService(llm)

    result = service.generate(
        create_retrieval_result()
    )

    assert len(result.citations) == 1


def test_citation_metadata_matches_chunk() -> None:
    llm = Mock()

    llm.generate.return_value = "Answer"

    service = GenerationService(llm)

    result = service.generate(
        create_retrieval_result()
    )

    citation = result.citations[0]

    assert citation.source_title == "Unknown"
    assert citation.page_number is None
    assert citation.section is None


def test_empty_retrieval_result() -> None:
    llm = Mock()

    llm.generate.return_value = "No evidence."

    service = GenerationService(llm)

    retrieval = RetrievalResult(
        query=SearchQuery(query="Unknown"),
        retrieved_chunks=[],
        retriever=RetrieverType.HYBRID,
        latency_ms=1.0,
    )

    result = service.generate(retrieval)

    assert result.answer == "No evidence."
    assert result.citations == []