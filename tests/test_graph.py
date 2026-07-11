from unittest.mock import Mock

from app.graph.builder import build_graph
from app.schemas.api import MedicalAnswer
from app.schemas.retrieval import (
    RetrievalResult,
    RetrieverType,
    SearchQuery,
)


def create_query() -> SearchQuery:
    return SearchQuery(
        query="What is diabetes?",
    )


def create_retrieval_result(
    query: SearchQuery,
) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        retrieved_chunks=[],
        retriever=RetrieverType.HYBRID,
        latency_ms=1.0,
    )


def create_answer() -> MedicalAnswer:
    return MedicalAnswer(
        answer="Diabetes is a chronic disease.",
        citations=[],
    )


def test_graph_executes_successfully() -> None:
    search_service = Mock()
    generation_service = Mock()

    query = create_query()

    retrieval_result = create_retrieval_result(query)
    answer = create_answer()

    search_service.search.return_value = retrieval_result
    generation_service.generate.return_value = answer

    graph = build_graph(
        search_service,
        generation_service,
    )

    result = graph.invoke(
        {
            "query": query,
            "retrieval_result": None,
            "answer": None,
        }
    )

    assert result["answer"] == answer


def test_search_service_called_once() -> None:
    search_service = Mock()
    generation_service = Mock()

    query = create_query()

    retrieval_result = create_retrieval_result(query)
    answer = create_answer()

    search_service.search.return_value = retrieval_result
    generation_service.generate.return_value = answer

    graph = build_graph(
        search_service,
        generation_service,
    )

    graph.invoke(
        {
            "query": query,
            "retrieval_result": None,
            "answer": None,
        }
    )

    search_service.search.assert_called_once_with(query)


def test_generation_service_called_once() -> None:
    search_service = Mock()
    generation_service = Mock()

    query = create_query()

    retrieval_result = create_retrieval_result(query)
    answer = create_answer()

    search_service.search.return_value = retrieval_result
    generation_service.generate.return_value = answer

    graph = build_graph(
        search_service,
        generation_service,
    )

    graph.invoke(
        {
            "query": query,
            "retrieval_result": None,
            "answer": None,
        }
    )

    generation_service.generate.assert_called_once_with(
        retrieval_result
    )


def test_retrieval_result_added_to_state() -> None:
    search_service = Mock()
    generation_service = Mock()

    query = create_query()

    retrieval_result = create_retrieval_result(query)
    answer = create_answer()

    search_service.search.return_value = retrieval_result
    generation_service.generate.return_value = answer

    graph = build_graph(
        search_service,
        generation_service,
    )

    result = graph.invoke(
        {
            "query": query,
            "retrieval_result": None,
            "answer": None,
        }
    )

    assert result["retrieval_result"] == retrieval_result


def test_answer_added_to_state() -> None:
    search_service = Mock()
    generation_service = Mock()

    query = create_query()

    retrieval_result = create_retrieval_result(query)
    answer = create_answer()

    search_service.search.return_value = retrieval_result
    generation_service.generate.return_value = answer

    graph = build_graph(
        search_service,
        generation_service,
    )

    result = graph.invoke(
        {
            "query": query,
            "retrieval_result": None,
            "answer": None,
        }
    )

    assert result["answer"] == answer