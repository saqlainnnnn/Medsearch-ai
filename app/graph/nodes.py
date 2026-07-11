from __future__ import annotations

from app.graph.state import GraphState
from app.services.generation import GenerationService
from app.services.search import SearchService


class SearchNode:
    """
    LangGraph node responsible for document retrieval.
    """

    def __init__(
        self,
        search_service: SearchService,
    ) -> None:
        self._search_service = search_service

    def __call__(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Execute retrieval.
        """

        retrieval_result = self._search_service.search(
            state["query"],
        )

        return {
            "retrieval_result": retrieval_result,
        }


class GenerationNode:
    """
    LangGraph node responsible for answer generation.
    """

    def __init__(
        self,
        generation_service: GenerationService,
    ) -> None:
        self._generation_service = generation_service

    def __call__(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Generate the final answer.
        """

        answer = self._generation_service.generate(
            state["retrieval_result"],
        )

        return {
            "answer": answer,
        }