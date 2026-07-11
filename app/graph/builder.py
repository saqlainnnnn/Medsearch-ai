from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import GenerationNode, SearchNode
from app.graph.state import GraphState
from app.services.generation import GenerationService
from app.services.search import SearchService


def build_graph(
    search_service: SearchService,
    generation_service: GenerationService,
) -> CompiledStateGraph:
    """
    Build the MedSearch LangGraph workflow.

    Parameters
    ----------
    search_service : SearchService
        Service responsible for retrieval.

    generation_service : GenerationService
        Service responsible for answer generation.

    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow.
    """

    builder = StateGraph(GraphState)

    builder.add_node(
        "search",
        SearchNode(search_service),
    )

    builder.add_node(
        "generate",
        GenerationNode(generation_service),
    )

    builder.add_edge(
        START,
        "search",
    )

    builder.add_edge(
        "search",
        "generate",
    )

    builder.add_edge(
        "generate",
        END,
    )

    return builder.compile()