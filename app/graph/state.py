from __future__ import annotations

from typing import TypedDict

from app.schemas.api import MedicalAnswer
from app.schemas.retrieval import RetrievalResult, SearchQuery


class GraphState(TypedDict):
    """
    State shared across the LangGraph workflow.
    """

    query: SearchQuery

    retrieval_result: RetrievalResult | None

    answer: MedicalAnswer | None