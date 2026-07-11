from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.schemas.retrieval import RetrievedChunk


class BaseReranker(ABC):
    """
    Base interface for reranking systems.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank retrieved chunks.

        Parameters
        ----------
        query : str
            User query.

        chunks : Sequence[RetrievedChunk]
            Retrieved chunks.

        top_k : int | None
            Maximum number of chunks to return.

        Returns
        -------
        list[RetrievedChunk]
            Reranked chunks.
        """
        raise NotImplementedError