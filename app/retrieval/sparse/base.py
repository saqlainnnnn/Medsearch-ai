from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.schemas.document import Chunk
from app.schemas.retrieval import RetrievedChunk


class BaseSparseRetriever(ABC):
    """
    Base interface for sparse retrieval systems.
    """

    @abstractmethod
    def build(
        self,
        chunks: Sequence[Chunk],
    ) -> None:
        """
        Build the sparse retrieval index.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks.
        """
        raise NotImplementedError