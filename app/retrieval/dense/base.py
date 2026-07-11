from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np

from app.schemas.document import EmbeddedChunk
from app.schemas.retrieval import RetrievedChunk


class BaseDenseRetriever(ABC):
    """
    Base interface for dense retrieval systems.
    """

    @abstractmethod
    def build(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> None:
        """
        Build the dense retrieval index.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks.
        """
        raise NotImplementedError