from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.schemas.document import Chunk, EmbeddedChunk


class BaseEmbedder(ABC):
    """
    Base interface for embedding generators.
    """

    @abstractmethod
    def embed(
        self,
        chunks: Sequence[Chunk],
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for document chunks.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for a search query.
        """
        raise NotImplementedError