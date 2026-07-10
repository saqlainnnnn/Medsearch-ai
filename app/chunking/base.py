from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.document import Chunk, Document


class BaseChunker(ABC):
    """
    Base interface for document chunkers.
    """

    @abstractmethod
    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        """
        Split a document into searchable chunks.
        """
        raise NotImplementedError