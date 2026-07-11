from __future__ import annotations

from pathlib import Path

from app.chunking.base import BaseChunker
from app.embeddings.base import BaseEmbedder
from app.ingestion.cleaner import BaseTextCleaner
from app.ingestion.loader import BaseDocumentLoader
from app.ingestion.parser import BaseDocumentParser
from app.schemas.document import EmbeddedChunk


class IngestionService:
    """
    High-level ingestion service.

    Coordinates loading, cleaning, parsing,
    chunking, and embedding.
    """

    def __init__(
        self,
        loader: BaseDocumentLoader,
        cleaner: BaseTextCleaner,
        parser: BaseDocumentParser,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
    ) -> None:
        self._loader = loader
        self._cleaner = cleaner
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder

    def ingest(
        self,
        document_path: str | Path,
    ) -> list[EmbeddedChunk]:
        """
        Execute the complete ingestion pipeline.
        """

        document_path = Path(document_path)

        pages = self._loader.load(document_path)

        cleaned_pages = self._cleaner.clean(pages)

        document = self._parser.parse(
            cleaned_pages,
            document_path,
        )

        chunks = self._chunker.chunk(document)

        embedded_chunks = self._embedder.embed(chunks)

        return embedded_chunks