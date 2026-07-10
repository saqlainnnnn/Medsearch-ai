from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.exceptions import DocumentParseError
from app.ingestion.models import ParsedPage
from app.schemas.document import Document, DocumentMetadata


class BaseDocumentParser(ABC):
    """
    Base interface for document parsers.
    """

    @abstractmethod
    def parse(
        self,
        pages: list[ParsedPage],
        document_path: Path,
    ) -> Document:
        """
        Parse extracted pages into a Document.
        """
        raise NotImplementedError


class MedicalDocumentParser(BaseDocumentParser):
    """
    Converts cleaned PDF pages into a validated Document.
    """

    def parse(
        self,
        pages: list[ParsedPage],
        document_path: Path,
    ) -> Document:
        """
        Build a Document object from cleaned pages.
        """

        if not pages:
            raise DocumentParseError("No pages were provided for parsing.")

        try:
            # Combine all page text into a single document
            content = "\n\n".join(page.text for page in pages)

            # Build metadata
            metadata = DocumentMetadata(
                title=document_path.stem,
                source=document_path.name,
            )

            # Count words
            word_count = len(content.split())

            document = Document(
                metadata=metadata,
                content=content,
                word_count=word_count,
            )

            return document

        except Exception as exc:
            raise DocumentParseError(
                f"Failed to parse document: {document_path}"
            ) from exc