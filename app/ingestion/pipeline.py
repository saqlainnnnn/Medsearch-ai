from __future__ import annotations

from pathlib import Path

from app.ingestion.cleaner import MedicalTextCleaner
from app.ingestion.loader import PDFDocumentLoader
from app.ingestion.parser import MedicalDocumentParser
from app.schemas.document import Document


class DocumentIngestionPipeline:
    """
    End-to-end document ingestion pipeline.
    """

    def __init__(
        self,
        loader: PDFDocumentLoader,
        cleaner: MedicalTextCleaner,
        parser: MedicalDocumentParser,
    ) -> None:
        self.loader = loader
        self.cleaner = cleaner
        self.parser = parser

    def ingest(
        self,
        document_path: Path,
    ) -> Document:

        pages = self.loader.load(document_path)
        cleaned_pages = self.cleaner.clean(pages)

        return self.parser.parse(
            pages=cleaned_pages,
            document_path=document_path,
        )