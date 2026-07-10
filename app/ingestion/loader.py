from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import fitz  # PyMuPDF

from app.ingestion.exceptions import PDFLoadError
from app.ingestion.models import ParsedPage


class BaseDocumentLoader(ABC):
    """
    Base interface for all document loaders.
    """

    @abstractmethod
    def load(
    self,
    document_path: Path,
) -> list[ParsedPage]:
        """
        Extract text from every page of a PDF.

        Parameters
        ----------
        document_path : Path
            Path to the PDF document.

        Returns
        -------
        list[ParsedPage]
            Pages extracted from the PDF.

        Raises
        ------
        PDFLoadError
            If the PDF cannot be opened or processed.
        """

        if not document_path.exists():
            raise PDFLoadError(f"Document not found: {document_path}")

        pages: list[ParsedPage] = []

        try:
            with fitz.open(document_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    pages.append(
                        ParsedPage(
                            page_number=page_number,
                            text=page.get_text("text"),
                        )
                    )

        except Exception as exc:
            raise PDFLoadError(
                f"Failed to load PDF: {document_path}"
            ) from exc

        return pages


class PDFDocumentLoader(BaseDocumentLoader):
    """
    Loads PDF documents using PyMuPDF.
    """

    def load(
        self,
        document_path: Path,
    ) -> list[ParsedPage]:
        """
        Extract text from every page of a PDF.

        Parameters
        ----------
        document_path : Path
            Path to the PDF document.

        Returns
        -------
        list[ParsedPage]
            Pages extracted from the PDF.

        Raises
        ------
        PDFLoadError
            If the PDF cannot be opened or processed.
        """

        if not document_path.exists():
            raise PDFLoadError(f"Document not found: {document_path}")

        try:
            pdf = fitz.open(document_path)
        except Exception as exc:
            raise PDFLoadError(f"Failed to open PDF: {document_path}") from exc

        pages: list[ParsedPage] = []

        try:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text()

                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=text,
                    )
                )

        except Exception as exc:
            raise PDFLoadError("Failed while extracting PDF pages.") from exc

        finally:
            pdf.close()

        return pages