from __future__ import annotations

from abc import ABC, abstractmethod
import re

from app.ingestion.models import ParsedPage


class BaseTextCleaner(ABC):
    """
    Base interface for text cleaners.
    """

    @abstractmethod
    def clean(
        self,
        pages: list[ParsedPage],
    ) -> list[ParsedPage]:
        """
        Clean extracted pages.
        """
        raise NotImplementedError


class MedicalTextCleaner(BaseTextCleaner):
    """
    Cleans extracted text from medical PDFs.
    """

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalize extracted PDF text.
        """

        # Normalize tabs
        text = text.replace("\t", " ")

        # Collapse multiple spaces
        text = re.sub(r"[ ]{2,}", " ", text)

        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def clean(
    self,
    pages: list[ParsedPage],
) -> list[ParsedPage]:
        """
        Clean every extracted page.
        """

        return [
            ParsedPage(
                page_number=page.page_number,
                text=self._clean_text(page.text),
            )
            for page in pages
        ]