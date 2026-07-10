from pathlib import Path

import fitz
import pytest

from app.ingestion.exceptions import PDFLoadError
from app.ingestion.loader import PDFDocumentLoader
from app.ingestion.models import ParsedPage


def test_load_valid_pdf(sample_pdf_path: Path) -> None:
    """
    Loader should successfully extract pages from a valid PDF.
    """
    loader = PDFDocumentLoader()

    pages = loader.load(sample_pdf_path)

    assert isinstance(pages, list)
    assert len(pages) > 0
    assert all(isinstance(page, ParsedPage) for page in pages)


def test_page_numbers_are_sequential(sample_pdf_path: Path) -> None:
    """
    Page numbers should start at 1 and increase sequentially.
    """
    loader = PDFDocumentLoader()

    pages = loader.load(sample_pdf_path)

    expected = list(range(1, len(pages) + 1))
    actual = [page.page_number for page in pages]

    assert actual == expected


def test_page_text_is_string(sample_pdf_path: Path) -> None:
    """
    Every extracted page should contain text as a string.
    """
    loader = PDFDocumentLoader()

    pages = loader.load(sample_pdf_path)

    assert all(isinstance(page.text, str) for page in pages)


def test_load_nonexistent_pdf_raises_error() -> None:
    """
    Loading a missing PDF should raise PDFLoadError.
    """
    loader = PDFDocumentLoader()

    with pytest.raises(PDFLoadError, match="Document not found"):
        loader.load(Path("does_not_exist.pdf"))


def test_invalid_pdf_raises_error(tmp_path: Path) -> None:
    """
    A non-PDF file should raise PDFLoadError.
    """
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("This is not a PDF.")

    loader = PDFDocumentLoader()

    with pytest.raises(PDFLoadError):
        loader.load(invalid_pdf)


def test_empty_pages_are_preserved(monkeypatch) -> None:
    """
    Empty page text should still produce ParsedPage objects.
    """

    class DummyPage:
        def get_text(self) -> str:
            return ""

    class DummyDocument:
        def __iter__(self):
            return iter([DummyPage(), DummyPage()])

        def close(self):
            pass

    monkeypatch.setattr(fitz, "open", lambda *_: DummyDocument())
    monkeypatch.setattr(Path, "exists", lambda _: True)

    loader = PDFDocumentLoader()

    pages = loader.load(Path("fake.pdf"))

    assert len(pages) == 2
    assert pages[0].text == ""
    assert pages[1].text == ""