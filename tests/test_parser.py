from pathlib import Path

import pytest

from app.ingestion.exceptions import DocumentParseError
from app.ingestion.models import ParsedPage
from app.ingestion.parser import MedicalDocumentParser
from app.schemas.document import Document


def test_parse_single_page() -> None:
    parser = MedicalDocumentParser()

    pages = [
        ParsedPage(
            page_number=1,
            text="This is a medical document.",
        )
    ]

    document = parser.parse(
        pages=pages,
        document_path=Path("diabetes.pdf"),
    )

    assert isinstance(document, Document)
    assert document.content == "This is a medical document."


def test_parse_multiple_pages() -> None:
    parser = MedicalDocumentParser()

    pages = [
        ParsedPage(1, "Page one."),
        ParsedPage(2, "Page two."),
    ]

    document = parser.parse(
        pages=pages,
        document_path=Path("diabetes.pdf"),
    )

    assert document.content == "Page one.\n\nPage two."


def test_word_count() -> None:
    parser = MedicalDocumentParser()

    pages = [
        ParsedPage(
            page_number=1,
            text="one two three four",
        )
    ]

    document = parser.parse(
        pages=pages,
        document_path=Path("test.pdf"),
    )

    assert document.word_count == 4


def test_document_title() -> None:
    parser = MedicalDocumentParser()

    document = parser.parse(
        pages=[ParsedPage(1, "Hello")],
        document_path=Path("covid_guidelines.pdf"),
    )

    assert document.metadata.title == "covid_guidelines"


def test_document_source() -> None:
    parser = MedicalDocumentParser()

    document = parser.parse(
        pages=[ParsedPage(1, "Hello")],
        document_path=Path("covid_guidelines.pdf"),
    )

    assert document.metadata.source == "covid_guidelines.pdf"


def test_empty_pages_raise_error() -> None:
    parser = MedicalDocumentParser()

    with pytest.raises(
        DocumentParseError,
        match="No pages were provided",
    ):
        parser.parse(
            pages=[],
            document_path=Path("test.pdf"),
        )


def test_created_document_is_valid() -> None:
    parser = MedicalDocumentParser()

    document = parser.parse(
        pages=[ParsedPage(1, "Medical document text.")],
        document_path=Path("paper.pdf"),
    )

    assert isinstance(document.document_id, object)
    assert document.language == "en"
    assert document.created_at is not None