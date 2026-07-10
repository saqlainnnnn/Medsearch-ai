from pathlib import Path
from unittest.mock import Mock

import pytest

from app.ingestion.exceptions import (
    DocumentParseError,
    PDFLoadError,
    TextCleaningError,
)
from app.ingestion.models import ParsedPage
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.schemas.document import Document, DocumentMetadata


@pytest.fixture
def sample_document() -> Document:
    return Document(
        metadata=DocumentMetadata(
            title="Test",
            source="test.pdf",
        ),
        content="Sample content",
        word_count=2,
    )


def test_pipeline_success(sample_document: Document) -> None:
    loader = Mock()
    cleaner = Mock()
    parser = Mock()

    pages = [ParsedPage(page_number=1, text="Hello")]

    loader.load.return_value = pages
    cleaner.clean.return_value = pages
    parser.parse.return_value = sample_document

    pipeline = DocumentIngestionPipeline(
        loader=loader,
        cleaner=cleaner,
        parser=parser,
    )

    result = pipeline.ingest(Path("test.pdf"))

    loader.load.assert_called_once_with(Path("test.pdf"))
    cleaner.clean.assert_called_once_with(pages)
    parser.parse.assert_called_once_with(
        pages=pages,
        document_path=Path("test.pdf"),
    )

    assert result is sample_document


def test_loader_error_propagates() -> None:
    loader = Mock()
    cleaner = Mock()
    parser = Mock()

    loader.load.side_effect = PDFLoadError("Failed")

    pipeline = DocumentIngestionPipeline(
        loader=loader,
        cleaner=cleaner,
        parser=parser,
    )

    with pytest.raises(PDFLoadError):
        pipeline.ingest(Path("test.pdf"))


def test_cleaner_error_propagates() -> None:
    loader = Mock()
    cleaner = Mock()
    parser = Mock()

    pages = [ParsedPage(page_number=1, text="Hello")]

    loader.load.return_value = pages
    cleaner.clean.side_effect = TextCleaningError("Failed")

    pipeline = DocumentIngestionPipeline(
        loader=loader,
        cleaner=cleaner,
        parser=parser,
    )

    with pytest.raises(TextCleaningError):
        pipeline.ingest(Path("test.pdf"))


def test_parser_error_propagates() -> None:
    loader = Mock()
    cleaner = Mock()
    parser = Mock()

    pages = [ParsedPage(page_number=1, text="Hello")]

    loader.load.return_value = pages
    cleaner.clean.return_value = pages
    parser.parse.side_effect = DocumentParseError("Failed")

    pipeline = DocumentIngestionPipeline(
        loader=loader,
        cleaner=cleaner,
        parser=parser,
    )

    with pytest.raises(DocumentParseError):
        pipeline.ingest(Path("test.pdf"))