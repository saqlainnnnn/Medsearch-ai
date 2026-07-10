import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ingestion.models import ParsedPage
from app.schemas.document import Chunk, Document, DocumentMetadata


# -----------------------------
# Schema fixtures
# -----------------------------

@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        title="Test Medical Document",
        authors=["John Doe"],
        journal="Medical Journal",
        publication_year=2024,
        source="WHO",
        keywords=["medicine"],
    )


@pytest.fixture
def sample_document(sample_metadata: DocumentMetadata) -> Document:
    return Document(
        metadata=sample_metadata,
        content="This is a sample medical document.",
        word_count=6,
    )


@pytest.fixture
def sample_chunk(sample_document: Document) -> Chunk:
    return Chunk(
        document_id=sample_document.document_id,
        chunk_index=0,
        content="This is a sample chunk.",
        token_count=6,
    )


# -----------------------------
# Ingestion fixtures
# -----------------------------

@pytest.fixture
def sample_pdf_path() -> Path:
    return Path("assets/sample_pdfs/diabetes.pdf")


@pytest.fixture
def sample_pages() -> list[ParsedPage]:
    return [
        ParsedPage(
            page_number=1,
            text="This is page one.",
        ),
        ParsedPage(
            page_number=2,
            text="This is page two.",
        ),
    ]


@pytest.fixture
def dirty_pages() -> list[ParsedPage]:
    return [
        ParsedPage(
            page_number=1,
            text="\tThis    is   page one.\n\n\n",
        ),
        ParsedPage(
            page_number=2,
            text="   Another\t\tpage.\n\n\n\nText   here.   ",
        ),
    ]