import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas.document import Chunk, Document, DocumentMetadata


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