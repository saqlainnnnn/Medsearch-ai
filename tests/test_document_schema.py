from uuid import UUID

import pytest
from pydantic import ValidationError

from app.schemas.document import (
    Chunk,
    Document,
    DocumentMetadata,
)

def test_document_metadata_creation():
    metadata = DocumentMetadata(
        title="Hypertension Guidelines",
        authors=["John Doe", "Jane Doe"],
        journal="Medical Journal",
        publication_year=2024,
        doi="10.1000/test",
        source="PubMed",
        url="https://example.com",
        keywords=["hypertension", "blood pressure"],
    )

    assert metadata.title == "Hypertension Guidelines"
    assert metadata.publication_year == 2024
    assert len(metadata.authors) == 2

def test_document_creation():
    metadata = DocumentMetadata(
        title="Test",
        source="WHO",
    )

    document = Document(
        metadata=metadata,
        content="This is a medical document.",
        word_count=5,
    )

    assert isinstance(document.document_id, UUID)
    assert document.language == "en"
    assert document.word_count == 5

def test_chunk_creation():
    metadata = DocumentMetadata(
        title="Test",
        source="WHO",
    )

    document = Document(
        metadata=metadata,
        content="Some medical text.",
        word_count=3,
    )

    chunk = Chunk(
        document_id=document.document_id,
        chunk_index=0,
        content="Some medical text.",
        token_count=3,
    )

    assert chunk.chunk_index == 0
    assert chunk.document_id == document.document_id

def test_missing_required_field():
    with pytest.raises(ValidationError):
        DocumentMetadata(
            source="WHO"
        )

def test_invalid_publication_year():
    with pytest.raises(ValidationError):
        DocumentMetadata(
            title="Test",
            source="WHO",
            publication_year=1800,
        )

def test_negative_chunk_index():
    metadata = DocumentMetadata(
        title="Test",
        source="WHO",
    )

    document = Document(
        metadata=metadata,
        content="Hello",
    )

    with pytest.raises(ValidationError):
        Chunk(
            document_id=document.document_id,
            chunk_index=-1,
            content="Hello",
        )

def test_empty_document_content():
    metadata = DocumentMetadata(
        title="Test",
        source="WHO",
    )

    with pytest.raises(ValidationError):
        Document(
            metadata=metadata,
            content="",
        )

def test_uuid_generation():
    metadata = DocumentMetadata(
        title="Test",
        source="WHO",
    )

    doc1 = Document(
        metadata=metadata,
        content="A",
    )

    doc2 = Document(
        metadata=metadata,
        content="B",
    )

    assert doc1.document_id != doc2.document_id