from uuid import UUID

import pytest
from pydantic import ValidationError

from app.schemas.document import Chunk, Document, DocumentMetadata


class TestDocumentMetadata:
    def test_creation(self):
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

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(
                source="WHO",
            )

    def test_invalid_publication_year(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(
                title="Test",
                source="WHO",
                publication_year=1800,
            )


class TestDocument:
    def test_creation(self, sample_metadata):
        document = Document(
            metadata=sample_metadata,
            content="This is a medical document.",
            word_count=5,
        )

        assert isinstance(document.document_id, UUID)
        assert document.language == "en"
        assert document.word_count == 5

    def test_empty_content(self, sample_metadata):
        with pytest.raises(ValidationError):
            Document(
                metadata=sample_metadata,
                content="",
            )

    def test_uuid_generation(self, sample_metadata):
        doc1 = Document(
            metadata=sample_metadata,
            content="A",
        )

        doc2 = Document(
            metadata=sample_metadata,
            content="B",
        )

        assert doc1.document_id != doc2.document_id


class TestChunk:
    def test_creation(self, sample_document):
        chunk = Chunk(
            document_id=sample_document.document_id,
            chunk_index=0,
            content="Some medical text.",
            token_count=3,
        )

        assert chunk.chunk_index == 0
        assert chunk.document_id == sample_document.document_id

    def test_negative_chunk_index(self, sample_document):
        with pytest.raises(ValidationError):
            Chunk(
                document_id=sample_document.document_id,
                chunk_index=-1,
                content="Hello",
            )