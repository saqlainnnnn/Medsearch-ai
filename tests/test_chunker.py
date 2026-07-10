from unittest.mock import Mock

import pytest

from app.chunking.chunker import RecursiveChunker
from app.chunking.exceptions import ChunkingError
from app.schemas.document import Document, DocumentMetadata, Chunk


@pytest.fixture
def sample_document() -> Document:
    return Document(
        metadata=DocumentMetadata(
            title="Test Document",
            source="test.pdf",
        ),
        content="This is a sample medical document used for testing the recursive chunker.",
        word_count=12,
    )


def test_chunk_single_document(sample_document: Document) -> None:
    chunker = RecursiveChunker(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = chunker.chunk(sample_document)

    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)


def test_chunk_multiple_chunks() -> None:
    text = "Lorem ipsum dolor sit amet. " * 500

    document = Document(
        metadata=DocumentMetadata(
            title="Large Document",
            source="large.pdf",
        ),
        content=text,
        word_count=len(text.split()),
    )

    chunker = RecursiveChunker(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) > 1


def test_document_id_preserved(sample_document: Document) -> None:
    chunker = RecursiveChunker()

    chunks = chunker.chunk(sample_document)

    assert all(
        chunk.document_id == sample_document.document_id
        for chunk in chunks
    )


def test_chunk_indices_are_sequential() -> None:
    text = "Hello world. " * 400

    document = Document(
        metadata=DocumentMetadata(
            title="Sequential",
            source="seq.pdf",
        ),
        content=text,
        word_count=len(text.split()),
    )

    chunker = RecursiveChunker(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks = chunker.chunk(document)

    assert [
        chunk.chunk_index for chunk in chunks
    ] == list(range(len(chunks)))


def test_token_count_is_positive(sample_document: Document) -> None:
    chunker = RecursiveChunker()

    chunks = chunker.chunk(sample_document)

    assert all(
        chunk.token_count > 0
        for chunk in chunks
    )


def test_page_number_defaults_to_none(sample_document: Document) -> None:
    chunker = RecursiveChunker()

    chunks = chunker.chunk(sample_document)

    assert all(
        chunk.page_number is None
        for chunk in chunks
    )


def test_section_defaults_to_none(sample_document: Document) -> None:
    chunker = RecursiveChunker()

    chunks = chunker.chunk(sample_document)

    assert all(
        chunk.section is None
        for chunk in chunks
    )


def test_chunking_error(monkeypatch, sample_document: Document) -> None:
    chunker = RecursiveChunker()

    splitter = Mock()
    splitter.split_text.side_effect = RuntimeError("Boom")

    monkeypatch.setattr(chunker, "_splitter", splitter)

    with pytest.raises(ChunkingError):
        chunker.chunk(sample_document)