from unittest.mock import Mock

import pytest

from app.embeddings.embedder import SentenceTransformerEmbedder
from app.embeddings.exceptions import EmbeddingError
from app.schemas.document import (
    Chunk,
    Document,
    DocumentMetadata,
    EmbeddedChunk,
)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    document = Document(
        metadata=DocumentMetadata(
            title="Test Document",
            source="test.pdf",
        ),
        content="Sample document.",
        word_count=2,
    )

    return [
        Chunk(
            document_id=document.document_id,
            chunk_index=0,
            content="This is the first chunk.",
            token_count=5,
        ),
        Chunk(
            document_id=document.document_id,
            chunk_index=1,
            content="This is the second chunk.",
            token_count=5,
        ),
    ]


def test_embed_single_chunk(sample_chunks: list[Chunk]) -> None:
    embedder = SentenceTransformerEmbedder()

    embedded = embedder.embed([sample_chunks[0]])

    assert len(embedded) == 1
    assert isinstance(embedded[0], EmbeddedChunk)


def test_embed_multiple_chunks(sample_chunks: list[Chunk]) -> None:
    embedder = SentenceTransformerEmbedder()

    embedded = embedder.embed(sample_chunks)

    assert len(embedded) == len(sample_chunks)


def test_chunk_metadata_preserved(sample_chunks: list[Chunk]) -> None:
    embedder = SentenceTransformerEmbedder()

    embedded = embedder.embed(sample_chunks)

    for original, embedded_chunk in zip(
        sample_chunks,
        embedded,
        strict=True,
    ):
        assert embedded_chunk.chunk == original


def test_embedding_is_list_of_floats(
    sample_chunks: list[Chunk],
) -> None:
    embedder = SentenceTransformerEmbedder()

    embedded = embedder.embed([sample_chunks[0]])

    vector = embedded[0].embedding

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


def test_embedding_dimension_consistent(
    sample_chunks: list[Chunk],
) -> None:
    embedder = SentenceTransformerEmbedder()

    embedded = embedder.embed(sample_chunks)

    dimension = len(embedded[0].embedding)

    assert dimension > 0

    for chunk in embedded:
        assert len(chunk.embedding) == dimension


def test_model_loading_error(monkeypatch) -> None:
    def broken_loader(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.embeddings.embedder.SentenceTransformer",
        broken_loader,
    )

    with pytest.raises(EmbeddingError):
        SentenceTransformerEmbedder()


def test_embedding_generation_error(
    monkeypatch,
    sample_chunks: list[Chunk],
) -> None:
    embedder = SentenceTransformerEmbedder()

    broken_model = Mock()
    broken_model.encode.side_effect = RuntimeError("boom")

    monkeypatch.setattr(
        embedder,
        "_model",
        broken_model,
    )

    with pytest.raises(EmbeddingError):
        embedder.embed(sample_chunks)