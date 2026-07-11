from pathlib import Path
from unittest.mock import Mock

from app.schemas.document import Document, EmbeddedChunk
from app.services.ingestion import IngestionService


def test_ingestion_pipeline() -> None:
    loader = Mock()
    cleaner = Mock()
    parser = Mock()
    chunker = Mock()
    embedder = Mock()

    loader.load.return_value = ["page"]

    cleaner.clean.return_value = ["clean page"]

    parser.parse.return_value = Mock(spec=Document)

    chunker.chunk.return_value = ["chunk"]

    embedder.embed.return_value = [Mock(spec=EmbeddedChunk)]

    service = IngestionService(
        loader=loader,
        cleaner=cleaner,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
    )

    result = service.ingest(
        Path("document.pdf")
    )

    assert len(result) == 1

    loader.load.assert_called_once_with(
        Path("document.pdf")
    )

    cleaner.clean.assert_called_once()

    parser.parse.assert_called_once_with(
        ["clean page"],
        Path("document.pdf"),
    )

    chunker.chunk.assert_called_once()

    embedder.embed.assert_called_once_with(
        ["chunk"]
    )