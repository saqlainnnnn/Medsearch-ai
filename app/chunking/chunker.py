from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunking.base import BaseChunker
from app.chunking.exceptions import ChunkingError
from app.schemas.document import Chunk, Document


class RecursiveChunker(BaseChunker):
    """
    Splits a Document into overlapping text chunks using
    LangChain's RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        """
        Initialize the recursive text chunker.

        Parameters
        ----------
        chunk_size : int
            Maximum size of each chunk in characters.

        chunk_overlap : int
            Number of overlapping characters between chunks.
        """

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        """
        Split a document into searchable chunks.

        Parameters
        ----------
        document : Document
            Parsed document.

        Returns
        -------
        list[Chunk]
            Searchable chunks.

        Raises
        ------
        ChunkingError
            If chunk creation fails.
        """

        try:
            texts = self._splitter.split_text(document.content)

            chunks: list[Chunk] = []

            for index, text in enumerate(texts):
                chunks.append(
                    Chunk(
                        document_id=document.document_id,
                        chunk_index=index,
                        content=text,
                        page_number=None,
                        section=None,
                        token_count=len(text.split()),
                    )
                )

            return chunks

        except Exception as exc:
            raise ChunkingError(
                "Failed to chunk document."
            ) from exc