from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from app.embeddings.base import BaseEmbedder
from app.embeddings.exceptions import EmbeddingError
from app.schemas.document import Chunk, EmbeddedChunk
from app.config.settings import settings


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Generates dense embeddings using a SentenceTransformer model.
    """

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        """
        Initialize the embedding model.

        Parameters
        ----------
        model_name : str | None
            Optional HuggingFace model override.
        """
        try:
            self._model = SentenceTransformer(
                model_name or settings.embedding_model
            )
        except Exception as exc:
            raise EmbeddingError(
                "Failed to load embedding model."
            ) from exc

    def embed(
        self,
        chunks: Sequence[Chunk],
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for document chunks.

        Parameters
        ----------
        chunks : Sequence[Chunk]
            Chunks to embed.

        Returns
        -------
        list[EmbeddedChunk]
            Embedded chunks.

        Raises
        ------
        EmbeddingError
            If embedding generation fails.
        """
        try:
            texts = [
                chunk.content
                for chunk in chunks
            ]

            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return [
                EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding.tolist(),
                )
                for chunk, embedding in zip(
                    chunks,
                    embeddings,
                    strict=True,
                )
            ]

        except Exception as exc:
            raise EmbeddingError(
                "Failed to generate embeddings."
            ) from exc

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for a search query.

        Parameters
        ----------
        query : str
            User search query.

        Returns
        -------
        list[float]
            Query embedding.

        Raises
        ------
        EmbeddingError
            If embedding generation fails.
        """
        try:
            formatted_query = (
                "Represent this sentence for searching "
                f"relevant passages: {query}"
            )

            embedding = self._model.encode(
                formatted_query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return embedding.tolist()

        except Exception as exc:
            raise EmbeddingError(
                "Failed to generate query embedding."
            ) from exc