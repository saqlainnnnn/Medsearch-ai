from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from app.retrieval.reranker.base import BaseReranker
from app.retrieval.reranker.exceptions import RerankingError
from app.schemas.retrieval import RetrievedChunk


class CrossEncoderReranker(BaseReranker):
    """
    Reranks retrieved chunks using a Cross Encoder.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
    ) -> None:
        """
        Initialize the reranker.

        Parameters
        ----------
        model_name : str
            HuggingFace Cross Encoder model.
        """
        try:
            self._model = CrossEncoder(model_name)
        except Exception as exc:
            raise RerankingError(
                f"Failed to load reranker model: {model_name}"
            ) from exc

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank retrieved chunks.

        Parameters
        ----------
        query : str
            User query.

        chunks : Sequence[RetrievedChunk]
            Retrieved chunks.

        top_k : int | None
            Maximum number of chunks to return.

        Returns
        -------
        list[RetrievedChunk]
            Reranked chunks.

        Raises
        ------
        RerankingError
            If reranking fails.
        """
        if not chunks:
            return []

        try:
            pairs = [
                (query, chunk.chunk.content)
                for chunk in chunks
            ]

            scores = self._model.predict(pairs)

            scored_chunks = sorted(
                zip(chunks, scores, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )

            if top_k is not None:
                scored_chunks = scored_chunks[:top_k]

            reranked: list[RetrievedChunk] = []

            for rank, (chunk, score) in enumerate(
                scored_chunks,
                start=1,
            ):
                reranked.append(
                    chunk.model_copy(
                        update={
                            "retrieval_score": float(score),
                            "rank": rank,
                        }
                    )
                )

            return reranked

        except Exception as exc:
            raise RerankingError(
                "Failed to rerank retrieved chunks."
            ) from exc