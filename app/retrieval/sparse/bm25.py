from __future__ import annotations

from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from app.retrieval.sparse.base import BaseSparseRetriever
from app.retrieval.sparse.exceptions import SparseRetrievalError
from app.schemas.document import Chunk
from app.schemas.retrieval import RetrievedChunk, RetrieverType


class BM25Retriever(BaseSparseRetriever):
    """
    Sparse retriever backed by BM25.
    """

    def __init__(self) -> None:
        """
        Initialize the BM25 retriever.
        """
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize text for BM25.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        list[str]
            Lowercased whitespace-separated tokens.
        """
        return text.lower().split()

    def build(
        self,
        chunks: Sequence[Chunk],
    ) -> None:
        """
        Build the BM25 index.

        Parameters
        ----------
        chunks : Sequence[Chunk]
            Chunks to index.

        Raises
        ------
        SparseRetrievalError
            If indexing fails.
        """
        if not chunks:
            raise SparseRetrievalError(
                "No chunks provided."
            )

        try:
            corpus = [
                self._tokenize(chunk.content)
                for chunk in chunks
            ]

            self._bm25 = BM25Okapi(corpus)
            self._chunks = list(chunks)

        except Exception as exc:
            raise SparseRetrievalError(
                "Failed to build BM25 index."
            ) from exc

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Search the BM25 index.

        Parameters
        ----------
        query : str
            User query.

        top_k : int
            Number of chunks to retrieve.

        Returns
        -------
        list[RetrievedChunk]
            Ranked retrieved chunks.

        Raises
        ------
        SparseRetrievalError
            If searching fails.
        """
        if self._bm25 is None:
            raise SparseRetrievalError(
                "The BM25 index has not been built."
            )

        try:
            query_tokens = self._tokenize(query)

            scores = self._bm25.get_scores(query_tokens)

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k]

            results: list[RetrievedChunk] = []

            for rank, index in enumerate(
                ranked_indices,
                start=1,
            ):
                results.append(
                    RetrievedChunk(
                        chunk=self._chunks[index],
                        retrieval_score=float(scores[index]),
                        rank=rank,
                        retriever=RetrieverType.BM25,
                    )
                )

            return results

        except Exception as exc:
            raise SparseRetrievalError(
                "Failed to perform sparse retrieval."
            ) from exc