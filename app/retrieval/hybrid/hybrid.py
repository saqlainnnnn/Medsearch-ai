from __future__ import annotations

from app.retrieval.dense.base import BaseDenseRetriever
from app.retrieval.hybrid.rrf import ReciprocalRankFusion
from app.retrieval.sparse.base import BaseSparseRetriever
from app.schemas.retrieval import RetrievedChunk

from collections.abc import Sequence
from app.schemas.document import EmbeddedChunk



class HybridRetriever:
    """
    Hybrid retriever that combines dense and sparse retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: BaseDenseRetriever,
        sparse_retriever: BaseSparseRetriever,
        rrf: ReciprocalRankFusion | None = None,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Parameters
        ----------
        dense_retriever : BaseDenseRetriever
            Dense retrieval implementation.

        sparse_retriever : BaseSparseRetriever
            Sparse retrieval implementation.

        rrf : ReciprocalRankFusion | None
            Fusion algorithm. If None, a default RRF instance is used.
        """
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._rrf = rrf or ReciprocalRankFusion()

    def build(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        """
        Build both dense and sparse retrieval indexes.

        Parameters
        ----------
        embedded_chunks : list[EmbeddedChunk]
            Embedded document chunks.
        """

        self._dense.build(
            embedded_chunks,
        )

        self._sparse.build(
            [chunk.chunk for chunk in embedded_chunks],
        )

    def search(
        self,
        query: str,
        query_embedding: Sequence[float],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Perform hybrid retrieval.

        Parameters
        ----------
        query : str
            User query.

        query_embedding : list[float]
            Dense embedding of the query.

        top_k : int
            Number of results to return.

        Returns
        -------
        list[RetrievedChunk]
            Fused retrieval results.
        """

        dense_results = self._dense.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        sparse_results = self._sparse.search(
            query=query,
            top_k=top_k,
        )

        return self._rrf.fuse(
            rankings=[
                dense_results,
                sparse_results,
            ],
            top_k=top_k,
        )