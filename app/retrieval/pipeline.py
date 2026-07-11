from __future__ import annotations

import time

from app.schemas.document import EmbeddedChunk
from app.embeddings.base import BaseEmbedder
from app.retrieval.hybrid.hybrid import HybridRetriever
from app.schemas.retrieval import RetrievalResult, SearchQuery, RetrieverType


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline.

    Responsibilities
    ----------------
    1. Embed the user query.
    2. Perform hybrid retrieval.
    3. Package the results into a RetrievalResult.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        hybrid_retriever: HybridRetriever,
    ) -> None:
        self._embedder = embedder
        self._hybrid_retriever = hybrid_retriever

    def retrieve(
        self,
        query: SearchQuery,
    ) -> RetrievalResult:
        """
        Execute the retrieval pipeline.

        Parameters
        ----------
        query : SearchQuery
            User search query.

        Returns
        -------
        RetrievalResult
            Retrieved chunks and metadata.
        """

        start_time = time.perf_counter()

        query_embedding = self._embedder.embed_query(
            query.query,
        )

        retrieved_chunks = self._hybrid_retriever.search(
            query=query.query,
            query_embedding=query_embedding,
            top_k=query.top_k,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            retriever=RetrieverType.HYBRID,
            latency_ms=latency_ms,
        )
    
    def build(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        """
        Build all retrieval indexes.
        """

        self._hybrid_retriever.build(
            embedded_chunks,
        )