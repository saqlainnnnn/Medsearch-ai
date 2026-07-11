from __future__ import annotations

from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker.base import BaseReranker
from app.schemas.retrieval import RetrievalResult, SearchQuery


class SearchService:
    """
    High-level search service.

    Coordinates retrieval and reranking.
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        reranker: BaseReranker,
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._reranker = reranker

    def search(
        self,
        query: SearchQuery,
    ) -> RetrievalResult:
        """
        Execute the complete search pipeline.

        Parameters
        ----------
        query : SearchQuery
            User search query.

        Returns
        -------
        RetrievalResult
            Reranked retrieval results.
        """

        retrieval_result = self._retrieval_pipeline.retrieve(
            query,
        )

        reranked_chunks = self._reranker.rerank(
            query=query.query,
            chunks=retrieval_result.retrieved_chunks,
            top_k=query.top_k,
        )

        return retrieval_result.model_copy(
            update={
                "retrieved_chunks": reranked_chunks,
            }
        )