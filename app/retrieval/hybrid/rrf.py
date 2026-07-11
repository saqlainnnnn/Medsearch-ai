from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from app.schemas.retrieval import RetrievedChunk


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) algorithm.

    Combines multiple ranked retrieval results into a single ranking.
    """

    def __init__(
        self,
        k: int = 60,
    ) -> None:
        """
        Initialize the RRF algorithm.

        Parameters
        ----------
        k : int
            Ranking constant used in the RRF score.
        """
        self._k = k

    def fuse(
        self,
        rankings: Sequence[Sequence[RetrievedChunk]],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Fuse multiple ranked retrieval results.

        Parameters
        ----------
        rankings : Sequence[Sequence[RetrievedChunk]]
            Ranked retrieval outputs.

        top_k : int
            Number of results to return.

        Returns
        -------
        list[RetrievedChunk]
            Fused ranking.
        """
        scores: dict[tuple, float] = defaultdict(float)
        chunks: dict[tuple, RetrievedChunk] = {}

        for ranking in rankings:
            for chunk in ranking:
                key = (
                    chunk.chunk.document_id,
                    chunk.chunk.chunk_index,
                )

                scores[key] += 1.0 / (
                    self._k + chunk.rank
                )

                chunks[key] = chunk

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[RetrievedChunk] = []

        for rank, (key, score) in enumerate(
            ranked[:top_k],
            start=1,
        ):
            chunk = chunks[key]

            results.append(
                chunk.model_copy(
                    update={
                        "retrieval_score": score,
                        "rank": rank,
                    }
                )
            )

        return results