from __future__ import annotations

from collections.abc import Sequence

import faiss
import numpy as np

from app.retrieval.dense.base import BaseDenseRetriever
from app.retrieval.dense.exceptions import DenseRetrievalError
from app.schemas.document import EmbeddedChunk
from app.schemas.retrieval import RetrievedChunk, RetrieverType


class FAISSDenseRetriever(BaseDenseRetriever):
    """
    Dense retriever backed by a FAISS index.
    """

    def __init__(
        self,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the dense retriever.

        Parameters
        ----------
        normalize : bool
            Whether to L2-normalize embeddings before indexing
            and querying.
        """
        self._normalize = normalize
        self._index: faiss.Index | None = None
        self._embedded_chunks: list[EmbeddedChunk] = []

    def _create_index(
        self,
        dimension: int,
    ) -> faiss.Index:
        """
        Create a FAISS index.

        Parameters
        ----------
        dimension : int
            Embedding dimension.

        Returns
        -------
        faiss.Index
            Initialized FAISS index.
        """
        return faiss.IndexFlatIP(dimension)

    def build(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> None:
        """
        Build the FAISS index.

        Parameters
        ----------
        embedded_chunks : Sequence[EmbeddedChunk]
            Embedded chunks to index.

        Raises
        ------
        DenseRetrievalError
            If indexing fails.
        """
        if not embedded_chunks:
            raise DenseRetrievalError(
                "No embedded chunks provided."
            )

        try:
            vectors = np.asarray(
                [chunk.embedding for chunk in embedded_chunks],
                dtype=np.float32,
            )

            if self._normalize:
                faiss.normalize_L2(vectors)

            self._index = self._create_index(
                vectors.shape[1],
            )

            self._index.add(vectors)

            self._embedded_chunks = list(embedded_chunks)

        except Exception as exc:
            raise DenseRetrievalError(
                "Failed to build FAISS index."
            ) from exc

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Search the FAISS index.

        Parameters
        ----------
        query_embedding : Sequence[float]
            Embedded query.

        top_k : int
            Number of results to retrieve.

        Returns
        -------
        list[RetrievedChunk]
            Retrieved chunks ordered by similarity.

        Raises
        ------
        DenseRetrievalError
            If search fails.
        """
        if self._index is None:
            raise DenseRetrievalError(
                "The FAISS index has not been built."
            )

        try:
            query = np.asarray(
                [query_embedding],
                dtype=np.float32,
            )

            if self._normalize:
                faiss.normalize_L2(query)

            scores, indices = self._index.search(
                query,
                top_k,
            )

            results: list[RetrievedChunk] = []

            for rank, (score, index) in enumerate(
                zip(scores[0], indices[0], strict=True),
                start=1,
            ):
                if index == -1:
                    continue

                embedded_chunk = self._embedded_chunks[index]

                results.append(
                    RetrievedChunk(
                        chunk=embedded_chunk.chunk,
                        retrieval_score=float(score),
                        rank=rank,
                        retriever=RetrieverType.DENSE,
                    )
                )

            return results

        except Exception as exc:
            raise DenseRetrievalError(
                "Failed to perform dense retrieval."
            ) from exc