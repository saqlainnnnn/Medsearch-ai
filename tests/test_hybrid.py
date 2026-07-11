from unittest.mock import Mock

from app.retrieval.hybrid.hybrid import HybridRetriever
from app.schemas.retrieval import RetrievedChunk


def test_hybrid_calls_dense_sparse_and_rrf() -> None:
    dense = Mock()
    sparse = Mock()
    rrf = Mock()

    dense.search.return_value = ["dense"]
    sparse.search.return_value = ["sparse"]
    rrf.fuse.return_value = ["final"]

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        rrf=rrf,
    )

    results = hybrid.search(
        query="diabetes",
        query_embedding=[0.1, 0.2],
        top_k=5,
    )

    dense.search.assert_called_once()
    sparse.search.assert_called_once()
    rrf.fuse.assert_called_once()

    assert results == ["final"]


def test_top_k_forwarded() -> None:
    dense = Mock()
    sparse = Mock()
    rrf = Mock()

    dense.search.return_value = []
    sparse.search.return_value = []
    rrf.fuse.return_value = []

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        rrf=rrf,
    )

    hybrid.search(
        query="diabetes",
        query_embedding=[0.1, 0.2],
        top_k=7,
    )

    dense.search.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        top_k=7,
    )

    sparse.search.assert_called_once_with(
        query="diabetes",
        top_k=7,
    )

    rrf.fuse.assert_called_once()