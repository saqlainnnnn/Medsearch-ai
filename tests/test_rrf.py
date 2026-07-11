from uuid import UUID,uuid4

from app.retrieval.hybrid.rrf import ReciprocalRankFusion
from app.schemas.document import Chunk
from app.schemas.retrieval import RetrievedChunk, RetrieverType


def create_chunk(
    chunk_index: int,
    rank: int,
    retriever: RetrieverType,
    document_id: UUID | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            document_id=document_id or uuid4(),
            chunk_index=chunk_index,
            content=f"Chunk {chunk_index}",
            token_count=2,
        ),
        retrieval_score=1.0,
        rank=rank,
        retriever=retriever,
    )


def test_fuse_single_ranking() -> None:
    rrf = ReciprocalRankFusion()

    ranking = [
        create_chunk(0, 1, RetrieverType.DENSE),
        create_chunk(1, 2, RetrieverType.DENSE),
    ]

    results = rrf.fuse([ranking])

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_fuse_multiple_rankings() -> None:
    rrf = ReciprocalRankFusion()

    shared = uuid4()

    dense = [
        create_chunk(0, 1, RetrieverType.DENSE, shared),
        create_chunk(1, 2, RetrieverType.DENSE, shared),
    ]

    sparse = [
        create_chunk(0, 1, RetrieverType.BM25, shared),
        create_chunk(2, 2, RetrieverType.BM25, shared),
    ]

    results = rrf.fuse([dense, sparse])

    assert len(results) == 3


def test_duplicate_chunks_are_merged() -> None:
    rrf = ReciprocalRankFusion()

    shared = uuid4()

    dense = [
        RetrievedChunk(
            chunk=Chunk(
                document_id=shared,
                chunk_index=0,
                content="same",
                token_count=1,
            ),
            retrieval_score=0.9,
            rank=1,
            retriever=RetrieverType.DENSE,
        )
    ]

    sparse = [
        RetrievedChunk(
            chunk=Chunk(
                document_id=shared,
                chunk_index=0,
                content="same",
                token_count=1,
            ),
            retrieval_score=12.0,
            rank=1,
            retriever=RetrieverType.BM25,
        )
    ]

    results = rrf.fuse([dense, sparse])

    assert len(results) == 1


def test_top_k_respected() -> None:
    rrf = ReciprocalRankFusion()

    ranking = [
        create_chunk(i, i + 1, RetrieverType.DENSE)
        for i in range(5)
    ]

    results = rrf.fuse(
        [ranking],
        top_k=2,
    )

    assert len(results) == 2


def test_ranks_are_reassigned() -> None:
    rrf = ReciprocalRankFusion()

    ranking = [
        create_chunk(0, 5, RetrieverType.DENSE),
        create_chunk(1, 10, RetrieverType.DENSE),
    ]

    results = rrf.fuse([ranking])

    assert results[0].rank == 1
    assert results[1].rank == 2


def test_scores_are_updated() -> None:
    rrf = ReciprocalRankFusion()

    ranking = [
        create_chunk(0, 1, RetrieverType.DENSE),
    ]

    results = rrf.fuse([ranking])

    assert results[0].retrieval_score != 1.0