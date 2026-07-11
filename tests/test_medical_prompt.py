from app.prompts.medical_prompt import (
    SYSTEM_PROMPT,
    build_medical_prompt,
)
from app.schemas.document import Chunk
from app.schemas.retrieval import RetrievedChunk, RetrieverType


def create_chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            document_id="00000000-0000-0000-0000-000000000001",
            chunk_index=index,
            content=f"Medical context {index}",
            token_count=2,
        ),
        retrieval_score=1.0,
        rank=index + 1,
        retriever=RetrieverType.HYBRID,
    )


def test_system_prompt_present() -> None:
    prompt = build_medical_prompt(
        question="What is diabetes?",
        contexts=[],
    )

    assert SYSTEM_PROMPT in prompt


def test_question_present() -> None:
    prompt = build_medical_prompt(
        question="What is diabetes?",
        contexts=[],
    )

    assert "What is diabetes?" in prompt


def test_single_context_present() -> None:
    prompt = build_medical_prompt(
        question="Question",
        contexts=[create_chunk(0)],
    )

    assert "Medical context 0" in prompt


def test_multiple_contexts_present() -> None:
    prompt = build_medical_prompt(
        question="Question",
        contexts=[
            create_chunk(0),
            create_chunk(1),
        ],
    )

    assert "Medical context 0" in prompt
    assert "Medical context 1" in prompt


def test_context_numbering() -> None:
    prompt = build_medical_prompt(
        question="Question",
        contexts=[
            create_chunk(0),
            create_chunk(1),
        ],
    )

    assert "[Context 1]" in prompt
    assert "[Context 2]" in prompt


def test_empty_context_is_valid() -> None:
    prompt = build_medical_prompt(
        question="Question",
        contexts=[],
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0