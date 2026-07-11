from __future__ import annotations

from collections.abc import Sequence

from app.schemas.retrieval import RetrievedChunk


SYSTEM_PROMPT = """You are MedSearch AI, an evidence-grounded medical assistant.

You must answer ONLY using the provided context.

Rules:
1. Do not fabricate medical facts.
2. If the context is insufficient, clearly state that the available documents do not provide enough information.
3. Do not mention information that is not supported by the retrieved context.
4. Write concise, professional, evidence-based responses.
5. Do not invent citations or references.
6. Do not state that you are an AI assistant.
"""


def build_medical_prompt(
    question: str,
    contexts: Sequence[RetrievedChunk],
) -> str:
    """
    Build a prompt for evidence-grounded medical question answering.
    """

    context_text = "\n\n".join(
        f"[Context {i}]\n{chunk.chunk.content}"
        for i, chunk in enumerate(
            contexts,
            start=1,
        )
    )

    return f"""{SYSTEM_PROMPT}

Retrieved Context:
{context_text}

Question:
{question}

Answer:
"""