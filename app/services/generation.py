from __future__ import annotations

from app.llm.base import BaseLLM
from app.prompts.medical_prompt import build_medical_prompt
from app.schemas.api import Citation, MedicalAnswer
from app.schemas.retrieval import RetrievalResult


class GenerationService:
    """
    Generates evidence-grounded medical answers.
    """

    def __init__(
        self,
        llm: BaseLLM,
    ) -> None:
        self._llm = llm

    def generate(
        self,
        retrieval_result: RetrievalResult,
    ) -> MedicalAnswer:
        """
        Generate a medical answer from retrieved evidence.

        Parameters
        ----------
        retrieval_result : RetrievalResult
            Retrieved and reranked context.

        Returns
        -------
        MedicalAnswer
            Generated answer with citations.
        """

        prompt = build_medical_prompt(
            question=retrieval_result.query.query,
            contexts=retrieval_result.retrieved_chunks,
        )

        answer = self._llm.generate(prompt)

        citations = [
            Citation(
                source_title="Unknown",
                page_number=chunk.chunk.page_number,
                section=chunk.chunk.section,
            )
            for chunk in retrieval_result.retrieved_chunks
        ]

        return MedicalAnswer(
            answer=answer,
            citations=citations,
        )