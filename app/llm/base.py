from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Base interface for language models.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate text from a prompt.

        Parameters
        ----------
        prompt : str
            Prompt sent to the language model.

        Returns
        -------
        str
            Generated response.
        """
        raise NotImplementedError