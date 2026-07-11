from __future__ import annotations

from cerebras.cloud.sdk import Cerebras

from app.config.settings import settings
from app.llm.base import BaseLLM
from app.llm.exceptions import LLMError


class CerebrasLLM(BaseLLM):
    """
    Cerebras Cloud language model client.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        """
        Initialize the Cerebras client.
        """
        self._model = model or settings.llm_model
        self._temperature = temperature
        self._max_tokens = max_tokens

        try:
            self._client = Cerebras(
                api_key=settings.cerebras_api_key,
            )

        except Exception as exc:
            raise LLMError(
                "Failed to initialize Cerebras client."
            ) from exc

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate text from a prompt.

        Parameters
        ----------
        prompt : str
            Prompt to send to the model.

        Returns
        -------
        str
            Generated text.

        Raises
        ------
        LLMError
            If generation fails.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=self._temperature,
                max_completion_tokens=self._max_tokens,
            )

            content = response.choices[0].message.content

            if content is None:
                raise LLMError(
                    "Model returned an empty response."
                )

            return content

        except LLMError:
            raise

        except Exception:
            import traceback

            traceback.print_exc()
            raise