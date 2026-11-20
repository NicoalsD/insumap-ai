"""Text generators: the OpenAI-compatible adapter, the redacting decorator and the
offline generator used by tests and CI."""

from __future__ import annotations

from typing import Any, Final, override

import openai
from openai import OpenAI

from insumap_ai.providers.base import (
    ChatMessage,
    ModelInfo,
    Prompt,
    TextGenerationError,
    TextGenerator,
)
from insumap_ai.redaction import PhiRedactor

PROMPT_VERSION: Final = "prompt-v1"
FINISHED: Final = "stop"


class OpenAiCompatibleTextGeneratorAdapter(TextGenerator):
    """Calls any provider with the OpenAI Chat Completions API (Gemini, DeepSeek,
    OpenRouter...). Only ``LLM_BASE_URL``, ``LLM_MODEL`` and ``LLM_API_KEY`` choose
    it. Temperature 0, no retries, never logs the prompt.

    Pattern: P06 Adapter
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        max_output_tokens: int,
        reasoning_effort: str | None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort

    @property
    @override
    def info(self) -> ModelInfo:
        return ModelInfo(name=self._model, version=f"0.1.0+{PROMPT_VERSION}")

    @override
    def generate(self, prompt: Prompt, *, json_output: bool) -> str:
        messages: list[Any] = [{"role": "system", "content": prompt.system}]
        messages.extend(
            {"role": message.role.value, "content": message.content}
            for message in prompt.messages
        )
        options: dict[str, Any] = {}
        if json_output:
            options["response_format"] = {"type": "json_object"}
        if self._reasoning_effort:
            options["reasoning_effort"] = self._reasoning_effort
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0,
                max_tokens=self._max_tokens,
                **options,
            )
        except openai.OpenAIError as error:
            # The type is enough to diagnose (429, timeout...); never the body.
            raise TextGenerationError(type(error).__name__) from error
        if not completion.choices:
            raise TextGenerationError("NO_CHOICES")
        choice = completion.choices[0]
        content = choice.message.content
        if choice.finish_reason != FINISHED or not content or not content.strip():
            raise TextGenerationError(f"FINISH_{choice.finish_reason}")
        return str(content)


class PhiRedactingTextGenerator(TextGenerator):
    """Removes names, documents, record numbers and contact data before calling the
    LLM, even though the backend already sends pseudonyms. Last barrier before the
    network.

    Pattern: P09 Decorator
    """

    def __init__(self, inner: TextGenerator, redactor: PhiRedactor) -> None:
        self._inner = inner
        self._redactor = redactor

    @property
    @override
    def info(self) -> ModelInfo:
        return self._inner.info

    @override
    def generate(self, prompt: Prompt, *, json_output: bool) -> str:
        clean = Prompt(
            system=self._redactor.redact(prompt.system).text,
            messages=tuple(
                ChatMessage(m.role, self._redactor.redact(m.content).text)
                for m in prompt.messages
            ),
        )
        return self._inner.generate(clean, json_output=json_output)


class OfflineTextGenerator(TextGenerator):
    """No LLM: every call fails, so IA5 and IA6 answer with their deterministic
    template (``is_fallback = true``). Keeps CI and tests off the network."""

    @property
    @override
    def info(self) -> ModelInfo:
        return ModelInfo(name="offline-template", version="offline@1.0.0")

    @override
    def generate(self, prompt: Prompt, *, json_output: bool) -> str:
        raise TextGenerationError("OFFLINE")
