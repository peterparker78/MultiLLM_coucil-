"""Base for any backend that exposes an OpenAI-compatible /v1/chat/completions
endpoint. This is the cheap path: OpenAI itself, every local server (Ollama,
LM Studio, llama.cpp, vLLM), and many cloud relays (Groq, Together, OpenRouter,
xAI, DeepSeek) all subclass this with just a base URL and a few defaults.

Because the wire format already matches our internal contract, tool calls,
tool-result messages, and finish reasons flow through unchanged — there is no
per-message conversion to write.
"""

from __future__ import annotations

from typing import Any

from ..provider import Provider
from ..types import (
    ChatCompletionResponse,
    Choice,
    Function,
    Message,
    ToolCall,
    Usage,
)


class OpenAICompatibleProvider(Provider):
    # Subclasses override these to point at their endpoint.
    DEFAULT_BASE_URL: str | None = None
    DEFAULT_API_KEY: str | None = None
    DEFAULT_TIMEOUT: float | None = None

    def __init__(self, **config: Any):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for this provider. "
                "Install with: pip install openai"
            ) from e

        config.setdefault("base_url", self.DEFAULT_BASE_URL)
        config.setdefault("api_key", self.DEFAULT_API_KEY)
        if self.DEFAULT_TIMEOUT is not None:
            config.setdefault("timeout", self.DEFAULT_TIMEOUT)
        # Drop unset values so the SDK falls back to its own env-based defaults
        # (e.g. OPENAI_API_KEY) rather than receiving None.
        config = {k: v for k, v in config.items() if v is not None}
        self.client = openai.OpenAI(**config)

    def chat_completions_create(
        self, model: str, messages: list[Any], **kwargs: Any
    ) -> ChatCompletionResponse:
        response = self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return normalize_openai_response(response, provider=self._provider_key())

    def _provider_key(self) -> str:
        return type(self).__name__.replace("Provider", "").lower()


def normalize_openai_response(response: Any, provider: str = "") -> ChatCompletionResponse:
    """Map an OpenAI SDK ChatCompletion (or any duck-typed equivalent) to our type."""
    choices: list[Choice] = []
    for choice in response.choices:
        msg = choice.message
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type=getattr(tc, "type", "function"),
                    function=Function(
                        name=tc.function.name, arguments=tc.function.arguments
                    ),
                )
                for tc in msg.tool_calls
            ]
        choices.append(
            Choice(
                index=choice.index,
                message=Message(
                    role=msg.role, content=msg.content, tool_calls=tool_calls
                ),
                finish_reason=choice.finish_reason,
            )
        )

    usage = None
    if getattr(response, "usage", None):
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    return ChatCompletionResponse(
        id=getattr(response, "id", ""),
        model=getattr(response, "model", ""),
        choices=choices,
        usage=usage,
        created=getattr(response, "created", 0) or 0,
        provider=provider,
    )
