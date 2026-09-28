"""Anthropic (frontier). The heavyweight adapter: Anthropic's Messages API differs
from OpenAI's in several ways, so this module does real translation in both
directions. It is the template for any non-OpenAI-shaped backend (Google, Bedrock,
Cohere) you add later.

Differences handled here:
  - system prompt is a top-level arg, not a message
  - tool calls are `tool_use` content blocks; tool results are `tool_result` blocks
  - finish reasons and token-usage field names differ
  - max_tokens is required
"""

from __future__ import annotations

import json
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

_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}
_DEFAULT_MAX_TOKENS = 1024


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an object (messages may arrive as either)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class AnthropicMessageConverter:
    def convert_messages(self, messages: list[Any]) -> tuple[Any, list[dict[str, Any]]]:
        system: Any = None
        converted: list[dict[str, Any]] = []

        for m in messages:
            role = _get(m, "role")
            content = _get(m, "content")

            if role == "system":
                system = content
                continue

            if role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": _get(m, "tool_call_id"),
                                "content": content if content is not None else "",
                            }
                        ],
                    }
                )
                continue

            tool_calls = _get(m, "tool_calls")
            if role == "assistant" and tool_calls:
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = _get(tc, "function")
                    args = _get(fn, "arguments")
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": _get(tc, "id"),
                            "name": _get(fn, "name"),
                            "input": json.loads(args) if isinstance(args, str) else (args or {}),
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
                continue

            converted.append({"role": role, "content": content})

        return system, converted

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for t in tools:
            fn = t["function"]
            out.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return out

    def normalize_response(self, response: Any, model: str) -> ChatCompletionResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        function=Function(
                            name=block.name, arguments=json.dumps(block.input)
                        ),
                    )
                )

        message = Message(
            role="assistant",
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
        )
        finish_reason = _FINISH_REASON_MAP.get(
            response.stop_reason, response.stop_reason
        )
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        return ChatCompletionResponse(
            id=response.id,
            model=model,
            choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
            usage=usage,
            provider="anthropic",
        )


class AnthropicProvider(Provider):
    def __init__(self, **config: Any):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for this provider. "
                "Install with: pip install anthropic"
            ) from e
        self.client = anthropic.Anthropic(**config)
        self.converter = AnthropicMessageConverter()

    def chat_completions_create(
        self, model: str, messages: list[Any], **kwargs: Any
    ) -> ChatCompletionResponse:
        kwargs.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)

        system, converted = self.converter.convert_messages(messages)
        if system is not None:
            kwargs["system"] = system

        tools = kwargs.get("tools")
        if tools:
            kwargs["tools"] = self.converter.convert_tools(tools)
        elif "tools" in kwargs:
            kwargs.pop("tools")  # don't pass an empty tools list

        response = self.client.messages.create(
            model=model, messages=converted, **kwargs
        )
        return self.converter.normalize_response(response, model)
