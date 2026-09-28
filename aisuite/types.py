"""Normalized response types — the OpenAI Chat Completions shape is our internal
contract. Every provider, frontier or local, returns these dataclasses so calling
code never sees provider-specific objects.

`Message.tool_calls` is part of the contract from day one: it is the seam the
future agent/tool-calling layer plugs into without touching any provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Function:
    name: str
    arguments: str  # JSON-encoded string, per OpenAI convention


@dataclass
class ToolCall:
    id: str
    function: Function
    type: str = "function"


@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None  # set on role="tool" result messages
    name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            out["name"] = self.name
        return out


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: Optional[str] = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatCompletionResponse:
    id: str
    model: str
    choices: list[Choice]
    usage: Optional[Usage] = None
    created: int = 0
    provider: str = ""  # our addition: which backend served this response

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "provider": self.provider,
            "created": self.created,
            "choices": [
                {
                    "index": c.index,
                    "finish_reason": c.finish_reason,
                    "message": c.message.to_dict(),
                }
                for c in self.choices
            ],
            "usage": (
                {
                    "prompt_tokens": self.usage.prompt_tokens,
                    "completion_tokens": self.usage.completion_tokens,
                    "total_tokens": self.usage.total_tokens,
                }
                if self.usage
                else None
            ),
        }
