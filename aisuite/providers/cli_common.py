"""Shared plumbing for CLI-backed providers (Claude Code, Codex): providers that
shell out to a vendor's own command-line client so a seat runs on the user's
consumer subscription login instead of an API key.

Not a provider itself (the factory only discovers *_provider.py files).
"""

from __future__ import annotations

import subprocess
from typing import Any, Optional

from ..types import ChatCompletionResponse, Choice, Message, Usage


def split_messages(messages: list[Any]) -> tuple[str, str]:
    """(system text, prompt text). A single user turn becomes the prompt as-is;
    a longer history is flattened into a labelled transcript."""
    system: list[str] = []
    turns: list[tuple[str, str]] = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if not isinstance(content, str):  # content blocks -> text parts only
            content = "\n".join(
                b.get("text", "") for b in (content or []) if isinstance(b, dict) and b.get("type") == "text"
            )
        if role == "system":
            system.append(content or "")
        elif role in ("user", "assistant"):
            turns.append((role, content or ""))
    if len(turns) == 1 and turns[0][0] == "user":
        prompt = turns[0][1]
    else:
        prompt = "\n\n".join(f"{role.capitalize()}: {text}" for role, text in turns)
    return "\n\n".join(s for s in system if s.strip()), prompt


def run_cli(cmd: list[str], stdin: str, cwd: str, timeout: float) -> subprocess.CompletedProcess:
    """Run a CLI with the prompt on stdin. `cwd` must be an empty directory so
    the tool sees no project files, CLAUDE.md, or memory."""
    return subprocess.run(
        cmd, input=stdin, capture_output=True, text=True, cwd=cwd, timeout=timeout,
    )


def response(provider: str, model: str, text: str, *, prompt_tokens: int = 0,
             completion_tokens: int = 0, request_id: Optional[str] = None) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=request_id or provider,
        model=model,
        choices=[Choice(index=0, message=Message(role="assistant", content=text), finish_reason="stop")],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens),
        provider=provider,
    )
