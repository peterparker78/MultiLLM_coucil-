"""Parallel fan-out: send the SAME prompt to many models at once.

This is the one library primitive the council needs. It runs each model
independently and concurrently — which also enforces the property the
groupthink research says matters most: members answer BLIND to each other,
with no cross-exposure during generation.

Per-model errors are captured, not raised, so a single failing seat (bad key,
timeout, model offline) does not sink the whole council.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from .types import ChatCompletionResponse


@dataclass
class FanOutResult:
    model: str  # the full "provider:model" string as requested
    response: Optional[ChatCompletionResponse] = None
    error: Optional[Exception] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def text(self) -> Optional[str]:
        if self.response and self.response.choices:
            return self.response.choices[0].message.content
        return None


def fan_out(
    client: Any,
    models: list[str],
    messages: list[Any],
    *,
    max_workers: Optional[int] = None,
    **kwargs: Any,
) -> list[FanOutResult]:
    """Call every model in `models` with the same `messages`, concurrently.

    Returns one FanOutResult per model, in the same order as `models`.
    Extra kwargs (temperature, max_tokens, tools, ...) are forwarded to each call.
    """

    def _call(model: str) -> FanOutResult:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            return FanOutResult(model=model, response=resp)
        except Exception as e:  # capture per-seat so the council survives one failure
            return FanOutResult(model=model, error=e)

    workers = max_workers or max(1, len(models))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_call, models))
