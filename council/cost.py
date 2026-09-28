"""Cost telemetry. A MeteredClient transparently wraps the real client and
records token usage per model on every call; the Meter prices it from
OpenRouter's live rate card. Ollama/local = free; unpriced models are flagged.

The goal is a visible feedback loop: every run reports tokens + $ per model so
you can see exactly where spend goes and tune (e.g. which role to move to a
cheaper model).
"""

from __future__ import annotations

import json
import threading
import urllib.request
from typing import Any, Optional

_PRICES: Optional[dict[str, tuple[float, float]]] = None  # slug -> ($/prompt-tok, $/completion-tok)


def get_prices() -> dict[str, tuple[float, float]]:
    """Fetch OpenRouter's public rate card once and cache it. Safe on failure
    (returns {} → costs report as unknown rather than crashing)."""
    global _PRICES
    if _PRICES is not None:
        return _PRICES
    out: dict[str, tuple[float, float]] = {}
    try:
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=6) as r:
            data = json.load(r).get("data", [])
        for m in data:
            p = m.get("pricing") or {}
            try:
                out[m["id"]] = (float(p.get("prompt", 0) or 0), float(p.get("completion", 0) or 0))
            except (TypeError, ValueError):
                continue
    except Exception:
        out = {}
    _PRICES = out
    return _PRICES


def _price_for(model: str) -> Optional[tuple[float, float]]:
    """Return ($/prompt-tok, $/completion-tok) for a 'provider:model' string, or
    None if the price is unknown."""
    provider, _, slug = model.partition(":")
    if provider in ("ollama", "lmstudio", "claudecode", "codex"):
        return (0.0, 0.0)  # local / cloud-quota / subscription login: no per-token API charge
    if provider == "openrouter":
        # ':online' enables web search; price the base model (the web plugin
        # adds a separate per-request surcharge not captured here).
        return get_prices().get(slug.removesuffix(":online"))
    return None  # e.g. openai: direct — price not in the OpenRouter card


class Meter:
    def __init__(self) -> None:
        self._by_model: dict[str, list[int]] = {}  # model -> [calls, prompt_tok, completion_tok]
        self._lock = threading.Lock()  # record() is hit concurrently from fan_out threads

    def record(self, model: str, usage: Any) -> None:
        with self._lock:
            e = self._by_model.setdefault(model, [0, 0, 0])
            e[0] += 1
            if usage is not None:
                e[1] += int(getattr(usage, "prompt_tokens", 0) or 0)
                e[2] += int(getattr(usage, "completion_tokens", 0) or 0)

    def summary(self) -> dict[str, Any]:
        with self._lock:  # snapshot, then price outside the lock (pricing may hit the network)
            snapshot = {m: list(v) for m, v in self._by_model.items()}
        rows = []
        total_cost = 0.0
        total_tokens = 0
        partial = False
        for model, (calls, pt, ct) in sorted(snapshot.items()):
            price = _price_for(model)
            if price is None:
                cost: Optional[float] = None
                partial = True
            else:
                cost = pt * price[0] + ct * price[1]
                total_cost += cost
            total_tokens += pt + ct
            rows.append({
                "model": model,
                "calls": calls,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cost_usd": round(cost, 4) if cost is not None else None,
            })
        return {
            "per_model": rows,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "cost_partial": partial,  # True if some models had no known price
        }


class _MeteredCompletions:
    def __init__(self, inner: Any, meter: Meter) -> None:
        self._inner = inner
        self._meter = meter

    def create(self, model: str, messages: list, **kw: Any) -> Any:
        resp = self._inner.chat.completions.create(model=model, messages=messages, **kw)
        try:
            self._meter.record(model, getattr(resp, "usage", None))
        except Exception:
            pass  # telemetry must never break a run
        return resp


class _MeteredChat:
    def __init__(self, inner: Any, meter: Meter) -> None:
        self.completions = _MeteredCompletions(inner, meter)


class MeteredClient:
    """Drop-in wrapper exposing .chat.completions.create, recording every call."""

    def __init__(self, inner: Any, meter: Meter) -> None:
        self._inner = inner
        self.chat = _MeteredChat(inner, meter)
