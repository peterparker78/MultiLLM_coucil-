"""Council configuration. Defaults run entirely through one OpenRouter key
(matching Karpathy's setup); swap in `anthropic:` or a local `ollama:` seat freely.
Override via env vars or CLI flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _seats_from_env() -> list[str]:
    raw = os.getenv("COUNCIL_SEATS")
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return [
        "openrouter:openai/gpt-6-sol",
        "openrouter:anthropic/claude-opus-5.5",
        "openrouter:google/gemini-3.1-pro-preview",
        "openrouter:x-ai/grok-4.7",
        # add a private/offline voice: "ollama:llama3.1"
    ]


@dataclass
class CouncilConfig:
    seats: list[str] = field(default_factory=_seats_from_env)
    # The chairman synthesizes; the clerk does extraction/clustering/judging.
    # Not Opus 5.5: its biosafety classifier refuses clerical work on clinical
    # content (a clinical-trial board had every clerk call refused, Sep 2026).
    # The clerk must NOT share the chairman's model: it judges idea survival in —
    # and audits the faithfulness of — the chairman's synthesis, and same-model
    # self-judging biases the groupthink gap. Keep it capable so the ledger and
    # survival scoring stay honest.
    chairman: str = field(
        default_factory=lambda: os.getenv("COUNCIL_CHAIRMAN", "openrouter:google/gemini-3.1-pro-preview")
    )
    clerk: str = field(
        default_factory=lambda: os.getenv("COUNCIL_CLERK", "openrouter:anthropic/claude-opus-5")
    )
    red_team: str = field(
        default_factory=lambda: os.getenv("COUNCIL_REDTEAM", "openrouter:anthropic/claude-opus-5")
    )
    # Phase B: optional fact-checking of the syntheses. The verifier must be a
    # web-capable model (an OpenRouter ':online' model) to actually check claims.
    verify: bool = field(default_factory=lambda: os.getenv("COUNCIL_VERIFY") == "1")
    verifier: str = field(
        default_factory=lambda: os.getenv("COUNCIL_VERIFIER", "openrouter:openai/gpt-6-luna:online")
    )
    # Hidden-profile / emergent-composite pass (cross-model fragment combination).
    compose: bool = field(default_factory=lambda: os.getenv("COUNCIL_COMPOSE") == "1")
    # Independent judge for composites — must differ from the chairman that proposes.
    # Strong + reliable by default; a near-free strong alternative is
    # openrouter:qwen/qwen3-235b-a22b-thinking-2507.
    compose_judge: str = field(
        default_factory=lambda: os.getenv("COUNCIL_COMPOSE_JUDGE", "openrouter:openai/gpt-6-astra")
    )
    temperature: float = 0.7
    # Reasoning seats (e.g. Kimi K3) spend thousands of tokens thinking before
    # answering, and the cap covers both; 4096 left them with no visible answer.
    max_tokens: int = 16384
    # A member answer shorter than this (chars), or far below the cohort median,
    # is flagged "thin" and excluded from consensus rather than counted as a vote.
    min_answer_chars: int = 120
    brief_max_chars: int = 40000  # shared budget across all briefing documents
