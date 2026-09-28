"""Data structures for the council. Kept separate from logic so the survival
metric and ledger can be unit-tested as pure functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemberAnswer:
    model: str  # full "provider:model" string
    label: str  # anonymized, e.g. "Member A" — what the chair/judge sees
    text: str = ""
    ok: bool = True  # did the call return without error
    error: Optional[str] = None
    # substantive classification: a returned-but-empty/thin answer is NOT a full
    # vote. Only status == "ok" members feed extraction, peer review, and selection.
    status: str = "ok"  # ok | thin | refused | failed

    @property
    def usable(self) -> bool:
        return self.status == "ok"


@dataclass
class IdeaCard:
    id: str
    text: str
    source_label: str  # which member produced it (anonymized)


@dataclass
class IdeaCluster:
    id: str
    summary: str
    source_labels: list[str]  # distinct members whose answers contained this idea
    merit: float = 0.0  # 0-10, judged on the idea's own quality, NOT how many said it
    must_include: bool = False  # set by the red-team rescue pass

    @property
    def is_unique(self) -> bool:
        """A minority idea: raised by exactly one member. These are the ones
        the research shows councils tend to discard."""
        return len(set(self.source_labels)) == 1

    @property
    def is_shared(self) -> bool:
        return not self.is_unique


@dataclass
class Survival:
    unique_total: int
    unique_survived: int
    shared_total: int
    shared_survived: int

    @property
    def unique_rate(self) -> float:
        return self.unique_survived / self.unique_total if self.unique_total else 0.0

    @property
    def shared_rate(self) -> float:
        return self.shared_survived / self.shared_total if self.shared_total else 0.0
