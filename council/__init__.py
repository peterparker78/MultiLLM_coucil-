"""A groupthink-resistant LLM council, built on the aisuite library.

Reproduces Karpathy's 3-stage council as a BASELINE and adds a diversity-
preserving variant (idea-ledger with provenance + blind synthesis from the
ledger), then measures the unique-idea survival rate of each so you can SEE the
groupthink effect and the fix on the same inputs.
"""

from .council import RunResult, ModeResult, run_council
from .types import IdeaCard, IdeaCluster, MemberAnswer

__all__ = [
    "run_council",
    "RunResult",
    "ModeResult",
    "MemberAnswer",
    "IdeaCard",
    "IdeaCluster",
]
