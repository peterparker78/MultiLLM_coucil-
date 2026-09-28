"""Idea-survival metric — the number that makes the groupthink effect visible.

For a given synthesis we know which ledger ideas survived into it. We split
survival by unique (minority) vs shared (consensus) ideas. The research predicts
a naive council keeps shared ideas at a much higher rate than unique ones; a
diversity-preserving council should close that gap.
"""

from __future__ import annotations

from .types import IdeaCluster, Survival


def compute_survival(ledger: list[IdeaCluster], surviving_ids: set[str]) -> Survival:
    unique = [c for c in ledger if c.is_unique]
    shared = [c for c in ledger if c.is_shared]
    return Survival(
        unique_total=len(unique),
        unique_survived=sum(1 for c in unique if c.id in surviving_ids),
        shared_total=len(shared),
        shared_survived=sum(1 for c in shared if c.id in surviving_ids),
    )
