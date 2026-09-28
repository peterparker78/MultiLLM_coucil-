"""One JSON serializer for a RunResult, shared by the CLI and the web server."""

from __future__ import annotations

import dataclasses
from typing import Any

from .council import RunResult


def run_to_dict(run: RunResult) -> dict[str, Any]:
    return {
        "prompt": run.prompt,
        "members": [dataclasses.asdict(m) for m in run.members],
        "ledger": [
            {**dataclasses.asdict(c), "is_unique": c.is_unique} for c in run.ledger
        ],
        "best_raw": dataclasses.asdict(run.best_raw) if run.best_raw else None,
        "composites": run.composites,
        "warnings": run.warnings,
        "brief_used": run.brief_used,
        "cost": run.cost,
        "results": {
            mode: {
                "synthesis": res.synthesis,
                "unsupported_claims": res.unsupported_claims,
                "verification": res.verification,
                "survival": {
                    "unique_total": res.survival.unique_total,
                    "unique_survived": res.survival.unique_survived,
                    "unique_rate": res.survival.unique_rate,
                    "shared_total": res.survival.shared_total,
                    "shared_survived": res.survival.shared_survived,
                    "shared_rate": res.survival.shared_rate,
                },
            }
            for mode, res in run.results.items()
        },
    }
