"""Advisory board pipeline — a sibling of `council/` on the same aisuite engine.

Where the council is a *study* tool (baseline vs preserving, groupthink metric),
the advisory board is an *application*: diverse models wear assigned expert
lenses, optionally grounded in a briefing document, and the output is a board
recommendation that explicitly preserves dissent — reusing the council's idea
ledger, dissent-preservation, faithfulness check, verification, cost telemetry,
rating, and export building blocks.
"""

from .advisory import AdvisoryResult, run_advisory, to_dict
from .lenses import AdvisoryConfig, Lens, DEFAULT_LENSES

__all__ = [
    "run_advisory",
    "AdvisoryResult",
    "to_dict",
    "AdvisoryConfig",
    "Lens",
    "DEFAULT_LENSES",
]
