"""Orchestrator: one set of independent answers, one shared idea ledger, two
synthesis strategies scored against that same ledger. This shared-ledger design
is what makes the baseline-vs-preserving comparison apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import llm
from .config import CouncilConfig
from .cost import Meter, MeteredClient
from .metrics import compute_survival
from .types import IdeaCluster, MemberAnswer, Survival

# progress(stage_key, human_label) — optional UI hook, called before each step.
ProgressFn = Callable[[str, str], None]


@dataclass
class ModeResult:
    mode: str  # "baseline" | "preserving"
    synthesis: str
    survival: Survival
    unsupported_claims: list = field(default_factory=list)  # chairman-introduced, unsourced
    verification: list = field(default_factory=list)  # [{claim, verdict, note}] if fact-checked


@dataclass
class RunResult:
    prompt: str
    members: list[MemberAnswer]
    ledger: list[IdeaCluster]
    best_raw: Optional[MemberAnswer]
    results: dict[str, ModeResult] = field(default_factory=dict)
    composites: list = field(default_factory=list)  # emergent cross-model composites
    warnings: list[str] = field(default_factory=list)  # degraded stages, config hazards
    brief_used: bool = False  # briefing documents were shown to every seat
    cost: Optional[dict] = None  # token + $ telemetry per model


def run_council(
    client: Any,
    prompt: str,
    config: Optional[CouncilConfig] = None,
    modes: tuple[str, ...] = ("baseline", "preserving"),
    progress: Optional[ProgressFn] = None,
    brief_text: str = "",
) -> RunResult:
    cfg = config or CouncilConfig()
    kw = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
    # Metric/bookkeeping stages (extraction, clustering, survival, faithfulness,
    # rescue, composite judging) run cold: the groupthink gap must not vary with
    # the run's creative temperature. Creative stages (members, syntheses,
    # composer) keep cfg.temperature.
    judge_kw = {**kw, "temperature": 0.0}

    # Meter every model call for cost visibility.
    meter = Meter()
    client = MeteredClient(client, meter)

    def report(key: str, label: str) -> None:
        if progress:
            progress(key, label)

    # Stage 1 — independent, blind answers.
    report("answers", f"Gathering independent answers from {len(cfg.seats)} members")
    context = f"Briefing documents (context):\n\n{brief_text.strip()}" if brief_text.strip() else ""
    members = llm.generate_members(
        client, prompt, cfg.seats, min_answer_chars=cfg.min_answer_chars,
        context=context, **kw
    )

    # Shared idea ledger built once from those answers (used for BOTH metrics).
    report("ledger", "Extracting and clustering ideas into a ledger")
    warnings: list[str] = []
    cards = []
    for m in members:
        cards.extend(llm.extract_ideas(client, cfg.clerk, m, warnings=warnings, **judge_kw))
    ledger = llm.build_ledger(client, cfg.clerk, cards, warnings=warnings, **judge_kw)

    run = RunResult(prompt=prompt, members=members, ledger=ledger, best_raw=None,
                    warnings=warnings, brief_used=bool(context))

    # Baseline (Karpathy): peer-review ranking + chairman blend.
    best_label = None
    if "baseline" in modes:
        report("baseline", "Baseline: peer review + chairman synthesis")
        synth_b, best_label = llm.baseline_synthesis(client, cfg.chairman, prompt, members, **kw)
        surviving_b = llm.judge_survival(client, cfg.clerk, synth_b, ledger, warnings=warnings, **judge_kw)
        unsupported_b = llm.check_faithfulness(client, cfg.clerk, synth_b, members, warnings=warnings, **judge_kw)
        run.results["baseline"] = ModeResult(
            "baseline", synth_b, compute_survival(ledger, surviving_b), unsupported_b
        )

    # Preserving: red-team rescue marks endangered minority ideas, chair writes from ledger.
    if "preserving" in modes:
        report("preserving", "Preserving: red-team rescue + synthesis from ledger")
        rescue = llm.red_team_rescue(client, cfg.red_team, ledger, warnings=warnings, **judge_kw)
        for c in ledger:
            if c.id in rescue:
                c.must_include = True
        synth_p = llm.preserving_synthesis(client, cfg.chairman, prompt, ledger, **kw)
        surviving_p = llm.judge_survival(client, cfg.clerk, synth_p, ledger, warnings=warnings, **judge_kw)
        unsupported_p = llm.check_faithfulness(client, cfg.clerk, synth_p, members, warnings=warnings, **judge_kw)
        run.results["preserving"] = ModeResult(
            "preserving", synth_p, compute_survival(ledger, surviving_p), unsupported_p
        )

    # Optional hidden-profile pass: emergent cross-model composites (separate from
    # the synthesis, so it stays outside the faithfulness check).
    if cfg.compose:
        report("compose", "Composing cross-model ideas (hidden-profile pass)")
        if cfg.compose_judge == cfg.chairman:
            warnings.append(
                "compose_judge is the same model as the chairman — composite "
                "verdicts are not independent. Set COUNCIL_COMPOSE_JUDGE to a "
                "different model."
            )
        # Feed the RAW per-model cards (pre-clustering), not the merged ledger, so
        # model-specific fragments stay visible for genuine hidden-profile composites.
        proposed = llm.compose_ideas(client, cfg.chairman, prompt, cards, **kw)
        # independent judge (its own model, not the chairman that proposed them)
        run.composites = llm.judge_composites(client, cfg.compose_judge, prompt, proposed, **judge_kw)

    # Optional fact-checking of each synthesis with a web-capable verifier.
    if cfg.verify and cfg.verifier:
        report("verify", "Fact-checking syntheses against sources")
        for res in run.results.values():
            res.verification = llm.verify_claims(client, cfg.verifier, res.synthesis, **judge_kw)

    report("done", "Done")

    # Best raw answer: peer-ranked winner if we have it, else highest-merit contributor.
    run.best_raw = _pick_best_raw(members, ledger, best_label)
    run.cost = meter.summary()
    return run


def _pick_best_raw(
    members: list[MemberAnswer], ledger: list[IdeaCluster], best_label: Optional[str]
) -> Optional[MemberAnswer]:
    live = [m for m in members if m.usable and m.text.strip()]
    if not live:
        return None
    if best_label:
        for m in live:
            if m.label == best_label:
                return m
    # fallback: member contributing the most total idea-merit
    merit_by_label: dict[str, float] = {}
    for c in ledger:
        for label in set(c.source_labels):
            merit_by_label[label] = merit_by_label.get(label, 0.0) + c.merit
    if merit_by_label:
        top = max(merit_by_label, key=merit_by_label.get)
        for m in live:
            if m.label == top:
                return m
    return live[0]
