"""Advisory board orchestration.

Flow: each expert lens answers (its own role prompt + optional briefing) → the
council's idea ledger with provenance (which lens raised what) → red-team rescue
of endangered minority positions → a board recommendation written from the
ledger that explicitly preserves dissent → optional faithfulness + fact-check.

Reuses council.llm / cost / types so there is no duplicated plumbing.
"""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from aisuite.toolkits import literature
from council import llm as cl
from council.cost import Meter, MeteredClient
from council.types import IdeaCluster, MemberAnswer

from .lenses import AdvisoryConfig, Lens

_QUERY_SYS = (
    "You turn a clinical research question into a concise PubMed/Europe PMC search "
    "query. Output ONLY the query string — key drug, disease, population, and "
    "endpoint terms joined with AND/OR. No explanation, no quotes."
)


def extract_search_query(client: Any, model: str, question: str, **kw: Any) -> str:
    try:
        q = cl._complete(client, model, _QUERY_SYS, question, **kw).strip()
        return q.splitlines()[0][:300] if q else question[:300]
    except Exception:
        return question[:300]


def _format_evidence(query: str, evidence: list[dict]) -> str:
    if not evidence:
        return ""
    lines = [f'Relevant literature (search: "{query}"). Cite by [n] where you use a '
             "source; do not invent citations beyond this list:"]
    for i, e in enumerate(evidence, 1):
        cite = f"[{i}] {e.get('title','')} — {e.get('authors','')}. {e.get('journal','')} {e.get('year','')}."
        if e.get("id"):
            cite += f" PMID:{e['id']}."
        if e.get("abstract"):
            cite += f"\nAbstract: {e['abstract'][:800]}"
        lines.append(cite)
    return "\n\n".join(lines)

ProgressFn = Callable[[str, str], None]

_BOARD_SYS = (
    "You are the chair of an expert advisory board writing a recommendation for a "
    "decision-maker. You are given a ranked ledger of points, each attributed to the "
    "expert lens(es) that raised it. Write: (1) a clear recommendation, (2) key "
    "considerations integrating the high-merit points, and (3) an explicit "
    "'Dissenting / minority positions' section that PRESERVES single-expert concerns "
    "rather than averaging them away. You MUST include every point marked must_include. "
    "Attribute key positions to their expert lens. Do not invent facts beyond the ledger; "
    "where the ledger flags uncertainty, carry it through. "
    "Format in plain Markdown only — use Markdown tables and plain Unicode symbols "
    "(Δ, ≥, →, ×); do NOT use LaTeX or $...$ math notation."
)


@dataclass
class AdvisoryResult:
    question: str
    brief_used: bool
    experts: list[MemberAnswer]
    ledger: list[IdeaCluster]
    recommendation: str
    dissents: list[dict] = field(default_factory=list)  # minority points + their lens
    unsupported_claims: list = field(default_factory=list)
    verification: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # e.g. a lens that didn't return
    evidence: list[dict] = field(default_factory=list)  # cited literature, if grounded
    evidence_query: str = ""
    composites: list[dict] = field(default_factory=list)  # emergent cross-lens composites
    cost: Optional[dict] = None
    review: Optional[dict] = None  # human ratings, attached later


def _gather_experts(
    client: Any, lenses: list[Lens], question: str, context: str,
    min_chars: int, lens_max_tokens: int, **kw: Any
) -> list[MemberAnswer]:
    user = question if not context else f"{context}\n\n---\n\nQuestion:\n{question}"
    call_kw = {**kw, "max_tokens": lens_max_tokens}  # reasoning lenses need room

    def call(lens: Lens) -> MemberAnswer:
        try:
            resp = client.chat.completions.create(
                model=lens.model,
                messages=[{"role": "system", "content": lens.system},
                          {"role": "user", "content": user}],
                **call_kw,
            )
            return MemberAnswer(model=lens.model, label=lens.name,
                                text=resp.choices[0].message.content or "", ok=True)
        except Exception as e:  # one failed lens must not sink the board
            return MemberAnswer(model=lens.model, label=lens.name, text="", ok=False, error=str(e))

    with ThreadPoolExecutor(max_workers=max(1, len(lenses))) as pool:
        experts = list(pool.map(call, lenses))
    cl._classify_members(experts, min_chars)

    # Retry any lens that returned nothing usable once — a refused/empty critical
    # lens (e.g. biostatistics) is worse than a slow one.
    retry = [i for i, e in enumerate(experts) if not e.usable]
    if retry:
        with ThreadPoolExecutor(max_workers=max(1, len(retry))) as pool:
            for i, e in zip(retry, pool.map(lambda i: call(lenses[i]), retry)):
                experts[i] = e
        cl._classify_members(experts, min_chars)
    return experts


def _board_synthesis(client: Any, model: str, question: str, ledger: list[IdeaCluster], **kw: Any) -> str:
    if not ledger:
        return ""
    payload = sorted(
        ({
            "id": c.id,
            "point": c.summary,
            "merit": c.merit,
            "experts": sorted(set(c.source_labels)),
            "minority": c.is_unique,
            "must_include": c.must_include,
        } for c in ledger),
        key=lambda d: -d["merit"],
    )
    user = f"Question:\n{question}\n\nExpert ledger (ranked by merit):\n\n" + json.dumps(payload, indent=2)
    return cl._complete(client, model, _BOARD_SYS, user, **kw)


def run_advisory(
    client: Any,
    question: str,
    config: Optional[AdvisoryConfig] = None,
    brief_text: str = "",
    progress: Optional[ProgressFn] = None,
) -> AdvisoryResult:
    cfg = config or AdvisoryConfig()
    kw = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
    # Bookkeeping/judging stages run cold so the board's audit trail (ledger,
    # faithfulness, verdicts) doesn't vary with the run's creative temperature.
    judge_kw = {**kw, "temperature": 0.0}
    meter = Meter()
    client = MeteredClient(client, meter)

    def report(key: str, label: str) -> None:
        if progress:
            progress(key, label)

    # Optional literature grounding (deterministic RAG): derive a query, search
    # PubMed + Europe PMC, inject the cited abstracts as shared evidence.
    evidence: list[dict] = []
    evidence_query = ""
    evidence_text = ""
    if cfg.ground:
        report("evidence", "Searching the literature (PubMed + Europe PMC)")
        evidence_query = extract_search_query(client, cfg.clerk, question, **judge_kw)
        evidence = literature.gather_evidence(
            evidence_query, limit=cfg.evidence_limit, email=cfg.ncbi_email or None
        )
        evidence_text = _format_evidence(evidence_query, evidence)

    context_blocks = []
    if brief_text.strip():
        context_blocks.append(f"Briefing document (context):\n\n{brief_text}")
    if evidence_text:
        context_blocks.append(evidence_text)
    context = "\n\n".join(context_blocks)

    report("experts", f"Consulting {len(cfg.lenses)} expert lenses")
    experts = _gather_experts(
        client, cfg.lenses, question, context,
        cfg.min_answer_chars, cfg.lens_max_tokens, **kw
    )
    warnings = [
        f"{e.label} lens ({e.model}) returned no usable answer (status: {e.status}"
        + (f": {str(e.error)[:200]}" if e.error else "")
        + ") — its perspective is missing from this board."
        for e in experts if not e.usable
    ]

    report("ledger", "Building the expert idea ledger")
    cards = []
    for e in experts:
        cards.extend(cl.extract_ideas(client, cfg.clerk, e, warnings=warnings, **judge_kw))
    ledger = cl.build_ledger(client, cfg.clerk, cards, warnings=warnings, **judge_kw)

    report("recommend", "Red-team rescue + board recommendation")
    rescue = cl.red_team_rescue(client, cfg.red_team, ledger, warnings=warnings, **judge_kw)
    for c in ledger:
        if c.id in rescue:
            c.must_include = True
    recommendation = _board_synthesis(client, cfg.chairman, question, ledger, **kw)

    unsupported = cl.check_faithfulness(client, cfg.clerk, recommendation, experts, warnings=warnings, **judge_kw)
    verification: list = []
    if cfg.verify and cfg.verifier:
        report("verify", "Fact-checking the recommendation")
        verification = cl.verify_claims(client, cfg.verifier, recommendation, **judge_kw)

    composites: list[dict] = []
    if cfg.compose:
        report("compose", "Composing cross-lens ideas (hidden-profile pass)")
        if cfg.compose_judge == cfg.chairman:
            warnings.append(
                "compose_judge is the same model as the chairman — composite "
                "verdicts are not independent. Set COUNCIL_COMPOSE_JUDGE to a "
                "different model."
            )
        # Raw per-lens cards (pre-clustering), not the merged ledger, so lens-specific
        # fragments stay visible for genuine hidden-profile composites.
        proposed = cl.compose_ideas(client, cfg.chairman, question, cards, **kw)
        # independent judge (its own model, not the chairman that proposed them)
        composites = cl.judge_composites(client, cfg.compose_judge, question, proposed, **judge_kw)

    dissents = [
        {"point": c.summary, "expert": sorted(set(c.source_labels)), "merit": c.merit}
        for c in ledger if c.is_unique
    ]

    return AdvisoryResult(
        question=question,
        brief_used=bool(brief_text.strip()),
        experts=experts,
        ledger=ledger,
        recommendation=recommendation,
        dissents=dissents,
        unsupported_claims=unsupported,
        verification=verification,
        warnings=warnings,
        evidence=evidence,
        evidence_query=evidence_query,
        composites=composites,
        cost=meter.summary(),
    )


def to_dict(result: AdvisoryResult) -> dict[str, Any]:
    return {
        "kind": "advisory",
        "question": result.question,
        "prompt": result.question,  # alias so shared history/listing works
        "brief_used": result.brief_used,
        "experts": [dataclasses.asdict(e) for e in result.experts],
        "ledger": [{**dataclasses.asdict(c), "is_unique": c.is_unique} for c in result.ledger],
        "recommendation": result.recommendation,
        "dissents": result.dissents,
        "unsupported_claims": result.unsupported_claims,
        "verification": result.verification,
        "warnings": result.warnings,
        "evidence": result.evidence,
        "evidence_query": result.evidence_query,
        "composites": result.composites,
        "cost": result.cost,
        "review": result.review,
    }
