"""The council's LLM stages. Each function is a single, well-scoped model call.
They take a `client` (our aisuite.Client or any object exposing the same
chat.completions.create) so they can be mocked offline.

Stages map to the design:
  generate_members  -> Stage 1: independent, blind answers (uses fan_out)
  extract_ideas     -> decompose each answer into atomic idea cards (provenance)
  build_ledger      -> cluster cards, merit-rank each idea on its OWN quality
  red_team_rescue   -> flag high-value MINORITY ideas at risk of being dropped
  baseline_synthesis-> Karpathy: peer-review ranking + chairman blend (the baseline)
  preserving_synthesis -> chairman writes FROM the ledger, must keep top unique ideas
  judge_survival    -> which ledger ideas actually appear in a given synthesis
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from aisuite import fan_out

from .types import IdeaCard, IdeaCluster, MemberAnswer

_LABELS = [f"Member {chr(ord('A') + i)}" for i in range(26)]


class ModelRefused(RuntimeError):
    """The model returned no content: a safety-classifier refusal (finish_reason
    content_filter / refusal) or an empty completion. Raised so a degraded stage
    can name the cause in its warning instead of reporting a JSON parse error.
    (Opus 5.5's biosafety classifier refuses clerical work on clinical content.)"""


def _complete(client: Any, model: str, system: str, user: str, **kw: Any) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        resp = client.chat.completions.create(model=model, messages=messages, **kw)
    except Exception as e:
        # Some reasoning models (e.g. GPT-5 series) reject a non-default temperature.
        # Retry once without it before surfacing the error.
        if "temperature" in str(e).lower() and "temperature" in kw:
            kw = {k: v for k, v in kw.items() if k != "temperature"}
            resp = client.chat.completions.create(model=model, messages=messages, **kw)
        else:
            raise
    choice = resp.choices[0]
    content = choice.message.content or ""
    finish = getattr(choice, "finish_reason", None)
    if finish in ("content_filter", "refusal") or not content.strip():
        raise ModelRefused(f"{model} returned no content (finish_reason={finish or 'unknown'})")
    return content


def _parse_json(text: str) -> Any:
    """Defensively parse JSON from a model response (handles ``` fences anywhere,
    prose preamble from reasoning models, and trailing commentary)."""
    t = text.strip()
    if "```" in t:  # pull the first fenced block, wherever it sits
        parts = t.split("```")
        block = parts[1] if len(parts) >= 2 else t
        if block.lstrip().lower().startswith("json"):
            block = block.lstrip()[4:]
        t = block.strip()
    candidates = [i for i in (t.find("["), t.find("{")) if i != -1]
    if candidates:
        start = min(candidates)
        end = max(t.rfind("]"), t.rfind("}"))
        if end > start:
            t = t[start : end + 1]
    return json.loads(t)


def _as_list(obj: Any) -> list:
    """Coerce parsed JSON to a list — reasoning models sometimes wrap the array
    in an object like {"ideas": [...]}."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return v
        return [obj]
    return []


def _idea_text(item: Any) -> str:
    if isinstance(item, dict):
        for k in ("idea", "text", "summary", "claim", "point"):
            if item.get(k):
                return str(item[k])
        return ""
    return str(item)


def _sentence_fallback(text: str) -> list[str]:
    """Last-resort idea cards if the clerk's JSON is unparseable: split the answer
    into lines/sentences so the ledger is never silently empty when text exists."""
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip(" \t-*•#").lstrip("0123456789.) ")
        if not line:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            sent = sent.strip()
            if len(sent) >= 25:  # skip headers / fragments
                chunks.append(sent)
    return chunks[:40]


# --- Stage 1: independent member answers (blind to each other) ----------------

def generate_members(
    client: Any, prompt: str, seats: list[str], min_answer_chars: int = 120,
    context: str = "", **kw: Any
) -> list[MemberAnswer]:
    # Shared context (briefing documents) goes to every seat identically, ahead
    # of the question, so answers stay blind to each other but not to the brief.
    content = f"{context.strip()}\n\n---\n\nQuestion:\n{prompt}" if context.strip() else prompt
    messages = [{"role": "user", "content": content}]
    results = fan_out(client, seats, messages, **kw)
    members: list[MemberAnswer] = []
    for i, r in enumerate(results):
        members.append(
            MemberAnswer(
                model=r.model,
                label=_LABELS[i],
                text=r.text or "",
                ok=r.ok,
                error=None if r.ok else str(r.error),
            )
        )
    _classify_members(members, min_answer_chars)
    # Retry any seat that returned nothing usable once (parity with the advisory
    # pipeline) — a transient 429/timeout must not silently shrink the council.
    retry = [i for i, m in enumerate(members) if not m.usable]
    if retry:
        again = fan_out(client, [seats[i] for i in retry], messages, **kw)
        for i, r in zip(retry, again):
            members[i] = MemberAnswer(
                model=r.model,
                label=members[i].label,
                text=r.text or "",
                ok=r.ok,
                error=None if r.ok else str(r.error),
            )
        _classify_members(members, min_answer_chars)
    return members


def _classify_members(members: list[MemberAnswer], min_chars: int) -> None:
    """Flag thin/refused/failed answers so a degenerate member is not counted as
    a full vote. Thin = far below the cohort, with an absolute floor."""
    import statistics

    lengths = [len(m.text.strip()) for m in members if m.ok and m.text.strip()]
    median = statistics.median(lengths) if lengths else 0
    floor = max(min_chars, 0.35 * median)
    for m in members:
        if not m.ok:
            m.status = "failed"
        elif not m.text.strip():
            m.status = "refused"
        elif len(m.text.strip()) < floor:
            m.status = "thin"
        else:
            m.status = "ok"


# --- Idea extraction + ledger -------------------------------------------------

_EXTRACT_SYS = (
    "You are an idea-extraction analyst. Decompose an answer into its distinct, "
    "atomic ideas — each a single claim, mechanism, recommendation, or insight. "
    "Be exhaustive about substantive ideas; ignore filler. "
    'Return ONLY a JSON array of strings, e.g. ["idea one", "idea two"].'
)


def extract_ideas(
    client: Any, model: str, member: MemberAnswer,
    warnings: Optional[list[str]] = None, **kw: Any
) -> list[IdeaCard]:
    if not member.usable or not member.text.strip():
        return []
    reason = ""
    try:
        raw = _complete(
            client, model, _EXTRACT_SYS,
            f"Answer to decompose:\n\n{member.text}",
            **kw,
        )
        ideas = [_idea_text(t) for t in _as_list(_parse_json(raw))]
        ideas = [t for t in ideas if t.strip()]
        if not ideas:
            reason = "clerk returned no ideas"
    except Exception as e:
        ideas = []
        reason = f"{type(e).__name__}: {e}"
    if not ideas:  # clerk failed/refused -> degrade instead of zeroing the ledger
        ideas = _sentence_fallback(member.text)
        if warnings is not None:
            warnings.append(
                f"Idea extraction for {member.label} failed ({reason}); its answer was "
                "split into sentences instead — ledger quality is degraded for this run."
            )
    return [
        IdeaCard(id=f"{member.label}-{j}", text=t, source_label=member.label)
        for j, t in enumerate(ideas)
    ]


_LEDGER_SYS = (
    "You are a council clerk who clusters ideas. You receive idea cards from "
    "several anonymized members. Group cards that express the SAME underlying idea "
    "into one cluster. Critically: judge each cluster's MERIT on its own quality "
    "and usefulness (0-10), NOT on how many members raised it — a brilliant idea "
    "from one member must be able to outrank a mediocre idea shared by all. "
    'Return ONLY JSON: a list of objects with keys "summary" (string), '
    '"source_labels" (list of the member labels whose cards are in the cluster), '
    'and "merit" (number 0-10).'
)


def build_ledger(
    client: Any, model: str, cards: list[IdeaCard],
    warnings: Optional[list[str]] = None, **kw: Any,
) -> list[IdeaCluster]:
    if not cards:
        return []
    payload = [{"card": c.text, "member": c.source_label} for c in cards]
    parsed: list = []
    reason = "no clusters returned"
    try:
        raw = _complete(
            client, model, _LEDGER_SYS,
            "Idea cards (with their member of origin):\n\n"
            + json.dumps(payload, indent=2),
            **kw,
        )
        parsed = _as_list(_parse_json(raw))
    except Exception as e:
        parsed = []
        reason = f"{type(e).__name__}: {e}"
    out: list[IdeaCluster] = []
    for k, c in enumerate(parsed):
        # Guard per cluster: one malformed entry (e.g. "merit": "high") must not
        # discard the clerk's whole clustering and flatten merit ranking.
        if not isinstance(c, dict):
            continue
        summary = str(c.get("summary", "")).strip()
        if not summary:
            continue
        labels = c.get("source_labels") or []
        if not isinstance(labels, list):
            labels = [labels]
        try:
            merit = float(c.get("merit", 0) or 0)
        except (TypeError, ValueError):
            merit = 0.0  # degrade this cluster only, not the ledger
        out.append(
            IdeaCluster(
                id=f"c{k}",
                summary=summary,
                source_labels=[str(s) for s in labels],
                merit=merit,
            )
        )
    if not out:  # clustering failed -> one cluster per distinct card (degraded but non-empty)
        if warnings is not None:
            warnings.append(
                f"Idea clustering failed ({reason}); using one cluster per card with default "
                "merit — dedup and merit ranking are degraded for this run."
            )
        groups: dict[str, list[str]] = {}
        for c in cards:
            groups.setdefault(c.text, []).append(c.source_label)
        out = [
            IdeaCluster(id=f"c{k}", summary=text, source_labels=sorted(set(ms)), merit=5.0)
            for k, (text, ms) in enumerate(groups.items())
        ]
    return out


# --- Red-team rescue: champion endangered minority ideas ----------------------

_REDTEAM_SYS = (
    "You are a red-team advocate whose job is to RESCUE valuable ideas at risk of "
    "being lost. You are given an idea ledger. Identify the high-value ideas that "
    "were raised by only ONE member (minority ideas) and that a consensus-seeking "
    "summary would likely drop. Argue for the ones worth keeping. "
    'Return ONLY a JSON array of the cluster ids that MUST be preserved in the '
    'final answer, e.g. ["c2","c5"].'
)


def red_team_rescue(
    client: Any, model: str, ledger: list[IdeaCluster],
    warnings: Optional[list[str]] = None, **kw: Any,
) -> set[str]:
    minority = [c for c in ledger if c.is_unique]
    if not minority:
        return set()
    payload = [
        {"id": c.id, "summary": c.summary, "merit": c.merit, "raised_by": sorted(set(c.source_labels))}
        for c in ledger
    ]
    try:  # a failed rescue degrades the run (no must_include flags); it must not kill it
        raw = _complete(
            client, model, _REDTEAM_SYS,
            "Idea ledger:\n\n" + json.dumps(payload, indent=2),
            **kw,
        )
        ids = _parse_json(raw)
        return {str(i) for i in ids}
    except Exception as e:
        if warnings is not None:
            warnings.append(
                f"Red-team rescue failed ({type(e).__name__}: {e}); no minority ideas "
                "were marked must_include for this run."
            )
        return set()


# --- Baseline synthesis (Karpathy: peer review + chairman) --------------------

_REVIEW_SYS = (
    "You are a council member ranking answers. You are shown several anonymized "
    "answers to the same question. Rank them from best to worst by accuracy and "
    "insight. Return ONLY a JSON array of the labels in ranked order, best first, "
    'e.g. ["Member C","Member A","Member B"].'
)

_BASELINE_CHAIR_SYS = (
    "You are the council chairman. Read the anonymized member answers and their "
    "peer rankings, then write the single best comprehensive answer for the user. "
    "Do not mention the council process. Format in plain Markdown only — use Markdown "
    "tables and plain Unicode symbols (Δ, ≥, →, ×); do NOT use LaTeX or $...$ notation."
)


def _bundle(members: list[MemberAnswer]) -> str:
    return "\n\n".join(
        f"{m.label}:\n{m.text}" for m in members if m.usable and m.text.strip()
    )


def baseline_synthesis(
    client: Any, chairman: str, prompt: str, members: list[MemberAnswer], **kw: Any
) -> tuple[str, Optional[str]]:
    live = [m for m in members if m.usable and m.text.strip()]
    if not live:
        return "", None
    bundle = _bundle(members)

    # Stage 2: each member ranks the (anonymized) answers; aggregate by Borda count.
    scores: dict[str, float] = {m.label: 0.0 for m in live}
    for m in live:
        try:
            ranking = _parse_json(
                _complete(client, m.model, _REVIEW_SYS,
                          f"Question:\n{prompt}\n\nAnswers:\n\n{bundle}", **kw)
            )
            for rank, label in enumerate(ranking):
                if label in scores:
                    scores[label] += len(live) - rank
        except Exception:
            continue
    best_label = max(scores, key=scores.get) if scores else None

    # Stage 3: chairman blends.
    ranked = ", ".join(f"{l}={s:g}" for l, s in sorted(scores.items(), key=lambda x: -x[1]))
    synthesis = _complete(
        client, chairman, _BASELINE_CHAIR_SYS,
        f"Question:\n{prompt}\n\nMember answers:\n\n{bundle}\n\n"
        f"Peer ranking (higher = better): {ranked}",
        **kw,
    )
    return synthesis, best_label


# --- Preserving synthesis (write from the ledger, keep minority ideas) --------

_PRESERVING_CHAIR_SYS = (
    "You are the council chairman writing from an idea ledger. You are given a "
    "ranked list of distinct ideas (not the original answers). Write the single "
    "best comprehensive answer for the user that INTEGRATES the high-merit ideas. "
    "You MUST incorporate every idea marked must_include, even if only one member "
    "raised it — these are valuable minority insights. Prioritise by merit, not by "
    "how many members raised an idea. Do not mention the council process or ledger. "
    "Format in plain Markdown only — use Markdown tables and plain Unicode symbols "
    "(Δ, ≥, →, ×); do NOT use LaTeX or $...$ notation."
)


def preserving_synthesis(
    client: Any, chairman: str, prompt: str, ledger: list[IdeaCluster], **kw: Any
) -> str:
    if not ledger:
        return ""
    payload = sorted(
        (
            {
                "id": c.id,
                "idea": c.summary,
                "merit": c.merit,
                "minority_idea": c.is_unique,
                "must_include": c.must_include,
            }
            for c in ledger
        ),
        key=lambda d: -d["merit"],
    )
    return _complete(
        client, chairman, _PRESERVING_CHAIR_SYS,
        f"Question:\n{prompt}\n\nIdea ledger (ranked by merit):\n\n"
        + json.dumps(payload, indent=2),
        **kw,
    )


# --- Survival judging ---------------------------------------------------------

_JUDGE_SYS = (
    "You are an idea-survival auditor. You are given a final answer and a list of "
    "candidate ideas (each with an id). For each idea, decide whether it is "
    "meaningfully present in the final answer. Be strict: a passing mention counts, "
    "a dropped idea does not. "
    'Return ONLY a JSON array of the ids of ideas that ARE present, e.g. ["c0","c3"].'
)


def judge_survival(
    client: Any, model: str, synthesis: str, ledger: list[IdeaCluster],
    warnings: Optional[list[str]] = None, **kw: Any
) -> set[str]:
    if not ledger or not synthesis.strip():
        return set()
    payload = [{"id": c.id, "idea": c.summary} for c in ledger]
    try:
        raw = _complete(
            client, model, _JUDGE_SYS,
            f"Final answer:\n\n{synthesis}\n\nCandidate ideas:\n\n"
            + json.dumps(payload, indent=2),
            **kw,
        )
        return {str(i) for i in _as_list(_parse_json(raw))}
    except Exception as e:
        if warnings is not None:
            warnings.append(
                f"Idea-survival judging failed ({type(e).__name__}: {e}); this synthesis "
                "is scored as keeping 0 ideas — the groupthink metric is not valid for this run."
            )
        return set()


# --- Chairman faithfulness check --------------------------------------------

_FAITHFUL_SYS = (
    "You are a faithfulness auditor. Given a synthesized answer and the source "
    "member answers it was built from, flag ONLY specific, decision-driving claims "
    "the synthesis introduces that NO source supports — quantitative figures "
    "(percentages, rates, durations, sample sizes), specific statistics, named "
    "trials/studies/identifiers, or concrete assertions a reader would act on. "
    "Do NOT flag standard domain terminology or common knowledge (e.g. GCP, "
    "FDA/EMA, FPI, ICH, common acronyms, general regulatory frameworks) — those "
    "are not fabrications. Ignore wording and style; judge substance only. "
    'Return ONLY a JSON array of short strings naming the unsupported specifics, '
    "or [] if there are none."
)


_VERIFY_SYS = (
    "You are a fact-checker with web access. Extract the SPECIFIC, decision-driving "
    "factual claims from the answer below — prioritise quantitative figures (effect "
    "sizes, hazard ratios, rates, sample sizes, percentages, dates), named "
    "trials/identifiers, and regulatory or precedent claims. Do NOT spend verdicts on "
    "generic best-practice statements (e.g. 'OS is a standard secondary endpoint'); "
    "focus on the claims a reader would actually act on. Verify each against current "
    "authoritative sources, at most 8. "
    'Return ONLY a JSON array of objects: '
    '{"claim": string, "verdict": "verified" | "unverified" | "contradicted", '
    '"note": short string with the source or reason}.'
)


def verify_claims(client: Any, model: str, synthesis: str, **kw: Any) -> list[dict]:
    """Fact-check a synthesis with a web-capable (':online') verifier model.
    Returns per-claim verdicts. This is the part that catches confabulations a
    closed-book council cannot."""
    if not synthesis.strip():
        return []
    try:
        raw = _complete(
            client, model, _VERIFY_SYS, f"Answer to fact-check:\n\n{synthesis}", **kw
        )
        out = []
        for c in _as_list(_parse_json(raw)):
            if isinstance(c, dict) and c.get("claim"):
                out.append({
                    "claim": str(c["claim"]),
                    "verdict": str(c.get("verdict", "unverified")).lower(),
                    "note": str(c.get("note", "")),
                })
        return out[:12]
    except Exception:
        return []


_COMPOSE_SYS = (
    "You are looking at the RAW ideas each expert model independently raised on the "
    "same question, grouped by the model that raised them and BEFORE any clustering "
    "or de-duplication. The same idea may therefore appear under several models in "
    "different words. Find EMERGENT COMPOSITES: novel approaches that combine DISTINCT "
    "fragments from DIFFERENT models into something greater than any single model "
    "proposed, and that no single model stated whole (the 'hidden profile' case). "
    "Strongly PREFER combining fragments that appear to be held by only ONE model "
    "(not independently echoed by the others) — those are the genuine hidden-profile "
    "pieces. Do NOT combine two cards that express the SAME idea in different words "
    "(that is mere de-duplication, not composition), and do NOT return composites "
    "built only from ideas that most models already share. If the ideas are largely "
    "consensus and no genuine composite emerges, return an empty array. "
    "Cite the SPECIFIC idea ids you combine. For each, give: 'summary' (1-3 "
    "sentences), 'combines' (the specific idea ids, from DIFFERENT models), "
    "'source_models' (the distinct model labels that actually held those fragments), "
    "'rationale' (why the combination exceeds its parts), and 'value' (high|medium|low). "
    "Return ONLY a JSON array, or [] if no genuine composite emerges."
)


def compose_ideas(client: Any, model: str, question: str, cards: list, **kw: Any) -> list[dict]:
    """Hidden-profile pass over the RAW per-model idea cards (pre-clustering /
    pre-dedup), so genuinely model-specific fragments stay visible to combine.
    Feeding the merged ledger here collapsed model-specific ideas into 'shared'
    clusters and starved the composer; the raw cards preserve full provenance.
    Surfaced separately and flagged (intentionally not traceable to one member),
    so it stays outside the synthesis and its faithfulness check."""
    id_to_model = {c.id: c.source_label for c in cards}
    by_model: dict[str, list] = {}
    for c in cards:
        by_model.setdefault(c.source_label, []).append({"id": c.id, "idea": c.text})
    if len(by_model) < 2:
        return []  # need at least two models for a cross-model composite
    payload = [{"model": label, "ideas": ideas} for label, ideas in by_model.items()]
    try:
        raw = _complete(
            client, model, _COMPOSE_SYS,
            f"Question:\n{question}\n\nRaw ideas, grouped by the model that raised "
            "each (pre-clustering):\n\n"
            + json.dumps(payload, indent=2),
            **kw,
        )
        out = []
        for c in _as_list(_parse_json(raw)):
            if not isinstance(c, dict) or not c.get("summary"):
                continue
            combines = [str(x) for x in (c.get("combines") or [])]
            # each raw card belongs to exactly one model, so resolving the combined
            # ids gives a precise cross-model span.
            spanning = {id_to_model[cid] for cid in combines if cid in id_to_model}
            spanning |= {str(m) for m in (c.get("source_models") or [])}
            if len({s for s in spanning if s}) < 2:
                continue  # not a genuine cross-model composite
            out.append({
                "summary": str(c["summary"]),
                "source_models": sorted(s for s in spanning if s),
                "combines": combines,
                "rationale": str(c.get("rationale", "")),
                "value": str(c.get("value", "")).lower(),
            })
        return out[:6]
    except Exception:
        return []


_COMPOSE_JUDGE_SYS = (
    "You are an independent, skeptical judge of candidate 'emergent composites' "
    "(ideas claimed to combine fragments from different models into something greater "
    "than the parts). You did NOT propose these. For each candidate you MUST: "
    "(1) identify the specific distinct fragments it actually combines and which model "
    "uniquely contributed each; (2) decide whether it is a GENUINE composition that no "
    "single model held whole, versus mere synthesis of ideas the models already shared, "
    "a restatement, or a trivial merge. REJECT it if the fragments are not genuinely "
    "distributed across different models, if it merely rephrases consensus, or if it is "
    "obvious. Be strict: expect to reject several, and do not award 'strong' unless the "
    "cross-model combination is clear and adds real value. "
    'Return ONLY a JSON array aligned to the input order: [{"i": index, "verdict": '
    '"strong" | "weak" | "reject", "note": the distinct model-held fragments that '
    "genuinely combine, or why you rejected it}]."
)


def judge_composites(client: Any, model: str, question: str, composites: list[dict], **kw: Any) -> list[dict]:
    """Independent judge (a different model than the proposer): rate each composite
    and cull rejects. On judge failure, keep composites unjudged rather than drop all."""
    if not composites:
        return []
    payload = [
        {"i": i, "composite": c.get("summary", ""), "combines": c.get("source_models", []),
         "claimed_rationale": c.get("rationale", "")}
        for i, c in enumerate(composites)
    ]
    verdicts: dict[int, tuple[str, str]] = {}
    try:
        raw = _complete(
            client, model, _COMPOSE_JUDGE_SYS,
            f"Question:\n{question}\n\nCandidate composites:\n\n" + json.dumps(payload, indent=2),
            **kw,
        )
        for v in _as_list(_parse_json(raw)):
            if isinstance(v, dict) and "i" in v:
                verdicts[int(v["i"])] = (str(v.get("verdict", "")).lower(), str(v.get("note", "")))
    except Exception:
        verdicts = {}
    out = []
    for i, c in enumerate(composites):
        verdict, note = verdicts.get(i, ("", ""))
        if verdict == "reject":
            continue
        out.append({**c, "verdict": verdict, "judge_note": note})
    return out


def check_faithfulness(
    client: Any, model: str, synthesis: str, members: list[MemberAnswer],
    warnings: Optional[list[str]] = None, **kw: Any
) -> list[str]:
    """Flag claims the chairman introduced that no member supplied. Offline; the
    chairman is the highest-cost, least-checked stage."""
    sources = _bundle(members)
    if not synthesis.strip() or not sources.strip():
        return []
    try:
        raw = _complete(
            client, model, _FAITHFUL_SYS,
            f"Synthesis:\n\n{synthesis}\n\nSource member answers:\n\n{sources}",
            **kw,
        )
        return [str(x).strip() for x in _as_list(_parse_json(raw)) if str(x).strip()][:10]
    except Exception as e:
        if warnings is not None:
            warnings.append(
                f"Faithfulness check failed ({type(e).__name__}: {e}); unsupported claims "
                "in this synthesis were not audited."
            )
        return []
