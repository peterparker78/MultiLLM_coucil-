"""Controlled A/B for the composer-input change: ledger-fed vs raw-card-fed.

The two saved council runs differ in ledger, member answers, AND composer input,
so they cannot isolate the composer change. This harness freezes everything
upstream and varies ONLY the composer's input:

  1. Reconstruct the member answers from a saved run (cached — no model call).
  2. Regenerate cards + ledger ONCE, freeze both to a sidecar (so every repeat
     and both arms see byte-identical upstream input).
  3. N times: run the OLD composer (fed the clustered ledger) and the NEW
     composer (fed the raw per-model cards) on that frozen input, each followed
     by the SAME independent judge.
  4. Tabulate attribution breadth (the "all-N-models" artifact) and verdicts.

    PYTHONPATH=. python3 bench/compose_ab.py council_runs/787b830cb11f.json --repeats 5

Needs OPENROUTER_API_KEY. Proposer = COUNCIL_CHAIRMAN, judge = COUNCIL_COMPOSE_JUDGE,
extraction clerk = COUNCIL_CLERK. Costs a few cents per repeat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from council.clientfactory import make_client
from council import llm as cl
from council.config import CouncilConfig
from council.cost import Meter, MeteredClient
from council.types import IdeaCard, IdeaCluster, MemberAnswer

# ---- The ORIGINAL (pre-change) composer: fed the clustered ledger. ------------
# Kept here verbatim so both arms run against the same llm-module version; the
# live llm.compose_ideas is the NEW raw-card-fed path.
_OLD_COMPOSE_SYS = (
    "You are looking across distinct ideas raised by DIFFERENT expert models on the "
    "same question. Find EMERGENT COMPOSITES: novel approaches that combine fragments "
    "from different models into something greater than any single model proposed, and "
    "that no single model stated whole (the 'hidden profile' case). "
    "Strongly PREFER composites that incorporate at least one MINORITY idea (raised by "
    "only one model, marked minority:true) combined with ideas from other models, "
    "because those are the genuine hidden-profile cases. AVOID composites built only "
    "from ideas that all models already shared: those are mere consensus synthesis, "
    "not composition, and you should not return them. Do not restate or merge "
    "near-duplicates. If the ideas are largely consensus and no genuine composite "
    "emerges, return an empty array. "
    "Cite the SPECIFIC idea ids you combine. For each, give: 'summary' (1-3 "
    "sentences), 'combines' (the specific idea ids), 'source_models' (the distinct "
    "model labels that actually held those fragments), 'rationale' (why the "
    "combination exceeds its parts), and 'value' (high|medium|low). "
    "Return ONLY a JSON array, or [] if no genuine composite emerges."
)


def compose_from_ledger(client, model, question, ledger, **kw):
    """Original ledger-fed composer (the pre-change behaviour under test)."""
    if len(ledger) < 2:
        return []
    payload = [{"id": c.id, "idea": c.summary, "models": sorted(set(c.source_labels)),
                "minority": c.is_unique} for c in ledger]
    id_to_models = {c.id: set(c.source_labels) for c in ledger}
    try:
        raw = cl._complete(
            client, model, _OLD_COMPOSE_SYS,
            f"Question:\n{question}\n\nIdeas (with the models that raised each):\n\n"
            + json.dumps(payload, indent=2),
            **kw,
        )
        out = []
        for c in cl._as_list(cl._parse_json(raw)):
            if not isinstance(c, dict) or not c.get("summary"):
                continue
            combines = [str(x) for x in (c.get("combines") or [])]
            spanning = set()
            for cid in combines:
                spanning |= id_to_models.get(cid, set())
            spanning |= {str(m) for m in (c.get("source_models") or [])}
            if len({s for s in spanning if s}) < 2:
                continue
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


# ---- Freeze cards + ledger once from the saved member answers. ----------------
def load_or_freeze(run_path: Path, cfg: CouncilConfig, client, kw):
    sidecar = run_path.with_suffix(".frozen.json")
    if sidecar.exists():
        d = json.loads(sidecar.read_text())
        cards = [IdeaCard(**c) for c in d["cards"]]
        ledger = [IdeaCluster(**c) for c in d["ledger"]]
        print(f"(frozen input loaded from {sidecar.name}: "
              f"{len(cards)} cards, {len(ledger)} ledger ideas)", file=sys.stderr)
        return cards, ledger

    run = json.loads(run_path.read_text())
    members = [MemberAnswer(**m) for m in run["members"]]
    print(f"(regenerating frozen input from {len(members)} cached member answers)",
          file=sys.stderr)
    cards: list[IdeaCard] = []
    for m in members:
        cards.extend(cl.extract_ideas(client, cfg.clerk, m, **kw))
    ledger = cl.build_ledger(client, cfg.clerk, cards, **kw)
    sidecar.write_text(json.dumps({
        "cards": [{"id": c.id, "text": c.text, "source_label": c.source_label} for c in cards],
        "ledger": [{"id": c.id, "summary": c.summary, "source_labels": c.source_labels,
                    "merit": c.merit, "must_include": c.must_include} for c in ledger],
    }, indent=2))
    print(f"(froze {len(cards)} cards, {len(ledger)} ledger ideas -> {sidecar.name})",
          file=sys.stderr)
    return cards, ledger


def arm_metrics(proposed: list[dict], judged: list[dict], panel: int) -> dict:
    breadths = [len(set(c.get("source_models", []))) for c in proposed]
    verdicts = [j.get("verdict", "") for j in judged]
    return {
        "proposed": len(proposed),
        "kept": len(judged),
        "reject": len(proposed) - len(judged),
        "strong": verdicts.count("strong"),
        "weak": verdicts.count("weak"),
        "breadths": breadths,
        "all_panel": sum(1 for b in breadths if b >= panel),
    }


def run_repeat(i, cfg, base_client, question, cards, ledger, panel, kw):
    """One repeat: both arms on identical frozen input, own meter (no races)."""
    meter = Meter()
    client = MeteredClient(base_client, meter)

    prop_l = compose_from_ledger(client, cfg.chairman, question, ledger, **kw)
    judg_l = cl.judge_composites(client, cfg.compose_judge, question, prop_l, **kw)

    prop_c = cl.compose_ideas(client, cfg.chairman, question, cards, **kw)
    judg_c = cl.judge_composites(client, cfg.compose_judge, question, prop_c, **kw)

    print(f"  repeat {i+1}: ledger->{len(prop_l)} proposed, card->{len(prop_c)} proposed",
          file=sys.stderr)
    return (arm_metrics(prop_l, judg_l, panel),
            arm_metrics(prop_c, judg_c, panel),
            meter._by_model)


def aggregate(rows: list[dict]) -> dict:
    agg = {k: sum(r[k] for r in rows) for k in ("proposed", "kept", "reject", "strong", "weak", "all_panel")}
    breadths = [b for r in rows for b in r["breadths"]]
    agg["breadth_hist"] = dict(sorted(Counter(breadths).items()))
    agg["mean_breadth"] = round(sum(breadths) / len(breadths), 2) if breadths else 0.0
    return agg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="A/B the composer-input change.")
    p.add_argument("run", help="path to a saved council_runs/*.json")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args(argv)

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        return 1

    cfg = CouncilConfig()
    kw = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
    base_client = make_client()
    run_path = Path(args.run)

    cards, ledger = load_or_freeze(run_path, cfg, base_client, kw)
    if len(ledger) < 2:
        print("ERROR: frozen ledger has <2 ideas; nothing to compose.", file=sys.stderr)
        return 1
    panel = len({c.source_label for c in cards})
    question = json.loads(run_path.read_text())["prompt"]

    print(f"\nProposer (chairman): {cfg.chairman}\nJudge: {cfg.compose_judge}\n"
          f"Panel size: {panel} members · {len(cards)} cards · {len(ledger)} ledger ideas\n"
          f"Repeats: {args.repeats}\n", file=sys.stderr)

    merged = Meter()
    ledger_rows, card_rows = [], []
    with ThreadPoolExecutor(max_workers=min(args.repeats, 4)) as ex:
        futs = [ex.submit(run_repeat, i, cfg, base_client, question, cards, ledger, panel, kw)
                for i in range(args.repeats)]
        for f in futs:
            mL, mC, by_model = f.result()
            ledger_rows.append(mL)
            card_rows.append(mC)
            for model, (calls, pt, ct) in by_model.items():
                e = merged._by_model.setdefault(model, [0, 0, 0])
                e[0] += calls; e[1] += pt; e[2] += ct

    aL, aC = aggregate(ledger_rows), aggregate(card_rows)
    n = args.repeats

    def line(name, a):
        ap_pct = f"{100*a['all_panel']/a['proposed']:.0f}%" if a["proposed"] else "—"
        return (f"{name:<14}{a['proposed']/n:<10.1f}{a['kept']/n:<8.1f}"
                f"{a['strong']:<8}{a['weak']:<7}{a['reject']:<8}"
                f"{a['mean_breadth']:<14}{ap_pct}")

    print("\n" + "=" * 86)
    print(f"COMPOSER A/B  —  identical frozen input, {n} repeats, panel={panel}")
    print("=" * 86)
    print(f"{'arm':<14}{'proposed/r':<10}{'kept/r':<8}{'strong':<8}{'weak':<7}"
          f"{'reject':<8}{'mean_breadth':<14}all-panel")
    print("-" * 86)
    print(line("ledger-fed", aL))
    print(line("card-fed", aC))
    print("-" * 86)
    print(f"breadth histogram (# distinct source_models per proposed composite):")
    print(f"  ledger-fed: {aL['breadth_hist']}")
    print(f"  card-fed:   {aC['breadth_hist']}")
    print("  (all-panel = composites tagged with the FULL panel; lower = more specific "
          "attribution)")

    summary = merged.summary()
    print(f"\nCost: ${summary['total_cost_usd']:.4f} · {summary['total_tokens']:,} tokens"
          + (" (partial)" if summary["cost_partial"] else ""))

    out = run_path.with_suffix(".ab.json")
    out.write_text(json.dumps({
        "run": run_path.name, "repeats": n, "panel": panel,
        "proposer": cfg.chairman, "judge": cfg.compose_judge,
        "ledger_fed": aL, "card_fed": aC, "cost": summary,
    }, indent=2))
    print(f"Wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
