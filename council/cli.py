"""Council CLI.

    PYTHONPATH=. python3 -m council.cli "Your question or challenge here"

Options:
    --seats   comma-separated provider:model list (overrides COUNCIL_SEATS)
    --chairman provider:model for synthesis
    --modes   baseline,preserving (default both)
    --brief   briefing document shown to every seat (.txt/.md/.docx/.pdf); repeatable
    --out     directory for the JSON run record (default ./council_runs)

Requires OPENROUTER_API_KEY (and any other keys for non-OpenRouter seats).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .brief import brief_to_text, combine_briefs
from .clientfactory import make_client
from .config import CouncilConfig
from .council import RunResult, run_council
from .serialize import run_to_dict


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _print_summary(run: RunResult) -> None:
    print("\n" + "=" * 70)
    print("COUNCIL RESULT")
    print("=" * 70)
    print(f"\nPrompt: {run.prompt}\n")
    print(f"Seats ({len(run.members)}):")
    for m in run.members:
        status = "ok" if m.ok else f"FAILED ({m.error})"
        print(f"  {m.label}: {m.model}  [{status}]")

    unique = sum(1 for c in run.ledger if c.is_unique)
    shared = len(run.ledger) - unique
    print(f"\nIdea ledger: {len(run.ledger)} distinct ideas "
          f"({unique} minority / {shared} shared)")

    print("\n--- Idea-survival (the groupthink metric) ---")
    print(f"{'mode':<12}{'minority kept':<16}{'consensus kept':<16}gap")
    for mode, res in run.results.items():
        s = res.survival
        gap = s.shared_rate - s.unique_rate
        print(f"{mode:<12}"
              f"{_fmt_pct(s.unique_rate)+f' ({s.unique_survived}/{s.unique_total})':<16}"
              f"{_fmt_pct(s.shared_rate)+f' ({s.shared_survived}/{s.shared_total})':<16}"
              f"{_fmt_pct(gap)}")
    print("  (lower gap = less groupthink: minority ideas kept nearer to consensus ideas)")

    if run.cost:
        c = run.cost
        partial = " (partial — some models unpriced)" if c.get("cost_partial") else ""
        print(f"\n--- Cost: ${c.get('total_cost_usd', 0):.4f}{partial} "
              f"· {c.get('total_tokens', 0):,} tokens ---")
        print(f"{'model':<42}{'calls':<7}{'tokens':<12}cost")
        for r in c.get("per_model", []):
            cost = f"${r['cost_usd']:.4f}" if r.get("cost_usd") is not None else "n/a"
            toks = r["prompt_tokens"] + r["completion_tokens"]
            print(f"{r['model']:<42}{r['calls']:<7}{toks:<12,}{cost}")

    for mode, res in run.results.items():
        print("\n" + "-" * 70)
        print(f"SYNTHESIS [{mode}]")
        print("-" * 70)
        print(res.synthesis.strip())

    if run.best_raw:
        print("\n" + "-" * 70)
        print(f"BEST RAW ANSWER ({run.best_raw.label} = {run.best_raw.model})")
        print("-" * 70)
        print(run.best_raw.text.strip())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a groupthink-resistant LLM council.")
    p.add_argument("prompt", help="the question or challenge for the council")
    p.add_argument("--seats", help="comma-separated provider:model seats")
    p.add_argument("--chairman", help="provider:model for synthesis")
    p.add_argument("--modes", default="baseline,preserving")
    p.add_argument("--brief", action="append", default=[], metavar="FILE",
                   help="briefing document (.txt/.md/.docx/.pdf) shown to every seat; repeatable")
    p.add_argument("--out", default="council_runs")
    args = p.parse_args(argv)

    cfg = CouncilConfig()
    if args.seats:
        cfg.seats = [s.strip() for s in args.seats.split(",") if s.strip()]
    if args.chairman:
        cfg.chairman = args.chairman
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    if "openrouter" in "".join(cfg.seats + [cfg.chairman]) and not os.getenv("OPENROUTER_API_KEY"):
        print("error: OPENROUTER_API_KEY is not set (needed for openrouter seats).",
              file=sys.stderr)
        return 2

    brief_text = ""
    if args.brief:
        docs = [(os.path.basename(b), brief_to_text(b, max_chars=cfg.brief_max_chars)) for b in args.brief]
        brief_text = combine_briefs(docs, max_chars=cfg.brief_max_chars)
        print(f"(briefing: {len(brief_text)} chars from {len(docs)} document(s))", file=sys.stderr)

    client = make_client()  # aisuite by default; COUNCIL_CLIENT_FACTORY to bring your own
    run = run_council(client, args.prompt, cfg, modes=modes, brief_text=brief_text)

    _print_summary(run)

    os.makedirs(args.out, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in args.prompt)[:40]
    path = os.path.join(args.out, f"{safe or 'run'}.json")
    with open(path, "w") as f:
        json.dump(run_to_dict(run), f, indent=2)
    print(f"\nSaved run record -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
