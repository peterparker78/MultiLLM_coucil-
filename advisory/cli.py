"""Advisory board CLI.

    PYTHONPATH=. python3 -m advisory.cli "Your question" --brief protocol.pdf --brief synopsis.docx

Requires OPENROUTER_API_KEY for openrouter lenses; Ollama lenses need a local
server. --verify fact-checks the recommendation with the web verifier.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from council.clientfactory import make_client

from .advisory import run_advisory, to_dict
from .brief import brief_to_text, combine_briefs
from .lenses import AdvisoryConfig
from . import report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run an expert advisory board.")
    p.add_argument("question")
    p.add_argument("--brief", action="append", default=[], metavar="FILE",
                   help="briefing document (.txt/.md/.docx/.pdf); repeatable")
    p.add_argument("--verify", action="store_true", help="fact-check the recommendation (web)")
    p.add_argument("--out", default="advisory_runs")
    args = p.parse_args(argv)

    cfg = AdvisoryConfig()
    if args.verify:
        cfg.verify = True

    brief_text = ""
    if args.brief:
        docs = [(os.path.basename(b), brief_to_text(b, max_chars=cfg.brief_max_chars)) for b in args.brief]
        brief_text = combine_briefs(docs, max_chars=cfg.brief_max_chars)
        print(f"(briefing: {len(brief_text)} chars from {len(docs)} document(s))", file=sys.stderr)

    client = make_client()  # aisuite by default; COUNCIL_CLIENT_FACTORY to bring your own
    res = run_advisory(client, args.question, cfg, brief_text=brief_text)
    d = to_dict(res)

    print("\n" + "=" * 70 + "\nADVISORY BOARD\n" + "=" * 70)
    print(f"\nQuestion: {res.question}\n")
    print("Panel:")
    for e in res.experts:
        print(f"  {e.label}: {e.model}  [{e.status}]")
    print("\n--- RECOMMENDATION ---\n")
    print(res.recommendation.strip())
    if res.dissents:
        print("\n--- DISSENTING / MINORITY POSITIONS ---")
        for ds in res.dissents:
            print(f"  ({', '.join(ds['expert'])}) {ds['point']}")
    if res.cost:
        print(f"\nCost: ${res.cost.get('total_cost_usd', 0):.4f} · {res.cost.get('total_tokens', 0):,} tokens")

    os.makedirs(args.out, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in res.question)[:40]
    path = os.path.join(args.out, f"{safe or 'advisory'}.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    md_path = os.path.join(args.out, f"{safe or 'advisory'}.md")
    with open(md_path, "w") as f:
        f.write(report.to_markdown(d))
    print(f"\nSaved -> {path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
