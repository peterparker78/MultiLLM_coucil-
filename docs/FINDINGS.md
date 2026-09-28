# What the council demonstrated

*Run: "unconventional, rarely-tried approaches for patient retention in decentralised clinical trials" — 18 Jun 2026*

## Setup
A diverse, decorrelated panel: **Claude Opus 4.8** (frontier, Western) + three
Ollama-cloud reasoning models — **Gemma 4** (Western) and **Kimi K2.5** and
**DeepSeek V4** (Chinese). The same answers were synthesised two ways and scored
against one shared idea ledger:
(Seat note, 28 Sep 2026: Kimi K2.5 was withdrawn from Ollama cloud; the default
seats now run Kimi K3 and DeepSeek V4 Pro 0813, so later runs are not directly
comparable with the numbers below.)
- **Baseline** = Karpathy's 3-stage council (independent answers → peer-review
  ranking → chairman blend).
- **Preserving** (ours) = independent answers → idea ledger with provenance,
  each idea merit-ranked on its *own* quality (not how many models raised it) →
  red-team rescue of endangered minority ideas → blind synthesis from the ledger.

## Result
| Mode | Minority ideas kept | Consensus ideas kept | Gap |
|---|---|---|---|
| Baseline | 40% (2/5) | 95% (20/21) | **55%** |
| Preserving | 100% (5/5) | 100% (21/21) | **0%** |

26 distinct ideas surfaced; 5 were minority (single-model) ideas — **5× more
than our first run on a less-diverse, all-OpenAI-family panel**, confirming the
diverse panel does real work.

## The substantive finding
The groupthink wasn't abstract. Comparing the two syntheses, the baseline blend
**deleted the boldest, most unconventional ideas** — exactly the category the
prompt asked for:
- blockchain / smart-contract incentive transparency
- bio-avatars / generative contribution art
- aggressive technical-friction removal (SSO, magic links, offline sync)
- curiosity-gap information asymmetry

These came disproportionately from the **Chinese reasoning models (esp. Kimi)**.
A consensus blend — or a single model's persona sub-agents — would likely never
have surfaced them. The baseline's "strongest combination" was pure consensus
(navigator, monitoring, micro-tasks, dashboards). **On a prompt demanding
novelty, the naive council regressed to the conventional.**

## Why it matters / how we use it
- **Diversity is structural, not cosmetic.** Genuinely different model lineages
  (Western + Chinese) produce decorrelated ideas; same-model "expert personas"
  share one set of blind spots.
- **Preservation surfaces dissent; it does not endorse it.** Several rescued
  ideas (blockchain payments, gambling-style rewards, deposits) carry real
  coercion / undue-influence concerns. The pipeline keeps them **flagged with
  guardrails** for a human (the CMO) to adjudicate — the right division of labour.

## Caveat
This is one run, and the underlying groupthink evidence (Krishnan, Strange Loop
Canon) is a single informal experiment (n≈16 prompts), not peer-reviewed. The
effect is directionally credible and matches known human group dynamics, but it
is a decision-support signal, not proof.

---

# Composer input: raw cards vs merged ledger

*Controlled A/B — 20 Jun 2026 (`bench/compose_ab.py`)*

The emergent-composite (hidden-profile) pass originally fed the composer the
**clustered ledger**. But the clerk over-merges model-specific ideas into "shared"
clusters, so the composer drew from a consensus pool and kept tagging composites
"all N models." The fix: feed it the **raw per-model idea cards** (pre-clustering,
full provenance) instead. To isolate that change from run-to-run variance, we froze
the upstream once (cards + ledger from one saved run) and varied **only** the
composer's input — 5 repeats, same independent judge, panel = 4, gpt-5.5 proposer
/ gpt-5.1 judge.

| Composer input | strong / repeat | rejects | mean breadth | **tagged "all 4 models"** |
|---|---|---|---|---|
| Merged ledger (original) | 0.8 | 3 | 3.82 | **82%** |
| Raw cards (ours) | 2.0 | 1 | 2.89 | **4%** |

Breadth distribution (distinct models per composite): ledger-fed `{3:4, 4:18}` —
18 of 22 defaulted to the full panel; card-fed `{2:4, 3:23, 4:1}` — only 1 of 28.

## What it shows
- **The "all-N-models" artifact is eliminated** (82% → 4%). Because nothing but the
  composer's input differed, this is causal, not run variance — and mechanical, so
  it held across all 5 repeats. Composites now sit at a genuine 2–3 model breadth.
- **A bonus quality lift** — strong composites 0.8 → 2.0/repeat, rejects 3 → 1. This
  is driven mostly by the **larger, finer idea pool** (103 raw cards vs 10 merged
  ledger ideas — the merged ledger was "starving" the composer), a *separate* lever
  from the provenance fix. Both effects flow from feeding raw cards; both are
  favourable.

## Cost
The composer prompt grows ~10× (103 cards vs 10 ideas): ≈$0.165/repeat here, and it
**scales with card count** — wider panels or more verbose models will cost more on
this one call.

## Caveat
n=5 on a single prompt / single frozen ledger. The attribution result is
mechanism-level and generalises; the *magnitude* of the quality lift is specific to
this prompt's idea distribution and should not be quoted as a universal multiplier.
