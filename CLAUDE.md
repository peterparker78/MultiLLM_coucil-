# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[all]"                          # or .[openai] / .[anthropic] / .[dev]

# Tests — every test file is also a runnable script with its own __main__:
PYTHONPATH=. python3 tests/test_smoke.py         # offline: routing, factory, normalization
PYTHONPATH=. python3 -m pytest tests/            # full suite
PYTHONPATH=. python3 -m pytest tests/test_council.py::test_name   # single test

# Web UI (council at /, advisory at /advisory):
./start.sh                                        # needs OPENROUTER_API_KEY; see file header for seat defaults
PYTHONPATH=. COUNCIL_DEMO=1 python3 -m web.server # offline demo, no keys (uses council.fake.DemoClient)

# CLIs:
PYTHONPATH=. python3 -m council.cli "Your question"  [--seats ...] [--chairman ...] [--modes baseline,preserving]
PYTHONPATH=. python3 -m advisory.cli "Your question" [--brief a.pdf --brief b.docx] [--verify]
# council.cli also takes repeatable --brief; documents are shown to every seat

# Controlled A/B over a saved council run (needs OPENROUTER_API_KEY; a few cents/repeat):
PYTHONPATH=. python3 bench/compose_ab.py council_runs/<run>.json --repeats 5
```

`PYTHONPATH=.` is required for every invocation — the package is run in-tree, not installed onto the path by the scripts.

Tests, the demo client, and smoke tests all run with **no API key and no network**. Use `COUNCIL_DEMO=1` / `DemoClient` to exercise the full pipeline offline; the demo data is hand-tuned to reproduce the documented groupthink gap (baseline drops minority ideas, preserving keeps them).

## Architecture

Three layers, bottom-up. The lower layers know nothing about the higher ones.

### 1. `aisuite/` — the LLM routing engine

A from-scratch reimplementation of the routing core of [aisuite](https://github.com/andrewyng/aisuite). One OpenAI-style API over frontier and local models. Model strings are always `"<provider>:<model-name>"`.

- `client.py` — `Client.chat.completions.create()`; parses `provider:model`, routes to a provider.
- `provider.py` — `Provider` base + **convention-based `ProviderFactory`**: a provider is discovered by filename. Drop `providers/<key>_provider.py` with class `<Key>Provider` and it works — **no registration anywhere**.
- `types.py` — normalized `ChatCompletionResponse` (the OpenAI shape is the internal contract every provider must return; calling code never sees a provider's native objects). `Message.tool_calls` round-trips through every provider — this is the seam reserved for a future agent/tool-calling layer.
- `providers/` — three adapter styles: **OpenAI-compatible** (`openai_compatible.py`; subclasses are ~10 lines of base URL + defaults — covers openai, openrouter, ollama, lmstudio, moonshot), **native** (`anthropic_provider.py`; full message/tool/response translation — the template for adding Google/Bedrock/Cohere), and **CLI-backed** (`claudecode_provider.py`, `codex_provider.py`, shared `cli_common.py`: shell out to `claude -p` / `codex exec` so a seat runs on the user's claude.ai / ChatGPT subscription login, no API key). CLI-backed seats ignore `temperature`/`max_tokens`, run in an empty temp cwd with tools off so no CLAUDE.md/memory leaks into the prompt, cost ~20 s start-up per call, meter as $0, and share the subscription's rate limits — use them for member/lens seats or the chairman, not the call-heavy clerk (the two-temperature rule can't be enforced there).
- `concurrent.py` — `fan_out()`: runs N models concurrently and **blind to each other**; a failing seat is captured per-model (`FanOutResult.ok`/`.error`), never raised. This independence is the foundation the council pipeline depends on.
- Hosted providers (`openai`, `openrouter`) set `DEFAULT_TIMEOUT = 180s` so one hung seat can't stall a whole `fan_out` (SDK default is 600s). Override per provider with `Client(provider_configs={"openrouter": {"timeout": ...}})`.
- `toolkits/literature.py` — PubMed + Europe PMC search (used by advisory for literature grounding).

### 2. `council/` — the groupthink study tool

Compares two synthesis strategies over **one shared idea ledger**, so the comparison is apples-to-apples. `council/council.py::run_council` is the orchestrator:

1. **Independent blind answers** from each seat (`fan_out`); any seat that returns nothing usable is retried once, so a transient 429/timeout doesn't silently shrink the council.
2. **Shared idea ledger** — extract atomic idea cards per answer (with provenance), cluster + merit-rank each idea on its *own* quality. Built **once**, scored against by both modes.
3. **baseline** mode (Karpathy): peer-review ranking + chairman blend.
4. **preserving** mode: red-team flags endangered minority ideas as `must_include`, chairman writes *from the ledger*.
5. Each synthesis is scored by **idea-survival** (`metrics.py`): minority-kept vs consensus-kept rate. The **gap** is the groupthink metric (lower = less groupthink).
6. Optional **faithfulness check** (chairman-introduced unsourced claims), **fact-check** (web verifier), and the **emergent-composite / hidden-profile pass** (🧩): chairman proposes cross-model composites incorporating minority ideas, then an *independent* judge (`compose_judge`, must differ from chairman) rates strong/weak/reject and culls rejects. Composites live outside the synthesis so they're exempt from the faithfulness check.

`llm.py` holds every model call as one well-scoped function (`generate_members`, `extract_ideas`, `build_ledger`, `red_team_rescue`, `baseline_synthesis`, `preserving_synthesis`, `judge_survival`, etc.). Roles are split across models by cost: **chairman** (synthesis, strong model), **clerk** (mechanical JSON extraction/clustering/judging — call-heavy, the biggest cost lever, runs on a cheap model), **red_team**, **verifier**, **compose_judge**. All configured via `config.py` / `COUNCIL_*` env vars / CLI flags. `cost.py`'s `MeteredClient` wraps the client to meter every call.

### 3. `advisory/` — the advisory-board pipeline

Reuses `council.llm` / `cost` / `types` — **no duplicated plumbing**. Expert *lenses* (editable personas, `lenses.py`) each answer with their role prompt + optional briefing docs → the same idea ledger with provenance → red-team rescue → a board recommendation that **preserves dissent** (an explicit minority-positions section) rather than averaging it away. Briefing ingestion lives in `council/brief.py` (`.txt/.md/.docx/.pdf`; `combine_briefs` joins several documents under one shared character budget, `brief_max_chars`) and is used by both pipelines — `advisory/brief.py` only re-exports it. Optional literature grounding (`toolkits/literature`) and web fact-check.

### `web/` — FastAPI UI over both pipelines

Shared backend; `/` = council, `/advisory` = advisory. A run executes in a **background thread** so the browser polls live stage progress (the `progress(stage_key, label)` hook threaded through the orchestrators). Jobs are in-memory — this is a **single-user local tool**, not a multi-tenant server. Includes a human Adopt/Explore/Reject rating layer, run history (`council_runs/`, `advisory_runs/` JSON), per-seat `:online` web toggle, multi-file briefing upload (`/api/brief`) for both pages, and DOCX/MD exports (Garamond, house palette — see `report.py`). Seat/chairman/lens models are chosen with a provider + model picker (`static/models.js`) fed by `/api/models` (`web/models.py`: OpenRouter rate card, local Ollama tags, Ollama cloud library — each source is fetched independently and degrades to an empty list, never an error).

## Conventions

- **Adding a provider:** filename convention only — `providers/<key>_provider.py` + class `<Key>Provider`. Subclass `OpenAICompatibleProvider` for OpenAI-wire backends; follow `anthropic_provider.py` for native ones.
- Every council/advisory LLM stage takes a `client` exposing `chat.completions.create`, so it can be mocked offline. Keep new stages in that shape. That is also the **bring-your-own-model** seam: `council/clientfactory.py::make_client` returns `aisuite.Client()` unless `COUNCIL_CLIENT_FACTORY="pkg.module:function"` names a user factory; the CLIs, web UI and bench all construct their client through it. For users without code, `providers/byo_provider.py` (`byo:<model>`, `BYO_API_URL`/`BYO_API_KEY`) targets any OpenAI-compatible endpoint.
- **Two-temperature rule:** creative stages (member answers, syntheses, composer) use `cfg.temperature`; every metric/bookkeeping stage (extraction, clustering, survival, faithfulness, rescue, verification, composite judging) gets `judge_kw` (`temperature=0.0`) so the groupthink gap and audit trail don't vary with the run's creative temperature. New stages must pick a side.
- **Degrade, don't die:** a failed bookkeeping stage degrades the run and appends a note to `RunResult.warnings` (serialized, shown in the UI) — it never raises. Parse guards are per-item, not per-response: one malformed cluster must not discard the whole ledger. `llm._complete` raises `ModelRefused` when a model returns no content (safety refusal, `finish_reason=content_filter`, or empty) so the warning names the cause, not a JSON parse error. Every clerk stage takes `warnings=` — pass it from new callers.
- **Clerk model choice:** Opus 5.5's biosafety classifier refuses clerical work on clinical content (a clinical-trial board had all nine clerk calls refused). Keep the clerk/red-team on Opus 5 or another model without that classifier; gpt-4o-mini is too weak to cluster (collapses a 68-card ledger into 4 themes).
- **Judge independence:** the clerk and `compose_judge` must run on a *different model* from the chairman — they score and audit the chairman's output, and same-model self-judging biases the metrics (the orchestrators warn on `compose_judge == chairman`).
- The clerk does **mechanical JSON bookkeeping** — keep its prompts strict and parse defensively (`_complete` already retries once without `temperature` for reasoning models that reject it).
- Markdown output only in syntheses/recommendations — plain Unicode symbols (Δ, ≥, →, ×), **never LaTeX/`$...$`**. Watch for table overflow in DOCX/PDF exports.

## Experiments

`bench/compose_ab.py` freezes everything upstream of the composer (reconstructs member answers from a saved `council_runs/*.json`, regenerates cards + ledger *once* to a sidecar) and varies only the composer's input — ledger-fed (old) vs raw-per-model-card-fed (new) — each followed by the same independent judge, to isolate a composer change that two saved runs can't. Results of these runs are written up in `docs/FINDINGS.md`.
