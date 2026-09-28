# MultiLLM Council

A groupthink-resistant council of language models, plus an expert advisory board built on the same engine.

Ask several models the same question, blind to each other. Decompose every answer into atomic ideas with provenance. Then synthesise two ways and **measure** which ideas survive: the classic peer-review-and-blend council (Karpathy's design) versus a diversity-preserving variant that rescues endangered minority ideas. The difference between the two survival rates is the groupthink gap, and it is reported on every run.

Bring whatever models you like: OpenRouter, OpenAI, Anthropic, Moonshot, local Ollama or LM Studio, Ollama cloud, your own OpenAI-compatible endpoint, or your own client code. Your existing Claude Code or Codex CLI login also works as a seat, no API key needed.

## What you get

- **Council** (`/`): independent answers, a shared idea ledger, baseline vs preserving synthesis, idea-survival metrics, an optional faithfulness audit of the chairman, optional web fact-check, and an optional pass that proposes cross-model composites judged by an independent model.
- **Advisory board** (`/advisory`): editable expert lenses (regulatory, biostatistics, safety and ethics, operations, and any you add), briefing documents shown to every lens, literature grounding from PubMed and Europe PMC, and a recommendation that keeps dissent visible instead of averaging it away.
- **Web UI** with provider and model pickers, multi-file briefing upload, live stage progress, run history, an Adopt/Explore/Reject rating layer, and Markdown and Word exports.
- **CLIs** for both pipelines, and an **offline demo mode** that runs the whole thing on canned data with no keys.
- A small **routing engine** (`aisuite/`, a from-scratch reimplementation of the routing core of [aisuite](https://github.com/andrewyng/aisuite)) that gives every backend one OpenAI-style call.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone git@github.com:peterparker78/MultiLLM_coucil-.git
cd MultiLLM_coucil-
pip install -e ".[all]"          # SDKs, web UI, .docx/.pdf briefing support
pip install -e ".[dev]"          # pytest, only if you want to run the tests
```

Try it with no keys at all:

```bash
PYTHONPATH=. COUNCIL_DEMO=1 python3 -m web.server
```

Open http://127.0.0.1:8000. The demo data is hand-tuned to show the effect: the baseline synthesis drops minority ideas, the preserving one keeps them.

For a live run, set at least one key and start the app:

```bash
export OPENROUTER_API_KEY="sk-or-..."   # one key, hundreds of models; the simplest way in
./start.sh                               # see the file header for the default seats
```

`PYTHONPATH=.` is required for every command. The package runs in-tree.

## Choosing models

A seat is written `provider:model`. In the web UI each seat, the chairman, and each advisory lens has a provider dropdown and a model box that filters as you type. Groups whose list is exact (your ChatGPT entitlement, what your LM Studio has loaded, what Ollama has pulled) are closed dropdowns, so you cannot pick a model you cannot run.

| Provider prefix | What it reaches | Needs |
|---|---|---|
| `openrouter:` | Hundreds of frontier and open models through one key. `:online` suffix adds web search. | `OPENROUTER_API_KEY` |
| `openai:` | OpenAI directly | `OPENAI_API_KEY` |
| `anthropic:` | Anthropic directly | `ANTHROPIC_API_KEY` |
| `moonshot:` | Moonshot AI (Kimi) directly | `MOONSHOT_API_KEY`, optional `MOONSHOT_API_URL` |
| `ollama:` | A local Ollama daemon, including its cloud-hosted tags such as `kimi-k3:cloud` | `ollama serve`, optional `OLLAMA_API_URL` |
| `lmstudio:` | A local LM Studio server | LM Studio running, `LMSTUDIO_API_KEY` if auth is on |
| `byo:` | Any OpenAI-compatible endpoint you operate. See below. | `BYO_API_URL`, optional `BYO_API_KEY` |
| `claudecode:` | Claude through the Claude Code CLI on your claude.ai subscription login | `claude` installed and logged in |
| `codex:` | ChatGPT models through the Codex CLI on your ChatGPT subscription login | `codex` installed and logged in |

Roles are configured separately from seats, because they have different jobs and costs:

| Variable | Role | Notes |
|---|---|---|
| `COUNCIL_SEATS` | The members, comma-separated | Diversity matters more than strength |
| `COUNCIL_CHAIRMAN` | Writes the syntheses | A strong model |
| `COUNCIL_CLERK` | Extracts, clusters and judges ideas | Call-heavy. Must be a different model from the chairman, since it audits the chairman's output. Needs to be strong enough to cluster; small models collapse the ledger. |
| `COUNCIL_REDTEAM` | Flags endangered minority ideas | |
| `COUNCIL_VERIFIER` | Web fact-check | A web-capable model, e.g. an OpenRouter `:online` model |
| `COUNCIL_COMPOSE_JUDGE` | Judges proposed composites | Must differ from the chairman |

`start.sh` sets sensible defaults for all of these and any variable you export yourself wins.

## Bring your own models

You do not have to use the bundled routing engine. Three routes, from least to most code.

**1. Point at your own endpoint.** Anything that speaks the OpenAI chat-completions protocol works: vLLM, LiteLLM, TGI, Groq, Together, an Azure gateway, a company proxy.

```bash
export BYO_API_URL="https://my-gateway.example.com/v1"
export BYO_API_KEY="..."            # only if your server needs one
PYTHONPATH=. python3 -m council.cli "Your question" --seats byo:my-model-a,byo:my-model-b
```

The UI's "Your endpoint (BYO)" group lists whatever your server reports at `/models`.

**2. Add a provider file.** Providers are discovered by filename, with no registration. Drop `aisuite/providers/<key>_provider.py` containing a class `<Key>Provider` and `<key>:model` works everywhere. For an OpenAI-compatible backend that is about ten lines; copy `moonshot_provider.py`. For a backend with its own API, copy `anthropic_provider.py`, which shows the full message and response translation. For a backend you reach through a command-line tool, copy `codex_provider.py`.

**3. Supply your own client object.** Every stage of both pipelines only ever calls `client.chat.completions.create(model=..., messages=[...], **kw)` and reads `.choices[0].message.content`, `.choices[0].finish_reason` and optionally `.usage`. If you already have a client shaped like that, whether a different routing library, an in-house SDK or a mock, point the app at a factory function and skip the engine entirely:

```bash
export COUNCIL_CLIENT_FACTORY="mypackage.llm:make_client"
```

Seat and role strings are then whatever your client understands.

## How the council works

1. **Independent answers.** Every seat answers the question blind to the others. A failed seat is captured, not raised, and retried once.
2. **Shared idea ledger.** A clerk decomposes each answer into atomic ideas tagged with their author, clusters duplicates, and rates each idea on its own merit rather than on how many models raised it. The ledger is built once and both syntheses are scored against it.
3. **Baseline synthesis.** Members rank each other's answers and the chairman blends them.
4. **Preserving synthesis.** A red team marks endangered minority ideas as must-include and the chairman writes from the ledger.
5. **Survival scoring.** A blind judge checks which ledger ideas appear in each synthesis. Minority-kept versus consensus-kept is reported per mode, and the gap between them is the groupthink metric. Lower is better.
6. **Optional passes.** A faithfulness audit flags claims the chairman introduced that no member supplied. A web verifier fact-checks specific claims. A composite pass has the chairman propose cross-model combinations that no single member offered, which an independent judge then rates and culls.

Every metric stage runs at temperature zero regardless of the creative temperature, so the gap and the audit trail are comparable between runs. A failed bookkeeping stage degrades the run and records a warning that is shown in the UI. It never aborts the run.

## Command line

```bash
PYTHONPATH=. python3 -m council.cli "Your question" \
    [--seats a,b,c] [--chairman m] [--modes baseline,preserving] [--brief doc.pdf --brief notes.md]

PYTHONPATH=. python3 -m advisory.cli "Your question" [--brief protocol.pdf] [--verify]
```

Briefing documents (`.txt`, `.md`, `.docx`, `.pdf`) are shown to every seat under one shared character budget. Run records are written as JSON to `council_runs/` and `advisory_runs/`.

## Tests

```bash
PYTHONPATH=. python3 -m pytest tests/
PYTHONPATH=. python3 tests/test_smoke.py      # each test file also runs as a script
```

Every test runs offline with no keys. Network calls and vendor CLIs are faked.

## Privacy

Briefing documents and questions are sent to every seat. Cloud seats, including Ollama's cloud-hosted tags, send them to third parties. Only local Ollama and LM Studio models, or your own `byo:` endpoint, keep them on your machine. Do not upload confidential or patient-level material to a cloud panel.

Run records are stored locally and are ignored by git.

## Layout

```
aisuite/     routing engine: Client, provider discovery, fan_out, providers/
council/     the council pipeline, metrics, cost meter, briefing ingestion, CLI
advisory/    the advisory-board pipeline, lenses, CLI, report
web/         FastAPI server and the two single-page UIs
bench/       controlled A/B harness for the composite pass
docs/        FINDINGS.md, what the council demonstrated on real runs
tests/       offline test suite
```

`CLAUDE.md` is a guide for working on the code with Claude Code. It doubles as a concise architecture reference for anyone.

## License

MIT. See `LICENSE`.
