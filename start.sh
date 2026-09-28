#!/usr/bin/env bash
# Launch the LLM Council web app.
#
# Set your key ONCE in this terminal first (do not commit it anywhere):
#   export OPENROUTER_API_KEY="sk-or-..."
# Optionally also:  export OPENAI_API_KEY="sk-..."   (for openai: seats)
# For the ollama: seat, have `ollama serve` running.
#
# Then just run:  ./start.sh
#
# Any COUNCIL_* var you export yourself overrides the defaults below.

set -e
cd "$(dirname "$0")"

# Seats: a mixed panel — one frontier OpenRouter model (Opus 4.8) plus three
# free Ollama-cloud reasoning models (1 Western + 2 Chinese) to cut OpenRouter
# cost. Ollama-cloud = free within quota but NOT on-device/private. For truly
# private work, swap in local models (no :cloud tag) via the UI.
export COUNCIL_SEATS="${COUNCIL_SEATS:-openrouter:anthropic/claude-opus-4.8,ollama:gemma4:31b-cloud,ollama:kimi-k3:cloud,ollama:deepseek-v4-pro:0813-cloud}"
# Chairman synthesises (user-facing free text) — kept on a strong model.
export COUNCIL_CHAIRMAN="${COUNCIL_CHAIRMAN:-openrouter:openai/gpt-6-sol}"
# Clerical roles (extraction/clustering/judging/red-team) are mechanical JSON
# bookkeeping. This is the call-heavy role, so it's the biggest cost lever —
# but clustering needs a strong model: gpt-4o-mini collapsed a 68-card ledger
# into 4 themes with no shared ideas (Sep 2026), which zeroes the metric.
# Must differ from the chairman (it audits the chairman's output). Not Opus 5.5:
# its biosafety classifier refused every clerk call on a clinical-trial board.
export COUNCIL_CLERK="${COUNCIL_CLERK:-openrouter:anthropic/claude-opus-5}"
export COUNCIL_REDTEAM="${COUNCIL_REDTEAM:-openrouter:anthropic/claude-opus-5}"

# Per-seat web access: tick "web" on an OpenRouter seat in the UI (appends
# ':online'). Optional synthesis fact-checking uses a web-capable verifier:
export COUNCIL_VERIFIER="${COUNCIL_VERIFIER:-openrouter:openai/gpt-6-luna:online}"
# Independent judge for the emergent-composite pass (must differ from the chairman).
# Strong, skeptical, one call per run; near-free alt: openrouter:qwen/qwen3-235b-a22b-thinking-2507
export COUNCIL_COMPOSE_JUDGE="${COUNCIL_COMPOSE_JUDGE:-openrouter:openai/gpt-6-astra}"

if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "WARNING: OPENROUTER_API_KEY is not set."
  echo "         Run:  export OPENROUTER_API_KEY=\"sk-or-...\"   then ./start.sh again."
fi

exec env PYTHONPATH=. python3 -m web.server
