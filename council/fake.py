"""DemoClient — a deterministic, offline stand-in for a real client.

Lets the CLI and web UI run end-to-end with no API keys. Every member shares two
ideas and contributes one unique idea, so the survival metric has both consensus
and minority ideas to track. It deliberately encodes the documented behaviour:
the baseline chairman keeps only consensus ideas, the preserving chairman keeps
everything — so a demo run visibly reproduces the groupthink gap and its fix.
"""

from __future__ import annotations

import json
import re
from typing import Any

from aisuite.types import ChatCompletionResponse, Choice, Message, Usage

# Longer text so demo answers clear the substantive-response gate.
_SHARED = [
    "adopt clear and measurable success criteria for the programme up front",
    "pilot the approach with a small representative cohort before any full rollout",
]


def _resp(content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="demo",
        model="demo",
        choices=[Choice(index=0, message=Message(role="assistant", content=content), finish_reason="stop")],
        usage=Usage(1, 1, 2),
        provider="demo",
    )


def _json_tail(text: str) -> Any:
    return json.loads(text[text.find("[") :])


class DemoClient:
    def __init__(self):
        self.chat = self  # chain client.chat.completions.create onto one object
        self.completions = self

    def create(self, model: str, messages: list, **kw: Any) -> ChatCompletionResponse:
        sys = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]

        if not sys:  # Stage 1: a member answering the bare question
            if "thin" in model:  # degenerate member, for the health-gate test
                return _resp("Yes.")
            ideas = _SHARED + [f"a distinctive angle contributed only by {model}"]
            return _resp(" | ".join(ideas))

        if "faithfulness auditor" in sys:
            return _resp("[]")  # demo: everything supported

        if "independent, skeptical judge" in sys:  # demo: composite judge
            ids = _json_tail(user)
            return _resp(json.dumps([{"i": c.get("i", i), "verdict": "strong", "note": "genuine combination"}
                                     for i, c in enumerate(ids)]))

        if "EMERGENT COMPOSITES" in sys:  # demo: composition pass
            return _resp('[{"summary":"Combine the staffing idea with the scheduling idea into one workflow",'
                         '"combines":[],"source_models":["Member A","Member B"],'
                         '"rationale":"neither model proposed the combined workflow","value":"high"}]')

        if "concise PubMed" in sys:  # demo: literature query extraction
            return _resp("randomized controlled trial primary endpoint overall survival")

        if "fact-checker with web access" in sys:  # demo verification
            return _resp('[{"claim":"demo claim","verdict":"verified","note":"(demo, no real web call)"}]')

        if "chair of an expert advisory board" in sys:  # demo board recommendation
            points = _json_tail(user)
            return _resp("Recommendation. " + " ".join(p.get("point", "") for p in points))

        if "idea-extraction analyst" in sys:
            answer = user.split("\n\n", 1)[1]
            return _resp(json.dumps([s.strip() for s in answer.split("|") if s.strip()]))

        if "clusters ideas" in sys:
            cards = _json_tail(user)
            groups: dict[str, list[str]] = {}
            for c in cards:
                groups.setdefault(c["card"], []).append(c["member"])
            return _resp(json.dumps([
                {"summary": text, "source_labels": sorted(set(ms)), "merit": 7}
                for text, ms in groups.items()
            ]))

        if "red-team advocate" in sys:
            ledger = _json_tail(user)
            return _resp(json.dumps([c["id"] for c in ledger if len(c["raised_by"]) == 1]))

        if "ranking answers" in sys:
            return _resp(json.dumps(list(dict.fromkeys(re.findall(r"Member [A-Z]", user)))))

        if "writing from an idea ledger" in sys:  # preserving chair: keep everything
            return _resp(" ".join(c["idea"] for c in _json_tail(user)))

        if "council chairman" in sys:  # baseline chair: keep only consensus ideas
            bundle = user.split("Member answers:\n\n", 1)[1].split("\n\nPeer ranking")[0]
            counts: dict[str, int] = {}
            order: list[str] = []
            for section in bundle.split("\n\n"):
                _, text = section.split("\n", 1)
                for idea in (s.strip() for s in text.split("|")):
                    if idea not in counts:
                        order.append(idea)
                    counts[idea] = counts.get(idea, 0) + 1
            return _resp(" ".join(i for i in order if counts[i] >= 2))

        if "idea-survival auditor" in sys:
            final = user.split("Final answer:\n\n", 1)[1].split("\n\nCandidate ideas:")[0]
            ideas = _json_tail(user)
            return _resp(json.dumps([c["id"] for c in ideas if c["idea"] in final]))

        # any other role/system prompt is treated as an advisory expert lens
        if sys.strip():
            if "thin" in model:  # simulate a refused/degenerate lens for tests
                return _resp("N/A")
            ideas = _SHARED + [f"a distinctive concern from {model}"]
            return _resp(" | ".join(ideas))
        return _resp("")
