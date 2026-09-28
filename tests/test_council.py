"""Offline tests for the council. No network or keys.

Uses council.fake.DemoClient, which deliberately encodes the behaviour the
research describes — the BASELINE chairman keeps only consensus (shared) ideas
and drops minority ones, while the PRESERVING chairman keeps everything — so
these tests verify two things at once:
  1. the full orchestration runs end-to-end and wires every stage together, and
  2. the survival metric actually detects the groupthink gap AND its closure.

Run with:  PYTHONPATH=. python3 tests/test_council.py
"""

from council.config import CouncilConfig
from council.council import run_council
from council.fake import DemoClient
from council.metrics import compute_survival
from council.types import IdeaCluster


def test_compute_survival_pure():
    ledger = [
        IdeaCluster("c0", "x", ["Member A", "Member B"]),  # shared
        IdeaCluster("c1", "y", ["Member A"]),              # unique
        IdeaCluster("c2", "z", ["Member C"]),              # unique
    ]
    s = compute_survival(ledger, surviving_ids={"c0", "c1"})
    assert s.shared_total == 1 and s.shared_survived == 1
    assert s.unique_total == 2 and s.unique_survived == 1
    assert s.unique_rate == 0.5 and s.shared_rate == 1.0


def test_council_end_to_end_shows_and_fixes_groupthink():
    cfg = CouncilConfig()
    cfg.seats = ["demo:gpt", "demo:claude", "demo:gemini"]
    cfg.chairman = cfg.clerk = cfg.red_team = "demo:clerk"

    run = run_council(DemoClient(), "test prompt", cfg)

    # ledger: 2 shared + 3 unique ideas
    assert sum(c.is_shared for c in run.ledger) == 2
    assert sum(c.is_unique for c in run.ledger) == 3

    base = run.results["baseline"].survival
    pres = run.results["preserving"].survival

    # baseline keeps all consensus ideas but drops every minority idea (groupthink)
    assert base.shared_rate == 1.0
    assert base.unique_rate == 0.0

    # preserving keeps the minority ideas too — the gap closes
    assert pres.unique_rate == 1.0
    assert pres.shared_rate == 1.0

    # the headline comparison the project exists to demonstrate
    base_gap = base.shared_rate - base.unique_rate
    pres_gap = pres.shared_rate - pres.unique_rate
    assert base_gap > pres_gap

    assert run.best_raw is not None and run.best_raw.ok


def test_cost_meter():
    from types import SimpleNamespace
    from council import cost

    cost._PRICES = {"openai/gpt-x": (1e-6, 2e-6)}  # avoid a network fetch in tests
    m = cost.Meter()
    m.record("openrouter:openai/gpt-x", SimpleNamespace(prompt_tokens=1000, completion_tokens=500))
    m.record("ollama:kimi-k2.5:cloud", SimpleNamespace(prompt_tokens=2000, completion_tokens=2000))
    m.record("demo:unknown", SimpleNamespace(prompt_tokens=10, completion_tokens=10))
    s = m.summary()
    by = {r["model"]: r for r in s["per_model"]}
    assert by["openrouter:openai/gpt-x"]["cost_usd"] == round(1000 * 1e-6 + 500 * 2e-6, 4)  # 0.002
    assert by["ollama:kimi-k2.5:cloud"]["cost_usd"] == 0.0  # local = free
    assert by["demo:unknown"]["cost_usd"] is None and s["cost_partial"] is True
    assert s["total_cost_usd"] == 0.002 and s["total_tokens"] == 5520


def test_council_run_includes_cost():
    cfg = CouncilConfig()
    cfg.seats = ["demo:gpt", "demo:claude", "demo:gemini"]
    cfg.chairman = cfg.clerk = cfg.red_team = "demo:clerk"
    run = run_council(DemoClient(), "test prompt", cfg)
    assert run.cost is not None and "per_model" in run.cost
    assert run.cost["total_tokens"] > 0


def test_degenerate_member_flagged_and_excluded():
    cfg = CouncilConfig()
    cfg.seats = ["demo:gpt", "demo:claude", "demo:thin"]
    cfg.chairman = cfg.clerk = cfg.red_team = "demo:clerk"
    run = run_council(DemoClient(), "test prompt", cfg)
    thin = next(m for m in run.members if m.model == "demo:thin")
    assert thin.status == "thin" and not thin.usable  # flagged, not a full vote
    # its near-empty answer injected no ideas into the ledger
    sources = {lbl for c in run.ledger for lbl in c.source_labels}
    assert thin.label not in sources


def test_faithfulness_check_flags_unsupported_claims():
    from types import SimpleNamespace
    from council import llm
    from council.types import MemberAnswer

    class Auditor:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, model, messages, **kw):
            from aisuite.types import ChatCompletionResponse, Choice, Message, Usage
            return ChatCompletionResponse(
                id="x", model=model,
                choices=[Choice(0, Message(role="assistant", content='["Study 20130216"]'), "stop")],
                usage=Usage(1, 1, 2), provider="x",
            )

    members = [MemberAnswer(model="m", label="A", text="TOWER trial, NCT02013167", status="ok")]
    out = llm.check_faithfulness(Auditor(), "clerk", "TOWER, also Study 20130216", members)
    assert out == ["Study 20130216"]


def test_verify_claims_parses_verdicts():
    from council import llm
    from council.fake import DemoClient
    out = llm.verify_claims(DemoClient(), "demo:verifier", "Some synthesis with claims.")
    assert out and out[0]["verdict"] == "verified" and "claim" in out[0]


def test_online_seat_priced_on_base_model():
    from council import cost
    cost._PRICES = {"openai/gpt-4o": (1e-6, 2e-6)}
    m = cost.Meter()
    from types import SimpleNamespace
    m.record("openrouter:openai/gpt-4o:online", SimpleNamespace(prompt_tokens=1000, completion_tokens=0))
    row = m.summary()["per_model"][0]
    assert row["cost_usd"] == round(1000 * 1e-6, 4)  # ':online' stripped for pricing


def test_compose_requires_cross_model(monkeypatch=None):
    from council import llm
    from council.types import IdeaCard

    # raw per-model cards (pre-clustering): one card from each of two models
    cards = [
        IdeaCard("c0", "idea zero", "Member A"),
        IdeaCard("c1", "idea one", "Member B"),
    ]

    class Composer:
        def __init__(self): self.chat = self; self.completions = self
        def create(self, model, messages, **kw):
            from aisuite.types import ChatCompletionResponse, Choice, Message, Usage
            # one genuine cross-model composite (c0+c1) and one single-model (rejected)
            payload = ('[{"summary":"combine 0 and 1","combines":["c0","c1"],"value":"high"},'
                       '{"summary":"just c0 restated","combines":["c0"],"source_models":["Member A"]}]')
            return ChatCompletionResponse(id="x", model=model,
                choices=[Choice(0, Message(role="assistant", content=payload), "stop")],
                usage=Usage(1, 1, 2), provider="x")

    out = llm.compose_ideas(Composer(), "m", "q", cards)
    assert len(out) == 1  # single-model candidate filtered out
    assert set(out[0]["source_models"]) == {"Member A", "Member B"}


def test_judge_composites_culls_rejects():
    from council import llm

    class Judge:
        def __init__(self): self.chat = self; self.completions = self
        def create(self, model, messages, **kw):
            from aisuite.types import ChatCompletionResponse, Choice, Message, Usage
            payload = '[{"i":0,"verdict":"strong","note":"real"},{"i":1,"verdict":"reject","note":"trivial"}]'
            return ChatCompletionResponse(id="x", model=model,
                choices=[Choice(0, Message(role="assistant", content=payload), "stop")],
                usage=Usage(1, 1, 2), provider="x")

    cands = [{"summary": "good composite", "source_models": ["A", "B"]},
             {"summary": "trivial one", "source_models": ["A", "C"]}]
    out = llm.judge_composites(Judge(), "m", "q", cands)
    assert len(out) == 1 and out[0]["verdict"] == "strong"  # reject culled


def test_council_run_includes_judged_composites():
    cfg = CouncilConfig()
    cfg.seats = ["demo:gpt", "demo:claude", "demo:gemini"]
    cfg.chairman = cfg.clerk = cfg.red_team = "demo:clerk"
    cfg.compose = True
    run = run_council(DemoClient(), "test prompt", cfg)
    assert run.composites and run.composites[0]["summary"]
    assert run.composites[0]["verdict"] == "strong"  # independent judge ran


def _static_client(content: str):
    """A minimal client that answers every call with the same content."""
    from aisuite.types import ChatCompletionResponse, Choice, Message, Usage

    class C:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, model, messages, **kw):
            return ChatCompletionResponse(
                id="x", model=model,
                choices=[Choice(0, Message(role="assistant", content=content), "stop")],
                usage=Usage(1, 1, 2), provider="x",
            )

    return C()


def test_red_team_failure_is_captured_not_fatal():
    from council import llm
    from council.types import IdeaCluster

    class Down:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, model, messages, **kw):
            raise RuntimeError("429 rate limited")

    ledger = [IdeaCluster("c0", "a minority idea", ["Member A"])]
    warnings = []
    out = llm.red_team_rescue(Down(), "m", ledger, warnings=warnings)
    assert out == set()  # degraded, not raised — the run survives
    assert warnings and "rescue" in warnings[0].lower()


def test_build_ledger_keeps_good_clusters_when_one_is_malformed():
    from council import llm
    from council.types import IdeaCard

    payload = (
        '[{"summary":"good one","source_labels":["Member A"],"merit":8},'
        '{"summary":"bad merit","source_labels":["Member B"],"merit":"high"},'
        '{"summary":"good two","source_labels":["Member A","Member B"],"merit":6}]'
    )
    cards = [IdeaCard("A-0", "good one", "Member A"), IdeaCard("B-0", "good two", "Member B")]
    out = llm.build_ledger(_static_client(payload), "m", cards)
    # one malformed merit degrades that cluster only — no uniform-merit fallback
    assert [c.summary for c in out] == ["good one", "bad merit", "good two"]
    assert [c.merit for c in out] == [8.0, 0.0, 6.0]


def test_failed_seat_retried_once():
    from council import llm
    from aisuite.types import ChatCompletionResponse, Choice, Message, Usage

    answer = "A substantive answer with enough length to clear the thin-response gate. " * 3

    class FlakyOnce:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls: dict[str, int] = {}

        def create(self, model, messages, **kw):
            n = self.calls.get(model, 0)
            self.calls[model] = n + 1
            if model == "demo:flaky" and n == 0:
                raise RuntimeError("429 rate limited")
            return ChatCompletionResponse(
                id="x", model=model,
                choices=[Choice(0, Message(role="assistant", content=answer), "stop")],
                usage=Usage(1, 1, 2), provider="x",
            )

    client = FlakyOnce()
    members = llm.generate_members(client, "q", ["demo:ok", "demo:flaky"])
    flaky = next(m for m in members if m.model == "demo:flaky")
    assert client.calls["demo:flaky"] == 2  # retried exactly once
    assert flaky.usable  # the transient failure did not shrink the council


def test_judge_stages_run_cold():
    """Metric stages must run at temperature 0 regardless of the run temperature."""

    class Recorder:
        def __init__(self):
            self.inner = DemoClient()
            self.chat = self
            self.completions = self
            self.temps: dict[str, float] = {}

        def create(self, model, messages, **kw):
            sys = messages[0]["content"] if messages[0]["role"] == "system" else ""
            self.temps.setdefault(sys[:30] or "member", kw.get("temperature"))
            return self.inner.create(model, messages, **kw)

    rec = Recorder()
    cfg = CouncilConfig()
    cfg.seats = ["demo:gpt", "demo:claude", "demo:gemini"]
    cfg.chairman = cfg.clerk = cfg.red_team = "demo:clerk"
    run_council(rec, "test prompt", cfg)

    assert rec.temps["member"] == cfg.temperature  # creative stages keep run temp
    judge_key = next(k for k in rec.temps if k.startswith("You are an idea-survival"))
    clerk_key = next(k for k in rec.temps if k.startswith("You are an idea-extraction"))
    assert rec.temps[judge_key] == 0.0
    assert rec.temps[clerk_key] == 0.0


def test_compose_judge_same_as_chairman_is_warned():
    cfg = CouncilConfig()
    cfg.seats = ["demo:gpt", "demo:claude", "demo:gemini"]
    cfg.chairman = cfg.clerk = cfg.red_team = "demo:clerk"
    cfg.compose = True
    cfg.compose_judge = cfg.chairman  # not independent — must be flagged
    run = run_council(DemoClient(), "test prompt", cfg)
    assert any("compose_judge" in w for w in run.warnings)
    from council.serialize import run_to_dict
    assert run_to_dict(run)["warnings"]  # surfaced in the persisted result too


def test_meter_thread_safe_under_concurrent_records():
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from council import cost

    cost._PRICES = {}  # avoid a network fetch in tests
    m = cost.Meter()

    def hit(_):
        for _ in range(100):
            m.record("openrouter:x/y", SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(hit, range(8)))
    row = m.summary()["per_model"][0]
    assert row["calls"] == 800
    assert row["prompt_tokens"] == 800 and row["completion_tokens"] == 800


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} council tests passed.")


if __name__ == "__main__":
    _run_all()


def test_council_brief_reaches_every_seat_ahead_of_the_question():
    from council import llm

    class Capture:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.seen = []

        def create(self, model, messages, **kw):
            from aisuite.types import ChatCompletionResponse, Choice, Message, Usage
            self.seen.append(messages[-1]["content"])
            return ChatCompletionResponse(
                id="x", model=model,
                choices=[Choice(0, Message(role="assistant", content="A full answer. " * 20), "stop")],
                usage=Usage(1, 1, 2), provider="x",
            )

    cap = Capture()
    llm.generate_members(cap, "What next?", ["p:a", "p:b"], context="Briefing documents (context):\n\nN=200")
    assert len(cap.seen) == 2
    for content in cap.seen:
        assert content.startswith("Briefing documents (context):")
        assert content.rstrip().endswith("Question:\nWhat next?")

    cap = Capture()
    llm.generate_members(cap, "What next?", ["p:a"])
    assert cap.seen == ["What next?"]  # no brief → bare question, unchanged behaviour


def test_refused_clerk_call_is_named_in_warnings():
    """OpenRouter shape for a safety-classifier refusal: empty content with
    finish_reason content_filter. Must degrade with a warning naming the cause."""
    from council import llm
    from council.types import MemberAnswer

    class Refuser:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, model, messages, **kw):
            from aisuite.types import ChatCompletionResponse, Choice, Message, Usage
            return ChatCompletionResponse(
                id="x", model=model,
                choices=[Choice(0, Message(role="assistant", content=""), "content_filter")],
                usage=Usage(0, 0, 0), provider="x",
            )

    m = MemberAnswer(model="m", label="Member A",
                     text="Pay participant stipends within a week of each visit. Offer a pause option instead of withdrawal.", status="ok")
    w = []
    cards = llm.extract_ideas(Refuser(), "clerk", m, warnings=w)
    assert cards, "sentence fallback must still produce cards"
    assert len(w) == 1 and "ModelRefused" in w[0] and "content_filter" in w[0] and "Member A" in w[0]

    ledger = [IdeaCluster(id="c0", summary="x", source_labels=["Member A"], merit=5.0)]
    w = []
    assert llm.judge_survival(Refuser(), "clerk", "some synthesis", ledger, warnings=w) == set()
    assert w and "not valid" in w[0]
    w = []
    assert llm.check_faithfulness(Refuser(), "clerk", "some synthesis", [m], warnings=w) == []
    assert w and "Faithfulness" in w[0]
