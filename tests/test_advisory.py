"""Offline tests for the advisory pipeline (DemoClient routes by system prompt).

Run with:  PYTHONPATH=. python3 tests/test_advisory.py
"""

import os
import tempfile

from advisory.advisory import run_advisory, to_dict
from advisory.brief import brief_to_text
from advisory.lenses import AdvisoryConfig
from council.fake import DemoClient


def test_advisory_end_to_end():
    cfg = AdvisoryConfig()
    cfg.lenses = cfg.lenses[:3]  # DemoClient routes by role prompt, not the model
    res = run_advisory(DemoClient(), "Should we proceed with trial design X?", cfg)

    assert all(e.status == "ok" for e in res.experts)  # demo answers are substantive
    assert res.ledger and res.recommendation.strip()
    # each lens contributed a distinct concern -> minority positions preserved
    assert len(res.dissents) >= 1
    assert res.cost and "per_model" in res.cost

    d = to_dict(res)
    assert d["kind"] == "advisory" and d["prompt"] == res.question
    assert d["experts"] and d["recommendation"]


def test_advisory_uses_briefing():
    cfg = AdvisoryConfig()
    cfg.lenses = cfg.lenses[:2]
    res = run_advisory(DemoClient(), "Assess feasibility.", cfg, brief_text="Protocol summary: N=200, 12 sites.")
    assert res.brief_used is True


def test_brief_ingestion_text_formats():
    with tempfile.TemporaryDirectory() as d:
        txt = os.path.join(d, "b.txt")
        with open(txt, "w") as f:
            f.write("Endpoint is overall survival. Power 90%.")
        assert "overall survival" in brief_to_text(txt)

        md = os.path.join(d, "b.md")
        with open(md, "w") as f:
            f.write("# Brief\n\nPrimary endpoint: PFS.")
        assert "PFS" in brief_to_text(md)

        # truncation
        big = os.path.join(d, "big.txt")
        with open(big, "w") as f:
            f.write("x" * 50000)
        assert len(brief_to_text(big, max_chars=1000)) < 1100


def test_refused_lens_is_retried_and_warned():
    from advisory.lenses import Lens
    cfg = AdvisoryConfig()
    cfg.lenses = [
        Lens("Regulatory", "demo:reg", "You are a regulatory advisor on the board."),
        Lens("Biostatistics", "demo:thin", "You are a biostatistician on the board."),  # always thin
    ]
    res = run_advisory(DemoClient(), "What HR should we power for?", cfg)
    biostats = next(e for e in res.experts if e.model == "demo:thin")
    assert biostats.status != "ok"  # retried, still degenerate
    assert any("Biostatistics" in w for w in res.warnings)  # surfaced, not silent
    assert to_dict(res)["warnings"]


def test_advisory_composites():
    cfg = AdvisoryConfig()
    cfg.lenses = cfg.lenses[:3]
    cfg.compose = True
    res = run_advisory(DemoClient(), "How to design trial X?", cfg)
    assert res.composites and to_dict(res)["composites"][0]["summary"]


def test_advisory_compose_judge_same_as_chairman_is_warned():
    cfg = AdvisoryConfig()
    cfg.lenses = cfg.lenses[:2]
    cfg.compose = True
    cfg.compose_judge = cfg.chairman  # not independent — must be flagged
    res = run_advisory(DemoClient(), "How to design trial X?", cfg)
    assert any("compose_judge" in w for w in res.warnings)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} advisory tests passed.")


if __name__ == "__main__":
    _run_all()
