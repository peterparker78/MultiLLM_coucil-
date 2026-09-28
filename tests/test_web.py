"""Web layer: model-catalogue parsing, the /api/models and /api/brief endpoints,
and a brief-aware council run. Offline — network fetchers are monkeypatched and
the pipeline runs on the demo client.
Run: PYTHONPATH=. python3 -m pytest tests/test_web.py
"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("COUNCIL_DEMO", "1")
    from web import server
    monkeypatch.setattr(server, "_RUNS_DIR", tmp_path / "council_runs")
    monkeypatch.setattr(server, "_ADV_RUNS_DIR", tmp_path / "advisory_runs")
    return TestClient(server.app)


def test_parse_openrouter_skips_batch_and_labels_prices():
    from web.models import parse_openrouter
    data = {"data": [
        {"id": "openai/gpt-6-sol", "pricing": {"prompt": "0.000002", "completion": "0.00001"}},
        {"id": "openai/gpt-6-sol:batch", "pricing": {}},
        {"id": "a/free", "pricing": {"prompt": "0", "completion": "0"}},
    ]}
    out = parse_openrouter(data)
    assert [m["id"] for m in out] == ["a/free", "openai/gpt-6-sol"]
    assert "$2.00/$10.00" in out[1]["label"]


def test_parse_ollama_local_and_cloud_pages():
    from web.models import parse_cloud_library, parse_cloud_tags, parse_ollama_tags
    local = parse_ollama_tags({"models": [{"name": "kimi-k3:cloud"}, {"name": "qwen3.8:latest"}]})
    assert [m["id"] for m in local] == ["kimi-k3:cloud", "qwen3.8:latest"]
    assert local[0]["label"].endswith("(cloud)") and local[1]["label"].endswith("(local)")
    html = '<a href="/library/kimi-k3">x</a><a href="/library/deepseek-v4-pro">y</a><a href="/library/kimi-k3">'
    assert parse_cloud_library(html) == ["deepseek-v4-pro", "kimi-k3"]
    tags = "deepseek-v4-pro:cloud deepseek-v4-pro:0813-cloud deepseek-v4-pro:latest"
    assert parse_cloud_tags("deepseek-v4-pro", tags) == ["deepseek-v4-pro:0813-cloud", "deepseek-v4-pro:cloud"]


def test_models_endpoint_degrades_per_source(monkeypatch, tmp_path):
    from web import models

    def down():
        raise OSError("daemon not running")

    monkeypatch.setattr(models, "openrouter_models", lambda: [{"id": "a/b", "label": "a/b"}])
    monkeypatch.setattr(models, "ollama_local_models", down)
    monkeypatch.setattr(models, "ollama_cloud_models", lambda: [{"id": "kimi-k3:cloud", "label": "kimi-k3:cloud"}])

    def no_key():
        raise models.Unavailable("set ANTHROPIC_API_KEY")

    monkeypatch.setattr(models, "anthropic_models", no_key)
    monkeypatch.setattr(models, "openai_models", lambda: [{"id": "gpt-6-sol", "label": "gpt-6-sol"}])
    monkeypatch.setattr(models, "lmstudio_models", lambda: [])
    monkeypatch.setattr(models, "moonshot_models", lambda: [{"id": "kimi-k2.5", "label": "kimi-k2.5"}])
    monkeypatch.setattr(models, "claudecode_models", lambda: [{"id": "opus", "label": "opus"}])
    monkeypatch.setattr(models, "codex_models", lambda: [{"id": "default", "label": "default (gpt-6-astra)"}])
    monkeypatch.setattr(models, "byo_models", lambda: [])
    r = _client(monkeypatch, tmp_path).get("/api/models").json()
    assert r["openrouter"] == [{"id": "a/b", "label": "a/b"}]
    assert r["ollama_cloud"][0]["id"] == "kimi-k3:cloud"
    assert r["ollama"] == [] and "OSError" in r["errors"]["ollama"]
    # No key: the list is derived from OpenRouter's catalogue and the note stays so the UI can say so.
    assert r["notes"]["anthropic"] == "set ANTHROPIC_API_KEY"
    assert r["anthropic"] == []  # the stubbed OpenRouter list has no anthropic/ models to derive from
    assert r["openai"][0]["id"] == "gpt-6-sol" and "lmstudio" not in r["errors"]
    assert r["moonshot"][0]["id"] == "kimi-k2.5"
    assert r["claudecode"][0]["id"] == "opus" and r["codex"][0]["id"] == "default"


def test_parse_direct_provider_lists():
    from web.models import parse_anthropic_models, parse_openai_compatible_models, parse_openai_models
    a = parse_anthropic_models({"data": [{"id": "claude-sonnet-5", "display_name": "Claude Sonnet 5"},
                                         {"id": "claude-opus-5", "display_name": "Claude Opus 5"}]})
    assert [m["id"] for m in a] == ["claude-opus-5", "claude-sonnet-5"] and "Claude Opus 5" in a[0]["label"]
    o = parse_openai_models({"data": [{"id": "gpt-6-sol"}, {"id": "gpt-4o-audio-preview"}, {"id": "o3"},
                                      {"id": "text-embedding-3-small"}, {"id": "whisper-1"}]})
    assert [m["id"] for m in o] == ["gpt-6-sol", "o3"]
    lm = parse_openai_compatible_models({"data": [{"id": "b"}, {"id": "a"}]}, " (LM Studio)")
    assert [m["id"] for m in lm] == ["a", "b"] and lm[0]["label"].endswith("(LM Studio)")


def test_combine_briefs_shares_budget_and_labels_documents():
    from council.brief import combine_briefs
    out = combine_briefs([("short.md", "tiny"), ("long.txt", "x" * 1000)], max_chars=100)
    assert "### Document 1: short.md" in out and "### Document 2: long.txt" in out
    assert "tiny" in out  # the short one keeps its full text
    body = out.split("long.txt\n\n", 1)[1]
    assert "[document truncated]" in body and body.count("x") == 100 - 4  # leftover goes to the long one
    assert combine_briefs([("empty", "   ")]) == ""


def test_multi_brief_upload_then_council_run_uses_them(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    files = [
        ("files", ("protocol.md", b"# Protocol\nPrimary endpoint: PFS", "text/markdown")),
        ("files", ("sites.txt", b"N=200 across 12 sites", "text/plain")),
        ("files", ("scan.xyz", b"?", "application/octet-stream")),
    ]
    up = c.post("/api/brief", files=files).json()
    assert [b["name"] for b in up["briefs"]] == ["protocol.md", "sites.txt"]
    assert len(up["errors"]) == 1 and up["errors"][0].startswith("scan.xyz")

    body = {"prompt": "Design a study.", "briefs": [{"name": b["name"], "text": b["text"]} for b in up["briefs"]]}
    job = c.post("/api/run", json=body).json()["id"]
    for _ in range(400):
        st = c.get(f"/api/run/{job}").json()
        if st["status"] != "running":
            break
        time.sleep(0.05)
    assert st["status"] == "done", st.get("error")
    assert st["result"]["brief_used"] is True

    plain = c.post("/api/run", json={"prompt": "Design a study."}).json()["id"]
    for _ in range(400):
        st = c.get(f"/api/run/{plain}").json()
        if st["status"] != "running":
            break
        time.sleep(0.05)
    assert st["result"]["brief_used"] is False


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))


def test_direct_lists_derive_from_openrouter_without_a_key():
    from web.models import derive_direct_from_openrouter
    orl = [{"id": "anthropic/claude-opus-4.8", "label": ""}, {"id": "anthropic/claude-sonnet-5", "label": ""},
           {"id": "openai/gpt-6-sol", "label": ""}, {"id": "google/gemini-3.8-flash", "label": ""}]
    assert [m["id"] for m in derive_direct_from_openrouter(orl, "anthropic")] == ["claude-opus-4-8", "claude-sonnet-5"]
    assert [m["id"] for m in derive_direct_from_openrouter(orl, "openai")] == ["gpt-6-sol"]


def test_codex_cache_lists_only_visible_models_in_priority_order():
    from web.models import parse_codex_cache
    cache = {"models": [
        {"slug": "gpt-5.5", "display_name": "GPT-5.5", "visibility": "list", "priority": 7},
        {"slug": "gpt-reserve", "display_name": "GPT-Reserve", "visibility": "hide", "priority": 3},
        {"slug": "gpt-6-astra", "display_name": "GPT-6-Astra", "visibility": "list", "priority": 1},
    ]}
    out = parse_codex_cache(cache)
    assert [m["id"] for m in out] == ["gpt-6-astra", "gpt-5.5"] and "GPT-6-Astra" in out[0]["label"]
