"""CLI-backed providers (Claude Code / Codex) — the subprocess is faked, so this
is offline and needs neither CLI installed.
Run: PYTHONPATH=. python3 -m pytest tests/test_subscription_providers.py
"""
from __future__ import annotations

import json
import os
import subprocess

from aisuite.providers import cli_common
from aisuite.providers.claudecode_provider import ClaudecodeProvider
from aisuite.providers.codex_provider import CodexProvider

MSGS = [{"role": "system", "content": "Be a pirate."}, {"role": "user", "content": "Say OK."}]


def test_split_messages_single_turn_and_transcript():
    assert cli_common.split_messages(MSGS) == ("Be a pirate.", "Say OK.")
    sys_, prompt = cli_common.split_messages(MSGS + [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "Again"}])
    assert sys_ == "Be a pirate." and prompt == "User: Say OK.\n\nAssistant: OK\n\nUser: Again"


def test_claudecode_provider_parses_json_and_isolates_cwd(monkeypatch):
    seen = {}

    def fake_run(cmd, input, capture_output, text, cwd, timeout):
        seen.update(cmd=cmd, stdin=input, cwd=cwd)
        out = {"type": "result", "subtype": "success", "is_error": False, "result": "Arr, OK.",
               "session_id": "s1", "modelUsage": {"claude-opus-5-5": {}},
               "usage": {"input_tokens": 2, "cache_read_input_tokens": 100, "output_tokens": 4}}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = ClaudecodeProvider(binary="claude").chat_completions_create("opus", MSGS, temperature=0.0, max_tokens=99)
    assert r.choices[0].message.content == "Arr, OK." and r.model == "claude-opus-5-5"
    assert r.usage.prompt_tokens == 102 and r.usage.completion_tokens == 4
    assert seen["stdin"] == "Say OK." and "--system-prompt" in seen["cmd"] and "Be a pirate." in seen["cmd"]
    assert "--model" in seen["cmd"] and "opus" in seen["cmd"] and "--tools" in seen["cmd"]
    assert os.path.abspath(seen["cwd"]) != os.path.abspath(os.getcwd())  # never the project dir


def test_claudecode_provider_raises_on_cli_error(monkeypatch):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"is_error": True, "subtype": "error_during_execution", "result": "Not logged in"}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        ClaudecodeProvider(binary="claude").chat_completions_create("opus", MSGS)
    except RuntimeError as e:
        assert "Not logged in" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_codex_provider_reads_last_message_and_usage(monkeypatch):
    seen = {}

    def fake_run(cmd, input, capture_output, text, cwd, timeout):
        seen.update(cmd=cmd, stdin=input, cwd=cwd)
        with open(cmd[cmd.index("-o") + 1], "w") as f:
            f.write("Arr, OK.\n")
        events = [{"type": "thread.started"}, {"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 7}}]
        return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(json.dumps(e) for e in events), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = CodexProvider(binary="codex").chat_completions_create("gpt-6-sol", MSGS)
    assert r.choices[0].message.content == "Arr, OK." and r.usage.prompt_tokens == 50 and r.usage.completion_tokens == 7
    assert seen["stdin"].startswith("Be a pirate.") and seen["stdin"].endswith("Say OK.")  # system text prepended
    assert "-m" in seen["cmd"] and "gpt-6-sol" in seen["cmd"] and seen["cmd"][-1] == "-"
    assert "read-only" in seen["cmd"] and "--ephemeral" in seen["cmd"]

    r = CodexProvider(binary="codex").chat_completions_create("default", MSGS)
    assert "-m" not in seen["cmd"]  # "default" leaves the CLI's configured model alone
