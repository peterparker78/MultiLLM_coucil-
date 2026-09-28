"""Claude via the Claude Code CLI (`claude -p`): runs on the user's claude.ai
subscription login — no API key. Model strings: "claudecode:opus",
"claudecode:sonnet", "claudecode:haiku" or a full id such as
"claudecode:claude-fable-5-1"; "claudecode:default" uses the CLI's default.

Trade-offs: ~20 s of CLI start-up per call; `temperature`/`max_tokens` are
not controllable and are ignored; usage counts against the subscription's
rate limits (shared with interactive Claude Code). Runs in an empty temp
directory with tools disabled and no session persistence, so nothing from
the current project (CLAUDE.md, memory) leaks into the prompt.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from typing import Any

from ..provider import Provider
from ..types import ChatCompletionResponse
from .cli_common import response, run_cli, split_messages


class ClaudecodeProvider(Provider):
    def __init__(self, **config: Any):
        self.binary = config.get("binary") or shutil.which("claude") or "claude"
        self.timeout = float(config.get("timeout", 900))

    def chat_completions_create(self, model: str, messages: list[Any], **kwargs: Any) -> ChatCompletionResponse:
        system, prompt = split_messages(messages)
        cmd = [self.binary, "-p", "--output-format", "json", "--no-session-persistence", "--tools", ""]
        if model and model != "default":
            cmd += ["--model", model]
        if system:
            cmd += ["--system-prompt", system]
        with tempfile.TemporaryDirectory(prefix="council-claudecode-") as cwd:
            proc = run_cli(cmd, prompt, cwd, self.timeout)
        out = (proc.stdout or "").strip()
        if not out:
            raise RuntimeError(f"claude exited {proc.returncode} with no output: {(proc.stderr or '').strip()[:300]}")
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude returned non-JSON output: {out[:200]}")
        if data.get("is_error"):
            raise RuntimeError(f"claude error ({data.get('subtype')}): {str(data.get('result') or '')[:300]}")
        usage = data.get("usage") or {}
        prompt_tokens = sum(int(usage.get(k, 0) or 0) for k in
                            ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        used_model = next(iter((data.get("modelUsage") or {}).keys()), model)
        return response(
            "claudecode", used_model, data.get("result") or "",
            prompt_tokens=prompt_tokens, completion_tokens=int(usage.get("output_tokens", 0) or 0),
            request_id=data.get("session_id"),
        )
