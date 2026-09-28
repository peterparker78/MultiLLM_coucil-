"""ChatGPT models via the OpenAI Codex CLI (`codex exec`): runs on the user's
ChatGPT subscription login — no API key. Model strings: "codex:default" (the
CLI's configured model) or "codex:<model-id>" (e.g. "codex:gpt-6-sol").

Trade-offs: a few seconds of CLI start-up per call; `temperature`/`max_tokens`
are ignored; no system-prompt flag, so the system text is prepended to the
prompt; usage counts against the subscription's limits. Runs ephemeral, in an
empty temp directory, read-only sandbox, so nothing from the current project
leaks into the prompt and the agent cannot touch files.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any

from ..provider import Provider
from ..types import ChatCompletionResponse
from .cli_common import response, run_cli, split_messages


class CodexProvider(Provider):
    def __init__(self, **config: Any):
        self.binary = config.get("binary") or shutil.which("codex") or "codex"
        self.timeout = float(config.get("timeout", 900))

    def chat_completions_create(self, model: str, messages: list[Any], **kwargs: Any) -> ChatCompletionResponse:
        system, prompt = split_messages(messages)
        full = f"{system}\n\n---\n\n{prompt}" if system else prompt
        with tempfile.TemporaryDirectory(prefix="council-codex-") as cwd:
            last = os.path.join(cwd, "last_message.txt")
            cmd = [self.binary, "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only",
                   "--json", "-o", last]
            if model and model != "default":
                cmd += ["-m", model]
            cmd.append("-")  # prompt on stdin
            proc = run_cli(cmd, full, cwd, self.timeout)
            text = open(last, encoding="utf-8").read().strip() if os.path.exists(last) else ""
        usage: dict = {}
        errors: list[str] = []
        for line in (proc.stdout or "").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "turn.completed":
                usage = ev.get("usage") or {}
            elif ev.get("type") == "error" or (ev.get("type") == "turn.failed"):
                errors.append(str(ev.get("message") or ev.get("error") or ev)[:200])
        if not text:
            raise RuntimeError(
                f"codex exited {proc.returncode} with no answer: "
                + ("; ".join(errors) or (proc.stderr or "").strip()[:300])
            )
        return response(
            "codex", model or "default", text,
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
        )
