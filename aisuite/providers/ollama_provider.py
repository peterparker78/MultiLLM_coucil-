"""Ollama (local). Talks to a local Ollama server's OpenAI-compatible /v1 endpoint.

Host resolution order: explicit base_url -> $OLLAMA_API_URL -> localhost:11434.
The dummy api_key satisfies the OpenAI SDK; Ollama ignores it. The long timeout
accommodates slower local generation.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    DEFAULT_API_KEY = "ollama"
    DEFAULT_TIMEOUT = 300.0

    def __init__(self, **config: Any):
        base = (
            config.pop("base_url", None)
            or os.getenv("OLLAMA_API_URL")
            or "http://localhost:11434"
        )
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        config["base_url"] = base
        super().__init__(**config)
