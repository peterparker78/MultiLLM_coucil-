"""LM Studio (local). Same OpenAI-compatible shortcut as Ollama.

Host resolution: explicit base_url -> $LMSTUDIO_API_URL -> localhost:1234.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compatible import OpenAICompatibleProvider


class LmstudioProvider(OpenAICompatibleProvider):
    DEFAULT_API_KEY = "lm-studio"
    DEFAULT_TIMEOUT = 300.0

    def __init__(self, **config: Any):
        base = (
            config.pop("base_url", None)
            or os.getenv("LMSTUDIO_API_URL")
            or "http://localhost:1234"
        )
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        config["base_url"] = base
        # LM Studio can require a real token (Settings > Developer > Auth);
        # otherwise any placeholder is accepted.
        config.setdefault("api_key", os.getenv("LMSTUDIO_API_KEY") or self.DEFAULT_API_KEY)
        super().__init__(**config)
