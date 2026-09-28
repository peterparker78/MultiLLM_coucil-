"""Moonshot AI (Kimi) — direct access, OpenAI-compatible wire protocol.

Uses MOONSHOT_API_KEY (from platform.moonshot.ai; a kimi.com consumer
subscription is not an API key). The base URL defaults to the international
endpoint; set MOONSHOT_API_URL=https://api.moonshot.cn/v1 for the China one.
Model strings: "moonshot:<model-id>" with the id as Moonshot lists it.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compatible import OpenAICompatibleProvider


class MoonshotProvider(OpenAICompatibleProvider):
    DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
    DEFAULT_TIMEOUT = 180.0

    def __init__(self, **config: Any):
        config.setdefault("api_key", os.getenv("MOONSHOT_API_KEY"))
        config.setdefault("base_url", os.getenv("MOONSHOT_API_URL") or self.DEFAULT_BASE_URL)
        super().__init__(**config)
