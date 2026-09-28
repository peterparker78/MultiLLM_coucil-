"""OpenRouter (meta-provider). One key, hundreds of frontier and open models
behind a single OpenAI-compatible endpoint — ideal for assembling a diverse
council. Uses OPENROUTER_API_KEY.

Model strings keep OpenRouter's own slash form, e.g.
    "openrouter:anthropic/claude-opus-5.5"
    "openrouter:google/gemini-3.1-pro-preview"
The client splits only on the first ':', so the slashes pass through cleanly.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compatible import OpenAICompatibleProvider


class OpenrouterProvider(OpenAICompatibleProvider):
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    # Bound a hung request: one stuck seat must not stall a whole fan_out (the
    # SDK default is 600s). Override via provider_configs={"openrouter": {"timeout": ...}}.
    DEFAULT_TIMEOUT = 180.0

    def __init__(self, **config: Any):
        config.setdefault("api_key", os.getenv("OPENROUTER_API_KEY"))
        super().__init__(**config)
