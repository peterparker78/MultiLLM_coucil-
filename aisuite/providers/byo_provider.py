"""Bring your own endpoint: any server that speaks the OpenAI chat-completions
wire protocol (vLLM, LiteLLM, TGI, Groq, Together, Azure OpenAI gateways, a
company proxy...). Model strings: "byo:<model-id as your server names it>".

    export BYO_API_URL="https://my-gateway.example.com/v1"   # base URL incl. /v1
    export BYO_API_KEY="..."                                  # optional if the server needs none
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compatible import OpenAICompatibleProvider


class ByoProvider(OpenAICompatibleProvider):
    DEFAULT_API_KEY = "none"  # some servers require a header but no real key
    DEFAULT_TIMEOUT = 300.0

    def __init__(self, **config: Any):
        base = config.pop("base_url", None) or os.getenv("BYO_API_URL")
        if not base:
            raise ValueError(
                "byo: seats need BYO_API_URL (an OpenAI-compatible base URL such as "
                "https://host/v1); set BYO_API_KEY too if the server requires one."
            )
        config["base_url"] = base.rstrip("/")
        config.setdefault("api_key", os.getenv("BYO_API_KEY") or self.DEFAULT_API_KEY)
        super().__init__(**config)
