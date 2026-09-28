"""OpenAI (frontier). Uses OPENAI_API_KEY from the environment by default."""

from .openai_compatible import OpenAICompatibleProvider


class OpenaiProvider(OpenAICompatibleProvider):
    # Bound a hung request: one stuck seat must not stall a whole fan_out (the
    # SDK default is 600s). Override via provider_configs={"openai": {"timeout": ...}}.
    DEFAULT_TIMEOUT = 180.0
