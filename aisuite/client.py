"""The Client and its nested chat.completions.create() API.

Routing is the whole job here: parse "provider:model", lazily build the provider
(thread-safe, cached), and hand off. Everything model-specific lives in providers/.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from .provider import Provider, ProviderFactory
from .types import ChatCompletionResponse


class Client:
    def __init__(self, provider_configs: Optional[dict[str, dict[str, Any]]] = None):
        """provider_configs maps a provider key to its constructor kwargs, e.g.
        {"openai": {"api_key": "..."}, "ollama": {"base_url": "http://host:11434"}}.
        """
        self.provider_configs: dict[str, dict[str, Any]] = provider_configs or {}
        self.providers: dict[str, Provider] = {}
        self._lock = threading.Lock()
        self._chat: Optional[Chat] = None

    @property
    def chat(self) -> "Chat":
        if self._chat is None:
            self._chat = Chat(self)
        return self._chat

    def _get_provider(self, key: str) -> Provider:
        provider = self.providers.get(key)
        if provider is not None:
            return provider
        with self._lock:  # double-checked locking: avoid the race aisuite flagged as a TODO
            provider = self.providers.get(key)
            if provider is not None:
                return provider
            supported = ProviderFactory.get_supported_providers()
            if key not in supported:
                raise ValueError(
                    f"Unsupported provider '{key}'. Supported: {sorted(supported)}"
                )
            provider = ProviderFactory.create_provider(key, self.provider_configs.get(key, {}))
            self.providers[key] = provider
            return provider


class Chat:
    def __init__(self, client: Client):
        self.completions = Completions(client)


class Completions:
    def __init__(self, client: Client):
        self._client = client

    def create(self, model: str, messages: list[Any], **kwargs: Any) -> ChatCompletionResponse:
        if ":" not in model:
            raise ValueError(
                f"Invalid model '{model}'. Expected 'provider:model-name', "
                f"e.g. 'anthropic:claude-opus-4-8' or 'ollama:llama3.1'."
            )
        provider_key, model_name = model.split(":", 1)
        provider = self._client._get_provider(provider_key)
        return provider.chat_completions_create(model_name, messages, **kwargs)
