"""Provider base class and the convention-based factory.

A provider is any file `providers/<key>_provider.py` containing a class
`<Key>Provider`. Adding a backend = dropping in one file; no registration,
no edits to this module.
"""

from __future__ import annotations

import importlib
import pathlib
from abc import ABC, abstractmethod
from typing import Any

from .types import ChatCompletionResponse


class Provider(ABC):
    """Every backend implements this single method and returns the normalized type."""

    @abstractmethod
    def chat_completions_create(
        self, model: str, messages: list[Any], **kwargs: Any
    ) -> ChatCompletionResponse:
        ...


class ProviderFactory:
    _PROVIDERS_DIR = pathlib.Path(__file__).parent / "providers"
    _SUFFIX = "_provider.py"

    @classmethod
    def get_supported_providers(cls) -> set[str]:
        return {
            f.name[: -len(cls._SUFFIX)]
            for f in cls._PROVIDERS_DIR.glob("*_provider.py")
        }

    @classmethod
    def create_provider(cls, key: str, config: dict[str, Any]) -> Provider:
        module_name = f"{__package__}.providers.{key}_provider"
        class_name = f"{key.capitalize()}Provider"
        module = importlib.import_module(module_name)
        provider_class = getattr(module, class_name)
        return provider_class(**config)
