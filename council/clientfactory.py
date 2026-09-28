"""Bring your own model client.

Every council/advisory stage only needs an object shaped like the OpenAI
Python client: ``client.chat.completions.create(model=..., messages=[...],
**kw)`` returning something with ``.choices[0].message.content`` (str),
``.choices[0].finish_reason`` and, optionally, ``.usage`` carrying
``prompt_tokens`` / ``completion_tokens`` (for the cost meter).

By default the CLIs and the web UI use ``aisuite.Client()``. To substitute
your own client (a different routing library, an in-house gateway SDK, a
mock), point COUNCIL_CLIENT_FACTORY at a zero-argument callable:

    export COUNCIL_CLIENT_FACTORY="mypackage.llm:make_client"

Seat / chairman / clerk model strings are then whatever *your* client accepts.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

ENV = "COUNCIL_CLIENT_FACTORY"


def make_client() -> Any:
    spec = (os.getenv(ENV) or "").strip()
    if not spec:
        import aisuite as ai
        return ai.Client()
    module_name, sep, func_name = spec.partition(":")
    if not sep or not module_name or not func_name:
        raise ValueError(f"{ENV} must look like 'package.module:function', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"{ENV}: cannot import module {module_name!r}: {e}") from e
    factory = getattr(module, func_name, None)
    if not callable(factory):
        raise ValueError(f"{ENV}: {module_name}.{func_name} is not a callable")
    client = factory()
    if not hasattr(getattr(getattr(client, "chat", None), "completions", None), "create"):
        raise TypeError(f"{ENV}: {spec} returned an object without chat.completions.create")
    return client
