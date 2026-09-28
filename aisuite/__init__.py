"""aisuite — one interface, many models (frontier + local).

    import aisuite as ai
    client = ai.Client()
    resp = client.chat.completions.create(
        model="anthropic:claude-opus-4-8",   # or "ollama:llama3.1", "openai:gpt-4o"
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(resp.choices[0].message.content)
"""

from .client import Client
from .concurrent import FanOutResult, fan_out
from .types import (
    ChatCompletionResponse,
    Choice,
    Function,
    Message,
    ToolCall,
    Usage,
)

__all__ = [
    "Client",
    "fan_out",
    "FanOutResult",
    "ChatCompletionResponse",
    "Choice",
    "Message",
    "ToolCall",
    "Function",
    "Usage",
]
__version__ = "0.1.0"
