"""Smoke tests that exercise routing, factory discovery, and response
normalization WITHOUT any API key or network call. Run with:

    PYTHONPATH=. python3 tests/test_smoke.py
or  PYTHONPATH=. python3 -m pytest tests/
"""

import json
from types import SimpleNamespace

import aisuite as ai
from aisuite.provider import Provider, ProviderFactory
from aisuite.providers.anthropic_provider import AnthropicMessageConverter
from aisuite.providers.openai_compatible import normalize_openai_response
from aisuite.types import ChatCompletionResponse, Choice, Message, Usage


def test_factory_discovers_providers():
    supported = ProviderFactory.get_supported_providers()
    assert {"openai", "ollama", "lmstudio", "anthropic", "openrouter", "moonshot", "claudecode", "codex", "byo"} <= supported


def test_byo_provider_points_at_your_endpoint():
    import os
    saved = {k: os.environ.pop(k, None) for k in ("BYO_API_URL", "BYO_API_KEY")}
    try:
        try:
            ProviderFactory.create_provider("byo", {})
            raise AssertionError("byo without BYO_API_URL must fail loudly")
        except ValueError as e:
            assert "BYO_API_URL" in str(e)
        os.environ["BYO_API_URL"] = "https://gateway.example.com/v1/"
        p = ProviderFactory.create_provider("byo", {})
        assert type(p).__name__ == "ByoProvider"
        assert str(p.client.base_url).rstrip("/") == "https://gateway.example.com/v1"
        assert p.client.api_key == "none"
        os.environ["BYO_API_KEY"] = "secret"
        assert ProviderFactory.create_provider("byo", {}).client.api_key == "secret"
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_client_factory_hook_loads_your_client():
    import os, sys, types
    from council.clientfactory import ENV, make_client

    class Fake:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, model, messages, **kw):
            raise NotImplementedError

    mod = types.ModuleType("my_llm_stack")
    mod.make_client = lambda: Fake()
    mod.not_a_client = lambda: object()
    sys.modules["my_llm_stack"] = mod
    saved = os.environ.pop(ENV, None)
    try:
        os.environ[ENV] = "my_llm_stack:make_client"
        assert isinstance(make_client(), Fake)
        os.environ[ENV] = "my_llm_stack:not_a_client"
        try:
            make_client()
            raise AssertionError("an object without chat.completions.create must be rejected")
        except TypeError:
            pass
        os.environ[ENV] = "nonsense"
        try:
            make_client()
            raise AssertionError("malformed spec must be rejected")
        except ValueError:
            pass
        os.environ.pop(ENV)
        assert type(make_client()).__name__ == "Client"  # default: aisuite
    finally:
        sys.modules.pop("my_llm_stack", None)
        if saved is not None:
            os.environ[ENV] = saved



def test_lmstudio_provider_uses_env_token_when_set():
    import os
    saved = os.environ.pop("LMSTUDIO_API_KEY", None)
    try:
        assert ProviderFactory.create_provider("lmstudio", {}).client.api_key == "lm-studio"
        os.environ["LMSTUDIO_API_KEY"] = "real-token"
        assert ProviderFactory.create_provider("lmstudio", {}).client.api_key == "real-token"
    finally:
        os.environ.pop("LMSTUDIO_API_KEY", None)
        if saved is not None:
            os.environ["LMSTUDIO_API_KEY"] = saved


def test_moonshot_provider_is_openai_compatible_with_env_base_url():
    import os
    saved = os.environ.pop("MOONSHOT_API_URL", None)
    try:
        p = ProviderFactory.create_provider("moonshot", {"api_key": "k"})
        assert type(p).__name__ == "MoonshotProvider"
        assert str(p.client.base_url).rstrip("/") == "https://api.moonshot.ai/v1"
        os.environ["MOONSHOT_API_URL"] = "https://api.moonshot.cn/v1"  # China endpoint override
        p = ProviderFactory.create_provider("moonshot", {"api_key": "k"})
        assert str(p.client.base_url).rstrip("/") == "https://api.moonshot.cn/v1"
    finally:
        os.environ.pop("MOONSHOT_API_URL", None)
        if saved is not None:
            os.environ["MOONSHOT_API_URL"] = saved


def test_fan_out_runs_in_parallel_and_captures_errors():
    from aisuite.concurrent import fan_out

    class EchoProvider(Provider):
        def chat_completions_create(self, model, messages, **kwargs):
            if model == "boom":
                raise RuntimeError("seat failed")
            return ChatCompletionResponse(
                id="x",
                model=model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=f"answer from {model}"),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(1, 1, 2),
                provider="echo",
            )

    client = ai.Client()
    client.providers["openai"] = EchoProvider()  # inject; bypass real SDK init
    results = fan_out(
        client,
        ["openai:alpha", "openai:boom", "openai:beta"],
        [{"role": "user", "content": "hi"}],
    )
    # order preserved, matching the requested model list
    assert [r.model for r in results] == ["openai:alpha", "openai:boom", "openai:beta"]
    assert results[0].ok and results[0].text == "answer from alpha"
    # one failing seat is captured, not raised — council survives it
    assert not results[1].ok and isinstance(results[1].error, RuntimeError)
    assert results[2].text == "answer from beta"


def test_model_string_must_be_namespaced():
    client = ai.Client()
    raised = False
    try:
        client.chat.completions.create(model="gpt-4o", messages=[])
    except ValueError:
        raised = True
    assert raised, "expected ValueError for a model string without 'provider:'"


def test_routing_returns_normalized_response():
    class FakeProvider(Provider):
        def chat_completions_create(self, model, messages, **kwargs):
            return ChatCompletionResponse(
                id="x",
                model=model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content="hi"),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(1, 2, 3),
                provider="fake",
            )

    client = ai.Client()
    client.providers["openai"] = FakeProvider()  # inject; bypass real SDK init
    resp = client.chat.completions.create(
        model="openai:gpt-4o", messages=[{"role": "user", "content": "yo"}]
    )
    assert resp.choices[0].message.content == "hi"
    assert resp.model == "gpt-4o"  # provider received the model name without the prefix


def test_openai_normalization_with_tool_calls():
    fake = SimpleNamespace(
        id="resp_1",
        model="gpt-4o",
        created=123,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(
                                name="get_weather", arguments='{"city":"Paris"}'
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    resp = normalize_openai_response(fake, provider="openai")
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.function.name == "get_weather"
    assert json.loads(tc.function.arguments)["city"] == "Paris"
    assert resp.usage.total_tokens == 15
    assert resp.choices[0].finish_reason == "tool_calls"


def test_anthropic_message_conversion():
    conv = AnthropicMessageConverter()
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
    ]
    system, converted = conv.convert_messages(messages)
    assert system == "Be terse."  # system lifted out of the message list
    # assistant tool call became a tool_use block
    assert converted[1]["content"][0]["type"] == "tool_use"
    assert converted[1]["content"][0]["input"] == {"city": "Paris"}
    # tool result became a user tool_result block
    assert converted[2]["role"] == "user"
    assert converted[2]["content"][0]["type"] == "tool_result"
    assert converted[2]["content"][0]["tool_use_id"] == "call_1"


def test_anthropic_tool_schema_conversion():
    conv = AnthropicMessageConverter()
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    out = conv.convert_tools(openai_tools)
    assert out[0]["name"] == "get_weather"
    assert "input_schema" in out[0]  # OpenAI 'parameters' -> Anthropic 'input_schema'


def test_anthropic_response_normalization():
    conv = AnthropicMessageConverter()
    fake = SimpleNamespace(
        id="msg_1",
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="text", text="Let me check."),
            SimpleNamespace(
                type="tool_use", id="tu_1", name="get_weather", input={"city": "Paris"}
            ),
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=8),
    )
    resp = conv.normalize_response(fake, model="claude-opus-4-8")
    assert resp.choices[0].finish_reason == "tool_calls"  # 'tool_use' -> 'tool_calls'
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.function.name == "get_weather"
    assert json.loads(tc.function.arguments) == {"city": "Paris"}
    assert resp.usage.total_tokens == 20


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} smoke tests passed.")


if __name__ == "__main__":
    _run_all()
