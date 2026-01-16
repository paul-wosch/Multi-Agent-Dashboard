# tests/test_llm_client_langchain.py
import types
from typing import Dict

import pytest

from multi_agent_dashboard.llm_client import (
    ChatModelFactory,
    INSTRUMENTATION_MIDDLEWARE,
    LLMClient,
    TextResponse,
)
from multi_agent_dashboard.models import AgentRuntime, AgentSpec


class DummyModelFactory:
    def __init__(self):
        self.calls = []

    def get_model(self, model: str, **kwargs: Dict[str, Dict[str, object]]) -> str:
        self.calls.append((model, kwargs))
        return f"model::{model}::{kwargs.get('model_provider')}"


def test_chat_model_factory_is_provider_aware():
    factory = ChatModelFactory(init_fn=lambda model, **kwargs: f"{model}:{kwargs.get('model_provider')}")
    m1 = factory.get_model(
        "gpt-4.1-nano",
        provider_id="openai",
        endpoint=None,
        use_responses_api=True,
        model_class="openai-reg",
        provider_features={"structured_output": True},
        timeout=5,
    )
    m2 = factory.get_model(
        "gpt-4.1-nano",
        provider_id="ollama",
        endpoint="http://localhost:11434",
        use_responses_api=False,
        model_class="ollama-mini",
        provider_features={"structured_output": False},
        timeout=5,
    )
    assert m1 != m2, "Factory should cache models by provider metadata"


def test_create_agent_for_spec_injects_response_format_and_instrumentation(monkeypatch):
    recorded = {}

    def fake_create_agent(*, model, tools, middleware, system_prompt, response_format, **kwargs):
        recorded["response_format"] = response_format
        recorded["middleware"] = middleware

        class AgentStub:
            def __init__(self, system_prompt_value):
                self.system_prompt = system_prompt_value

        return AgentStub(system_prompt)

    monkeypatch.setattr("multi_agent_dashboard.llm_client._create_agent", fake_create_agent)
    monkeypatch.setattr("multi_agent_dashboard.llm_client._init_chat_model", lambda *args, **kw: "dummy-model")
    monkeypatch.setattr("multi_agent_dashboard.llm_client._SystemMessage", lambda text: {"role": "system", "content": text})
    monkeypatch.setattr("multi_agent_dashboard.llm_client._HumanMessage", lambda text: {"role": "user", "content": text})
    monkeypatch.setattr("multi_agent_dashboard.llm_client._LANGCHAIN_AVAILABLE", True)

    client = LLMClient()
    client._model_factory = DummyModelFactory()

    spec = types.SimpleNamespace(
        model="gpt-4.1-nano",
        provider_id="openai",
        endpoint=None,
        use_responses_api=True,
        model_class=None,
        provider_features={},
        system_prompt_template="system",
    )
    schema = {"type": "object", "properties": {"plan": {"type": "string"}}}
    _ = client.create_agent_for_spec(spec, response_format=schema)

    assert recorded["response_format"] is schema
    assert recorded["middleware"], "Instrumentation middleware must be added"
    assert INSTRUMENTATION_MIDDLEWARE is not None
    assert isinstance(recorded["middleware"][-1], INSTRUMENTATION_MIDDLEWARE)


def test_agent_runtime_consumes_instrumentation_events():
    class DummyLLMClient:
        _langchain_available = True

        def create_agent_for_spec(self, spec, **kwargs):
            class FakeAgent:
                system_prompt = spec.system_prompt_template
            return FakeAgent()

        def invoke_agent(self, agent, prompt, **kwargs):
            return TextResponse(
                text="{}",
                raw={
                    "_multi_agent_dashboard_events": [
                        {
                            "content_blocks": [
                                {
                                    "type": "web_search_call",
                                    "name": "web_search",
                                    "args": {"query": "instrumentize"},
                                    "id": "call-123",
                                    "status": "ok",
                                }
                            ],
                            "structured_response": {"plan": "instrumented"},
                        }
                    ],
                    "structured_response": {"plan": "instrumented"},
                },
                input_tokens=2,
                output_tokens=4,
                latency=0.05,
            )

        def create_text_response(self, *args, **kwargs):
            raise RuntimeError("Should not hit legacy path")

    spec = AgentSpec(
        name="instrumented",
        model="gpt-4.1-nano",
        prompt_template="Plan: {task}",
        role="planner",
        input_vars=["task"],
        output_vars=["plan"],
        tools={"enabled": False, "tools": []},
        reasoning_effort="medium",
        reasoning_summary="auto",
    )
    runtime = AgentRuntime(spec=spec, llm_client=DummyLLMClient())
    runtime.run({"task": "capture traces"})

    assert runtime.state["plan"] == "instrumented"
    tools = runtime.last_metrics.get("tools") or []
    assert tools and tools[0]["tool_type"] == "web_search"


def test_chat_model_factory_honors_ollama_provider_and_endpoint():
    """
    Regression: ensure ChatModelFactory honors provider metadata for non-OpenAI providers.

    - provider_id should be propagated as model_provider to the init function
    - endpoint should be passed as base_url to the init function (Ollama uses base_url)
    - provider_features should be passed through as 'profile'
    - identical metadata must yield cached instance (same object)
    - changing metadata (provider_features or endpoint) must yield a new instance
    """
    recorded = []

    def init_fn(model, **kwargs):
        # Record arguments passed to the low-level init function
        recorded.append((model, dict(kwargs)))
        # return a distinct object so identity checks work for caching assertions
        return {"model": model, "kwargs": dict(kwargs)}

    factory = ChatModelFactory(init_fn=init_fn)

    endpoint = "http://localhost:11434"
    m1 = factory.get_model(
        "llama3",
        provider_id="ollama",
        endpoint=endpoint,
        use_responses_api=False,
        model_class="ollama-mini",
        provider_features={"structured_output": False},
        timeout=3.5,
    )

    # init_fn should have been called once with the expected forwarded kwargs
    assert len(recorded) == 1
    model_arg, kw = recorded[0]
    assert model_arg == "llama3"
    # ChatModelFactory forwards provider_id -> model_provider (init_chat_model semantics)
    assert kw.get("model_provider") == "ollama"
    # endpoint should be forwarded as base_url for provider integrations like Ollama
    assert kw.get("base_url") == endpoint
    # provider_features forwarded under 'profile'
    assert kw.get("profile") == {"structured_output": False}

    # Repeating the exact same call should return the cached instance (no new init)
    m2 = factory.get_model(
        "llama3",
        provider_id="ollama",
        endpoint=endpoint,
        use_responses_api=False,
        model_class="ollama-mini",
        provider_features={"structured_output": False},
        timeout=3.5,
    )
    assert m1 is m2
    assert len(recorded) == 1

    # Changing provider_features should change the cache key and cause a new init
    m3 = factory.get_model(
        "llama3",
        provider_id="ollama",
        endpoint=endpoint,
        use_responses_api=False,
        model_class="ollama-mini",
        provider_features={"structured_output": True},
        timeout=3.5,
    )
    assert m3 is not m1
    assert len(recorded) == 2
