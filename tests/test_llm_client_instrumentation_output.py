# tests/test_llm_client_instrumentation_output.py
from typing import Any

from multi_agent_dashboard.llm_client import LLMClient


def _setup_langchain_client() -> LLMClient:
    client = LLMClient()
    client._langchain_available = True
    client._SystemMessage = lambda text: {"role": "system", "content": text}
    client._HumanMessage = lambda text: {"role": "user", "content": text}
    return client


def test_invoke_agent_merges_instrumentation_and_content_blocks() -> None:
    client = _setup_langchain_client()

    fake_events = [
        {
            "content_blocks": [
                {
                    "type": "web_search_call",
                    "name": "web_search",
                    "args": {"query": "instrumentize"},
                    "id": "call-xyz",
                    "status": "ok",
                }
            ],
            "structured_response": {"answer": "ok"},
            "text": "instrumentation text",
            "ts": 123456.0,
        }
    ]

    class FakeAgent:
        def invoke(self, state: Any, context: Any = None):
            return {
                "agent_response": {
                    "_multi_agent_dashboard_events": fake_events,
                    "instrumentation_events": fake_events,
                    "content_blocks": fake_events[0]["content_blocks"],
                },
                "text": '{"answer":"ok"}',
            }

    resp = client.invoke_agent(FakeAgent(), "do something")

    raw = resp.raw or {}
    assert "instrumentation_events" in raw
    assert "_multi_agent_dashboard_events" in raw
    assert raw["instrumentation_events"] == fake_events
    assert raw["_multi_agent_dashboard_events"] == fake_events

    assert isinstance(raw.get("content_blocks"), list)
    assert raw["content_blocks"][0]["type"] == "web_search_call"
    assert isinstance(resp.text, str) and resp.text


def test_invoke_agent_extracts_tokens_from_nested_usage() -> None:
    client = _setup_langchain_client()

    class FakeAgent:
        def invoke(self, state: Any, context: Any = None):
            return {
                "agent_response": {
                    "output": [
                        {
                            "response": {
                                "usage": {"input_tokens": 12, "completion_tokens": 34},
                                "text": "{}",
                            }
                        }
                    ]
                },
                "text": "{}",
            }

    resp = client.invoke_agent(FakeAgent(), "do something else")

    assert resp.input_tokens == 12
    assert resp.output_tokens == 34
    raw = resp.raw or {}
    usage = raw.get("usage") or raw.get("usage_metadata")
    assert isinstance(usage, dict)
    assert usage.get("input_tokens") == 12
    assert usage.get("completion_tokens") == 34
