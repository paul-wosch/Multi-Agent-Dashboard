# tests/test_engine_content_blocks_and_profile.py
from typing import Dict
import pytest

from multi_agent_dashboard.engine import MultiAgentEngine
from multi_agent_dashboard.models import AgentSpec
from multi_agent_dashboard.llm_client import TextResponse


class DummyLLMClientForEngine:
    _langchain_available = True

    def create_agent_for_spec(self, spec, **kwargs):
        class FakeAgent:
            system_prompt = spec.system_prompt_template
        return FakeAgent()

    def invoke_agent(self, agent, prompt, **kwargs):
        content_block = {
            "type": "web_search_call",
            "name": "web_search",
            "args": {"query": "instrumentize"},
            "id": "call-abc",
            "status": "ok",
        }
        events = [
            {
                "content_blocks": [content_block],
                "structured_response": {"answer": "ok"},
            }
        ]

        return TextResponse(
            text='{"answer":"ok"}',
            raw={
                "agent_response": {
                    "_multi_agent_dashboard_events": events,
                    "instrumentation_events": events,
                    "content_blocks": [content_block],
                    "output": [
                        {
                            "response": {
                                "usage": {"prompt_tokens": 3, "completion_tokens": 5},
                                "content_blocks": [content_block],
                                "structured_response": {"answer": "ok"},
                            }
                        }
                    ],
                },
                "detected_provider_profile": {
                    "tool_calling": True,
                    "structured_output": True,
                    "max_input_tokens": 131072,
                },
                "structured_response": {"answer": "ok"},
            },
            input_tokens=3,
            output_tokens=5,
            latency=0.02,
        )

    def create_text_response(self, *args, **kwargs):
        raise RuntimeError("Legacy path should not be called in this test")


class DummyLLMClientTokenFallback:
    _langchain_available = True

    def create_agent_for_spec(self, spec, **kwargs):
        class FakeAgent:
            system_prompt = spec.system_prompt_template
        return FakeAgent()

    def invoke_agent(self, agent, prompt, **kwargs):
        return TextResponse(
            text="{}",
            raw={"usage": {"prompt_tokens": 7, "completion_tokens": 9}},
            input_tokens=None,
            output_tokens=None,
            latency=0.01,
        )

    def create_text_response(self, *args, **kwargs):
        raise RuntimeError("Legacy path should not be called in this test")


def test_engine_persists_content_blocks_and_profile():
    client = DummyLLMClientForEngine()
    engine = MultiAgentEngine(llm_client=client)

    spec = AgentSpec(
        name="tester",
        model="gpt-test",
        prompt_template="Do {task}",
        role="tester",
        input_vars=["task"],
        output_vars=["answer"],
        tools={"enabled": True, "tools": ["web_search"]},
    )

    engine.add_agent(spec)

    result = engine.run_seq(steps=["tester"], initial_input="capture", strict=False)

    agent_cfg = result.agent_configs.get("tester")
    assert agent_cfg is not None

    extra = agent_cfg.get("extra") or {}
    assert "content_blocks" in extra or "content_blocks_summary" in extra
    assert "detected_provider_profile" in extra or agent_cfg.get("provider_features")

    pf = agent_cfg.get("provider_features") or {}
    assert ("tool_calling" in pf and pf["tool_calling"]) or ("detected_profile_present" in pf) or ("structured_output" in pf)

    metrics = result.agent_metrics.get("tester") or {}
    assert metrics.get("input_tokens") == 3
    assert metrics.get("output_tokens") == 5


def test_engine_token_fallback_from_raw_usage():
    client = DummyLLMClientTokenFallback()
    engine = MultiAgentEngine(llm_client=client)

    spec = AgentSpec(
        name="token_tester",
        model="gpt-test",
        prompt_template="Do {task}",
        role="tester",
        input_vars=["task"],
        output_vars=["answer"],
        tools={"enabled": False, "tools": []},
    )

    engine.add_agent(spec)

    result = engine.run_seq(steps=["token_tester"], initial_input="capture", strict=False)

    metrics = result.agent_metrics.get("token_tester") or {}
    assert metrics.get("input_tokens") == 7
    assert metrics.get("output_tokens") == 9


class DummyLLMClientWithToolCalls:
    _langchain_available = True

    def create_agent_for_spec(self, spec, **kwargs):
        class FakeAgent:
            system_prompt = spec.system_prompt_template

        return FakeAgent()

    def invoke_agent(self, agent, prompt, **kwargs):
        return TextResponse(
            text="{}",
            raw={
                "agent_response": {
                    "tool_calls": [
                        {
                            "tool_type": "web_search",
                            "id": "tool-123",
                            "args": {"query": "persist-tool"},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                }
            },
            input_tokens=1,
            output_tokens=2,
            latency=0.005,
        )

    def create_text_response(self, *args, **kwargs):
        raise RuntimeError("Legacy path should not be called")


def test_engine_reports_tool_usage_from_nested_agent_response_tool_calls():
    client = DummyLLMClientWithToolCalls()
    engine = MultiAgentEngine(llm_client=client)

    spec = AgentSpec(
        name="toolcall_tester",
        model="gpt-test",
        prompt_template="Do {task}",
        role="tester",
        input_vars=["task"],
        output_vars=["answer"],
        tools={"enabled": False, "tools": []},
    )
    engine.add_agent(spec)

    result = engine.run_seq(steps=["toolcall_tester"], initial_input="capture", strict=False)

    tool_usages = result.tool_usages.get("toolcall_tester") or []
    assert tool_usages
    assert tool_usages[0]["tool_type"] == "web_search"
    assert result.agent_metrics["toolcall_tester"]["input_tokens"] == 1
    assert result.agent_metrics["toolcall_tester"]["output_tokens"] == 2


class DummyLLMClientWithMergedProviderProfile(DummyLLMClientForEngine):
    _langchain_available = True

    def invoke_agent(self, agent, prompt, **kwargs):
        block = {
            "type": "web_search_call",
            "name": "web_search",
            "args": {"query": "merge-profile"},
            "id": "call-merge-1",
            "status": "ok",
        }
        events = [
            {
                "content_blocks": [block],
                "structured_response": {"answer": "ok"},
                "text": "merge-profile text",
                "ts": 1.2345,
            }
        ]

        return TextResponse(
            text='{"answer":"ok"}',
            raw={
                "instrumentation_events": events,
                "content_blocks": [block],
                "detected_provider_profile": {
                    "tool_calling": True,
                    "structured_output": False,
                    "max_input_tokens": 65536,
                },
                "structured_response": {"answer": "ok"},
            },
            input_tokens=3,
            output_tokens=4,
            latency=0.01,
        )

    def create_text_response(self, *args, **kwargs):
        raise RuntimeError("Legacy path should not be called")


def test_engine_merges_provider_features_with_detected_profile():
    client = DummyLLMClientWithMergedProviderProfile()
    engine = MultiAgentEngine(llm_client=client)

    spec = AgentSpec(
        name="provider_merge",
        model="gpt-test",
        prompt_template="Do {task}",
        role="tester",
        input_vars=["task"],
        output_vars=["answer"],
        tools={"enabled": True, "tools": ["web_search"]},
        provider_features={"structured_output": False},
    )

    engine.add_agent(spec)

    result = engine.run_seq(steps=["provider_merge"], initial_input="capture", strict=False)

    pf = result.agent_configs["provider_merge"]["provider_features"] or {}
    assert pf.get("structured_output") is False
    assert pf.get("tool_calling") is True
    assert pf.get("max_input_tokens") == 65536
