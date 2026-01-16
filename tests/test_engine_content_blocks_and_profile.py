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
        # Return a text response that includes detected_provider_profile and content_blocks
        return TextResponse(
            text='{"answer":"ok"}',
            raw={
                "_multi_agent_dashboard_events": [
                    {
                        "content_blocks": [
                            {
                                "type": "web_search_call",
                                "name": "web_search",
                                "args": {"query": "instrumentize"},
                                "id": "call-abc",
                                "status": "ok",
                            }
                        ],
                        "structured_response": {"answer": "ok"},
                    }
                ],
                # Simulate model-level profile detection
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

    # Agent configs should include detected profile and normalized content_blocks
    agent_cfg = result.agent_configs.get("tester")
    assert agent_cfg is not None

    # Extra should contain content_blocks (normalized) and detected_provider_profile
    extra = agent_cfg.get("extra") or {}
    assert "content_blocks" in extra or "content_blocks_summary" in extra
    assert "detected_provider_profile" in extra or agent_cfg.get("provider_features")

    # provider_features should reflect the derived hint (tool_calling/structured_output)
    pf = agent_cfg.get("provider_features") or {}
    # If derived features were set, they should include at least tool_calling and structured_output keys
    assert ("tool_calling" in pf and pf["tool_calling"]) or ("detected_profile_present" in pf) or ("structured_output" in pf)
