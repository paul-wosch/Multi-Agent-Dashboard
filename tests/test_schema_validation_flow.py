import json
import pytest
from multi_agent_dashboard.engine import MultiAgentEngine
from multi_agent_dashboard.models import AgentSpec


def _make_engine_with_response(text, structured=None):
    class DummyClient:
        def __init__(self):
            self.calls = []
            self._langchain_available = True

        def create_agent_for_spec(self, *args, **kwargs):
            return object()

        def invoke_agent(self, agent, prompt, **kwargs):
            self.calls.append(prompt)
            return type(
                "Resp",
                (),
                {
                    "text": text,
                    "raw": {"structured_response": structured} if structured else {},
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "latency": 0.1,
                },
            )

    engine = MultiAgentEngine(llm_client=DummyClient())
    return engine


def test_schema_missing_sets_flag_and_warning():
    engine = _make_engine_with_response('{"answer":"ok"}')
    spec = AgentSpec(name="a", model="m", prompt_template="Do {task}", output_vars=["answer"], structured_output_enabled=True)
    engine.add_agent(spec)
    res = engine.run_seq(steps=["a"], initial_input="hi", strict_schema_validation=True)
    assert res.agent_schema_validation_failed.get("a") is True
    assert res.strict_schema_exit is True


def test_schema_empty_sets_flag_and_warning():
    engine = _make_engine_with_response('{"answer":"ok"}')
    spec = AgentSpec(name="a", model="m", prompt_template="Do {task}", output_vars=["answer"], structured_output_enabled=True, schema_json="{}")
    engine.add_agent(spec)
    res = engine.run_seq(steps=["a"], initial_input="hi", strict_schema_validation=True)
    assert res.agent_schema_validation_failed.get("a") is True
    assert res.strict_schema_exit is True


def test_invalid_output_sets_flag_and_exit_when_strict():
    engine = _make_engine_with_response("not json")
    spec = AgentSpec(
        name="a",
        model="m",
        prompt_template="Do {task}",
        output_vars=["answer"],
        structured_output_enabled=True,
        schema_json=json.dumps({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}),
    )
    engine.add_agent(spec)
    res = engine.run_seq(steps=["a"], initial_input="hi", strict_schema_validation=True)
    assert res.agent_schema_validation_failed.get("a") is True
    assert res.strict_schema_exit is True


def test_valid_output_passes_without_exit():
    engine = _make_engine_with_response('{"answer":"ok"}')
    spec = AgentSpec(
        name="a",
        model="m",
        prompt_template="Do {task}",
        output_vars=["answer"],
        structured_output_enabled=True,
        schema_json=json.dumps({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}),
    )
    engine.add_agent(spec)
    res = engine.run_seq(steps=["a"], initial_input="hi", strict_schema_validation=True)
    assert res.agent_schema_validation_failed.get("a") is not True
    assert res.strict_schema_exit is False
