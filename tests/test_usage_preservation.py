import json

import pytest

from multi_agent_dashboard.llm_client import LLMClient


class DummyResult:
    def __init__(self, payload, usage_metadata=None, response_metadata=None):
        self._payload = payload
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}

    def model_dump(self):
        return self._payload


class DummyRawMessage:
    def __init__(self, response_metadata=None, usage_metadata=None, additional_kwargs=None):
        self.response_metadata = response_metadata or {}
        self.usage_metadata = usage_metadata
        self.additional_kwargs = additional_kwargs or {}
        self.tool_calls = []
        self.id = "raw_1"
        self.name = None


class DummyModel:
    def __init__(self, result):
        self._result = result

    def invoke(self, *args, **kwargs):
        return self._result

    def ainvoke(self, *args, **kwargs):
        return self._result

    def stream(self, *args, **kwargs):
        yield self._result

    def bind(self, *args, **kwargs):
        return self

    def bind_tools(self, *args, **kwargs):
        return self

    def with_config(self, *args, **kwargs):
        return self

    def with_structured_output(self, *args, **kwargs):
        return self


class DummyAgent:
    def __init__(self, message):
        self.system_prompt = ""
        self._message = message

    def invoke(self, state, context=None):
        return {"messages": [self._message]}


def _run_agent_with_message(message):
    client = LLMClient()
    agent = DummyAgent(message)
    resp = client.invoke_agent(agent, prompt="hi")
    return resp


def test_ollama_structured_with_schema_tokens_preserved():
    usage = {"prompt_tokens": 7, "completion_tokens": 9}
    resp_meta = {"prompt_eval_count": 7, "eval_count": 9}
    payload = {"answer": "ok"}
    raw = DummyRawMessage(response_metadata=resp_meta)
    result = {"raw": raw, "parsed": payload, "parsing_error": None}
    model = DummyModel(result)
    wrapped = LLMClient()._wrap_structured_output_model(model)
    msg = wrapped.invoke(None)

    resp = _run_agent_with_message(msg)
    assert resp.input_tokens == 7
    assert resp.output_tokens == 9
    assert isinstance(resp.raw.get("usage_metadata"), dict)


def test_ollama_structured_no_schema_tokens_preserved():
    usage = {"prompt_tokens": 5, "completion_tokens": 6, "input_tokens": 5, "output_tokens": 6, "total_tokens": 11}
    msg = DummyResult({"answer": "ok"}, usage_metadata=usage, response_metadata={"usage": usage})
    # no wrapper engaged; pass message directly
    resp = _run_agent_with_message(msg)
    assert resp.input_tokens == 5
    assert resp.output_tokens == 6


def test_openai_structured_tokens_preserved():
    usage = {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    msg = DummyResult({"answer": "ok"}, usage_metadata=usage, response_metadata={"usage": usage})
    resp = _run_agent_with_message(msg)
    assert resp.input_tokens == 3
    assert resp.output_tokens == 4
