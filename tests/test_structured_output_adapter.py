"""
Unit tests for the unified structured output adapter (Step 3 of LiteLLM integration).
"""
import pytest
from unittest.mock import Mock, patch
from multi_agent_dashboard import config
# Force LiteLLM path for these tests (they test unified LiteLLM JSON Schema format)
config.USE_LITELLM = True
from multi_agent_dashboard.llm_client import LLMClient


class MockSpec:
    """Minimal AgentSpec-like object for testing."""
    def __init__(self, structured_output_enabled=False, schema_json=None, schema_name=None, model="test-model", provider_id="openai", endpoint=None, use_responses_api=False, model_class=None, provider_features=None, temperature=None):
        self.structured_output_enabled = structured_output_enabled
        self.schema_json = schema_json
        self.schema_name = schema_name
        self.model = model
        self.provider_id = provider_id
        self.endpoint = endpoint
        self.use_responses_api = use_responses_api
        self.model_class = model_class
        self.provider_features = provider_features
        self.temperature = temperature
        self.system_prompt_template = None


def test_build_structured_output_adapter_no_schema():
    """When structured_output_enabled is False, returns None."""
    client = LLMClient()
    spec = MockSpec(structured_output_enabled=False)
    result = client._build_structured_output_adapter(spec, None)
    assert result is None


def test_build_structured_output_adapter_with_explicit_response_format():
    """Explicit response_format is passed through unchanged."""
    client = LLMClient()
    spec = MockSpec(structured_output_enabled=True)
    explicit = {"type": "json_object"}
    result = client._build_structured_output_adapter(spec, explicit)
    assert result == explicit


def test_build_structured_output_adapter_with_schema_dict():
    """When schema_json is a dict, returns LiteLLM JSON Schema format."""
    client = LLMClient()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"]
    }
    spec = MockSpec(
        structured_output_enabled=True,
        schema_json=schema,
        schema_name="test_schema"
    )
    result = client._build_structured_output_adapter(spec, None)
    assert result is not None
    assert result["type"] == "json_schema"
    assert "json_schema" in result
    json_schema = result["json_schema"]
    assert json_schema["name"] == "test_schema"
    assert json_schema["schema"] == schema
    assert json_schema["strict"] is True


def test_build_structured_output_adapter_with_schema_string():
    """schema_json as JSON string is parsed into dict."""
    client = LLMClient()
    schema_str = '{"type": "object", "properties": {"count": {"type": "integer"}}}'
    spec = MockSpec(
        structured_output_enabled=True,
        schema_json=schema_str,
        schema_name="count_schema"
    )
    result = client._build_structured_output_adapter(spec, None)
    assert result is not None
    assert result["type"] == "json_schema"
    json_schema = result["json_schema"]
    assert json_schema["name"] == "count_schema"
    assert json_schema["schema"] == {"type": "object", "properties": {"count": {"type": "integer"}}}
    assert json_schema["strict"] is True


def test_build_structured_output_adapter_with_schema_name_only():
    """When schema_json is None but schema_name references a registered schema."""
    from multi_agent_dashboard.structured_schemas import register_schema, get_schema
    # Clean up registry before test
    from multi_agent_dashboard.structured_schemas import SCHEMA_REGISTRY
    SCHEMA_REGISTRY.clear()
    
    schema = {"type": "object", "properties": {"foo": {"type": "string"}}}
    register_schema("my_schema", schema)
    
    client = LLMClient()
    spec = MockSpec(
        structured_output_enabled=True,
        schema_json=None,
        schema_name="my_schema"
    )
    result = client._build_structured_output_adapter(spec, None)
    assert result is not None
    assert result["type"] == "json_schema"
    json_schema = result["json_schema"]
    assert json_schema["name"] == "my_schema"
    assert json_schema["schema"] == schema
    assert json_schema["strict"] is True


def test_build_structured_output_adapter_no_schema_returns_none():
    """When no schema can be resolved, returns None."""
    client = LLMClient()
    spec = MockSpec(structured_output_enabled=True, schema_json=None, schema_name="nonexistent")
    result = client._build_structured_output_adapter(spec, None)
    assert result is None


def test_create_agent_for_spec_passes_response_format():
    """Test that create_agent_for_spec passes response_format to _create_agent."""
    client = LLMClient()
    # Enable LangChain path
    client._langchain_available = True
    client._create_agent = Mock()
    client._model_factory = Mock()
    mock_model = Mock()
    client._model_factory.get_model = Mock(return_value=mock_model)
    
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    spec = MockSpec(
        structured_output_enabled=True,
        schema_json=schema,
        schema_name="test",
        model="gpt-4o",
        provider_id="openai"
    )
    
    # Call create_agent_for_spec
    agent = client.create_agent_for_spec(spec, response_format=None)
    
    # Verify get_model called with correct arguments
    client._model_factory.get_model.assert_called_once()
    # Verify _create_agent called with response_format
    client._create_agent.assert_called_once()
    call_kwargs = client._create_agent.call_args.kwargs
    assert "response_format" in call_kwargs
    response_format = call_kwargs["response_format"]
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == schema
    # Ensure other parameters passed
    assert call_kwargs["model"] == mock_model
    assert call_kwargs["system_prompt"] is None
    assert len(call_kwargs["middleware"]) == 1  # Instrumentation middleware
    assert call_kwargs["tools"] == []


def test_create_agent_for_spec_with_explicit_response_format():
    """Explicit response_format overrides schema."""
    client = LLMClient()
    client._langchain_available = True
    client._create_agent = Mock()
    client._model_factory = Mock()
    mock_model = Mock()
    client._model_factory.get_model = Mock(return_value=mock_model)
    
    explicit_format = {"type": "json_object"}
    spec = MockSpec(structured_output_enabled=False)  # schema not used
    
    agent = client.create_agent_for_spec(spec, response_format=explicit_format)
    
    client._create_agent.assert_called_once()
    call_kwargs = client._create_agent.call_args.kwargs
    assert call_kwargs["response_format"] == explicit_format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])