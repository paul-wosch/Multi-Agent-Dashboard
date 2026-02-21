"""
Unit tests for the structured output adapter.
"""
import pytest
from unittest.mock import Mock, patch
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
    """When schema_json is a dict, returns raw schema for Ollama provider."""
    client = LLMClient()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"]
    }
    spec = MockSpec(
        structured_output_enabled=True,
        schema_json=schema,
        schema_name="test_schema",
        provider_id="ollama"
    )
    result = client._build_structured_output_adapter(spec, None)
    assert result is not None
    # Ollama returns raw schema dict
    assert result == schema
    # No wrapper
    assert "json_schema" not in result


def test_build_structured_output_adapter_with_schema_string():
    """schema_json as JSON string is parsed into dict; Ollama returns raw schema."""
    client = LLMClient()
    schema_str = '{"type": "object", "properties": {"count": {"type": "integer"}}}'
    spec = MockSpec(
        structured_output_enabled=True,
        schema_json=schema_str,
        schema_name="count_schema",
        provider_id="ollama"
    )
    result = client._build_structured_output_adapter(spec, None)
    assert result is not None
    # Ollama returns parsed schema dict
    expected = {"type": "object", "properties": {"count": {"type": "integer"}}}
    assert result == expected
    # No wrapper
    assert "json_schema" not in result


def test_build_structured_output_adapter_with_schema_name_only():
    """When schema_json is None but schema_name references a registered schema; Ollama returns raw schema."""
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
        schema_name="my_schema",
        provider_id="ollama"
    )
    result = client._build_structured_output_adapter(spec, None)
    assert result is not None
    # Ollama returns raw schema dict (name ignored)
    assert result == schema
    # No wrapper
    assert "json_schema" not in result


def test_build_structured_output_adapter_no_schema_returns_none():
    """When no schema can be resolved, returns None."""
    client = LLMClient()
    spec = MockSpec(structured_output_enabled=True, schema_json=None, schema_name="nonexistent")
    result = client._build_structured_output_adapter(spec, None)
    assert result is None


def test_create_agent_for_spec_passes_response_format():
    """Test that create_agent_for_spec passes response_format to _create_agent."""
    # Ensure response_format is passed
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
    expected_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "test",
            "schema": schema,
        },
    }
    mock_build = Mock(return_value=expected_response_format)
    # Patch binding to return original response_format (simulate binding failure)
    with patch.object(client, '_build_structured_output_adapter', mock_build), \
         patch('multi_agent_dashboard.llm_client.structured_output.StructuredOutputBinder.bind_structured_output') as mock_bind:
        # Make bind_structured_output return original response_format (simulating failure)
        mock_bind.return_value = (mock_model, expected_response_format)
        agent = client.create_agent_for_spec(spec, response_format=None)
    mock_build.assert_called_once_with(spec, None)
    
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
    
    with patch.object(client, '_build_structured_output_adapter', return_value=explicit_format), \
         patch('multi_agent_dashboard.llm_client.structured_output.StructuredOutputBinder.bind_structured_output') as mock_bind:
        # Make bind_structured_output return original response_format (simulating failure)
        mock_bind.return_value = (mock_model, explicit_format)
        agent = client.create_agent_for_spec(spec, response_format=explicit_format)
    
    client._create_agent.assert_called_once()
    call_kwargs = client._create_agent.call_args.kwargs
    assert call_kwargs["response_format"] == explicit_format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])