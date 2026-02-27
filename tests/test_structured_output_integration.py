"""
Integration tests for structured output with mocked providers.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard import config


class MockSpec:
    """Minimal AgentSpec-like object for testing."""
    def __init__(self, structured_output_enabled=False, schema_json=None, schema_name=None, model="gpt-4o", provider_id="openai", endpoint=None, use_responses_api=False, model_class=None, provider_features=None, temperature=None, max_output=0):
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
        self.max_output = max_output
        self.system_prompt_template = None

    def effective_max_output(self) -> int | None:
        """Mock implementation returning None (no limit)."""
        return None







def test_structured_output_adapter_provider_specific():
    """Verify _build_structured_output_adapter returns provider-specific formats."""
    client = LLMClient()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"]
    }
    
    # Test with OpenAI provider
    spec = MockSpec(
        structured_output_enabled=True,
        schema_json=schema,
        schema_name="test",
        provider_id="openai"
    )
    result = client._build_structured_output_adapter(spec, None)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["schema"] == schema
    assert "strict" not in result["json_schema"]
    
    # Test with Ollama provider
    spec.provider_id = "ollama"
    result2 = client._build_structured_output_adapter(spec, None)
    # Ollama returns raw schema dict
    assert result2 == schema
    
    # Test with DeepSeek provider
    spec.provider_id = "deepseek"
    result3 = client._build_structured_output_adapter(spec, None)
    # DeepSeek returns function-calling wrapper
    assert "name" in result3
    assert "description" in result3
    assert "parameters" in result3
    assert result3["parameters"] == schema


def test_create_agent_for_spec_passes_response_format_provider_specific():
    """Test create_agent_for_spec passes response_format (provider-specific LangChain path)."""
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
    
    # Patch binding to return original response_format (simulate binding failure)
    with patch('multi_agent_dashboard.llm_client.structured_output.StructuredOutputBinder.bind_structured_output') as mock_bind:
        # Make bind_structured_output return original response_format (simulating failure)
        mock_bind.side_effect = lambda spec, model_instance, response_format, provider_id, model, tools=None, strict=True: (model_instance, response_format)
        # Call create_agent_for_spec
        agent = client.create_agent_for_spec(spec, response_format=None)
    
    # Verify get_model called with correct arguments
    client._model_factory.get_model.assert_called_once()
    # Verify _create_agent called with response_format
    client._create_agent.assert_called_once()
    call_kwargs = client._create_agent.call_args.kwargs
    assert "response_format" in call_kwargs
    response_format = call_kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == schema



def test_structured_output_with_explicit_response_format():
    """Explicit response_format overrides schema (provider-specific LangChain path)."""
    client = LLMClient()
    client._langchain_available = True
    client._create_agent = Mock()
    client._model_factory = Mock()
    mock_model = Mock()
    client._model_factory.get_model = Mock(return_value=mock_model)
    
    explicit_format = {"type": "json_object"}
    spec = MockSpec(structured_output_enabled=True)  # schema present but ignored
    
    # Patch binding to return original response_format (simulate binding failure)
    with patch('multi_agent_dashboard.llm_client.structured_output.StructuredOutputBinder.bind_structured_output') as mock_bind:
        # Make bind_structured_output return original response_format (simulating failure)
        mock_bind.side_effect = lambda spec, model_instance, response_format, provider_id, model, tools=None, strict=True: (model_instance, response_format)
        agent = client.create_agent_for_spec(spec, response_format=explicit_format)
    
    client._create_agent.assert_called_once()
    call_kwargs = client._create_agent.call_args.kwargs
    assert call_kwargs["response_format"] == explicit_format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])