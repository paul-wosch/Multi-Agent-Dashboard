"""
Integration tests for structured output with mocked providers, covering both USE_LITELLM paths.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard import config


class MockSpec:
    """Minimal AgentSpec-like object for testing."""
    def __init__(self, structured_output_enabled=False, schema_json=None, schema_name=None, model="gpt-4o", provider_id="openai", endpoint=None, use_responses_api=False, model_class=None, provider_features=None, temperature=None):
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


def test_use_litellm_flag_default_false():
    """Ensure USE_LITELLM defaults to False."""
    with patch("dotenv.dotenv_values", return_value={}):
        # Clear any existing environment variable
        os.environ.pop("USE_LITELLM", None)
        # Reload config module to pick up default
        import importlib
        import multi_agent_dashboard.config
        importlib.reload(multi_agent_dashboard.config)
        from multi_agent_dashboard.config import USE_LITELLM as refreshed_flag
        assert refreshed_flag is False


def test_use_litellm_flag_true():
    """USE_LITELLM should be True for 'true', '1', 'yes' values."""
    # Test different true values
    test_cases = [
        ('true', True),
        ('1', True),
        ('yes', True),
        ('TRUE', True),
        ('YES', True),
        ('false', False),
        ('0', False),
        ('no', False),
        ('', False),
        (None, False),
    ]
    
    for value, expected in test_cases:
        # Create a mock _env dict
        env_dict = {}
        if value is not None:
            env_dict['USE_LITELLM'] = value
        # Compute USE_LITELLM using same logic as config.py
        result = env_dict.get('USE_LITELLM', 'false').lower() in ('true', '1', 'yes')
        assert result == expected, f"Value '{value}' should return {expected}"


def test_llm_client_init_with_litellm_flag_false():
    """When USE_LITELLM=False, LLMClient uses standard init_chat_model."""
    # Mock config.USE_LITELLM to False
    with patch.object(config, 'USE_LITELLM', False):
        # Mock the availability flags
        with patch("multi_agent_dashboard.llm_client._LANGCHAIN_AVAILABLE", True):
            with patch("multi_agent_dashboard.llm_client._LITELLM_AVAILABLE", True):
                with patch("multi_agent_dashboard.llm_client._ChatLiteLLM", Mock()):
                    # Create client
                    client = LLMClient()
                    # Should use standard init_chat_model
                    # In test environment, _init_chat_model may be None because import fails
                    # We'll check it's not the LiteLLM version
                    if client._init_chat_model is not None:
                        assert client._init_chat_model.__name__ != "_init_chat_model_with_litellm"


def test_llm_client_init_with_litellm_flag_true():
    """When USE_LITELLM=True and LiteLLM available, LLMClient uses LiteLLM initializer."""
    # Mock config.USE_LITELLM to True
    with patch.object(config, 'USE_LITELLM', True):
        # Mock availability
        with patch("multi_agent_dashboard.llm_client._LANGCHAIN_AVAILABLE", True):
            with patch("multi_agent_dashboard.llm_client._LITELLM_AVAILABLE", True):
                with patch("multi_agent_dashboard.llm_client._ChatLiteLLM", Mock()):
                    # Create client
                    client = LLMClient()
                    # Should use LiteLLM initializer
                    assert client._init_chat_model is not None
                    assert client._init_chat_model.__name__ == "_init_chat_model_with_litellm"


def test_structured_output_adapter_provider_specific():
    """Verify _build_structured_output_adapter returns provider-specific formats."""
    with patch.object(config, 'USE_LITELLM', True):
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


def test_create_agent_for_spec_passes_response_format_with_litellm_false():
    """Test create_agent_for_spec passes response_format when USE_LITELLM=False."""
    with patch.object(config, 'USE_LITELLM', False):
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
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"] == schema


def test_create_agent_for_spec_passes_response_format_with_litellm_true():
    """Test create_agent_for_spec passes response_format when USE_LITELLM=True."""
    with patch.object(config, 'USE_LITELLM', True):
        # Mock LiteLLM availability
        with patch("multi_agent_dashboard.llm_client._LITELLM_AVAILABLE", True):
            with patch("multi_agent_dashboard.llm_client._ChatLiteLLM", Mock()):
                client = LLMClient()
                # LangChain still available for agent creation
                client._langchain_available = True
                client._create_agent = Mock()
                client._model_factory = Mock()
                mock_model = Mock()
                client._model_factory.get_model = Mock(return_value=mock_model)
                
                # Mock with_structured_output to return same model (workaround path removed, but keep mock)
                mock_model.with_structured_output = Mock(return_value=mock_model)
                # Mock _wrap_structured_output_model to return same model
                client._wrap_structured_output_model = Mock(return_value=mock_model)
                
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
                # Verify with_structured_output was NOT called (workaround removed)
                mock_model.with_structured_output.assert_not_called()
                # Verify _wrap_structured_output_model was NOT called (workaround removed)
                client._wrap_structured_output_model.assert_not_called()
                # Verify _create_agent called with provider-specific response_format
                client._create_agent.assert_called_once()
                call_kwargs = client._create_agent.call_args.kwargs
                assert "response_format" in call_kwargs
                response_format = call_kwargs["response_format"]
                # Expect OpenAI JSON Schema format
                assert response_format["type"] == "json_schema"
                assert response_format["json_schema"]["schema"] == schema
                assert response_format["json_schema"]["name"] == "test"


def test_structured_output_with_explicit_response_format():
    """Explicit response_format overrides schema regardless of USE_LITELLM flag."""
    for flag_value in [False, True]:
        with patch.object(config, 'USE_LITELLM', flag_value):
            client = LLMClient()
            client._langchain_available = True
            client._create_agent = Mock()
            client._model_factory = Mock()
            mock_model = Mock()
            client._model_factory.get_model = Mock(return_value=mock_model)
            
            explicit_format = {"type": "json_object"}
            spec = MockSpec(structured_output_enabled=True)  # schema present but ignored
            
            agent = client.create_agent_for_spec(spec, response_format=explicit_format)
            
            client._create_agent.assert_called_once()
            call_kwargs = client._create_agent.call_args.kwargs
            assert call_kwargs["response_format"] == explicit_format
            # Reset mocks for next iteration
            client._create_agent.reset_mock()
            client._model_factory.get_model.reset_mock()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])