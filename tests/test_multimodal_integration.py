"""
Integration tests for multimodal file handling across USE_LITELLM paths.
"""
import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard import config


def _normalize_message(msg):
    """Extract (role, content) from a message dict or LangChain message object."""
    if hasattr(msg, 'content'):
        # LangChain message object
        content = msg.content
        # HumanMessage.type == 'human', SystemMessage.type == 'system', AIMessage.type == 'ai'
        role = msg.type if hasattr(msg, 'type') else 'unknown'
        return role, content
    elif isinstance(msg, dict):
        return msg.get('role', 'unknown'), msg.get('content')
    else:
        raise TypeError(f"Unsupported message type: {type(msg)}")

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


def test_multimodal_handler_used_when_litellm_true():
    """
    When USE_LITELLM=True and provider supports vision, multimodal_handler
    should be called and return content parts.
    """
    with patch.object(config, 'USE_LITELLM', True):
        # Mock the multimodal_handler module
        with patch('multi_agent_dashboard.multimodal_handler.prepare_multimodal_content') as mock_prepare:
            mock_prepare.return_value = (
                [{"type": "text", "text": "--- FILE: test.txt ---\nHello"}],
                []  # no processed files
            )
            # Create LLMClient with mocked litellm availability
            with patch('multi_agent_dashboard.llm_client._LITELLM_AVAILABLE', True):
                with patch('multi_agent_dashboard.llm_client._ChatLiteLLM'):
                    client = LLMClient()
                    # Mock agent with provider_features indicating vision support
                    agent = Mock()
                    agent.system_prompt = None
                    agent._provider_id = "openai"
                    agent._model = "gpt-4o"
                    agent._provider_features = {"image_inputs": True}
                    agent.invoke = Mock(return_value={"output": "response"})
                    
                    # Call invoke_agent with a text file
                    files = [{"filename": "test.txt", "content": b"Hello", "mime_type": "text/plain"}]
                    result = client.invoke_agent(
                        agent=agent,
                        prompt="Prompt",
                        files=files,
                        response_format=None,
                        stream=False,
                        context=None,
                    )
                    
                    # Ensure prepare_multimodal_content was called with correct args
                    mock_prepare.assert_called_once()
                    call_args = mock_prepare.call_args
                    assert call_args.kwargs['provider_id'] == "openai"
                    assert call_args.kwargs['model'] == "gpt-4o"
                    assert call_args.kwargs['files'] == files
                    assert call_args.kwargs['profile'] == {"image_inputs": True}
                    assert call_args.kwargs['prompt'] == "Prompt"
                    
                    # Agent.invoke should have been called with content parts
                    agent.invoke.assert_called_once()
                    call_args = agent.invoke.call_args
                    messages = call_args[0][0]['messages']
                    # Expect a single user message with content parts
                    assert len(messages) == 1
                    role, content = _normalize_message(messages[0])
                    assert role in ('user', 'human')
                    assert content == [{"type": "text", "text": "--- FILE: test.txt ---\nHello"}]


def test_multimodal_fallback_when_no_vision():
    """
    When provider does not support vision (image_inputs=False), multimodal_handler
    should return a concatenated string.
    """
    with patch.object(config, 'USE_LITELLM', True):
        with patch('multi_agent_dashboard.multimodal_handler.prepare_multimodal_content') as mock_prepare:
            mock_prepare.return_value = (
                "Prompt\n\n--- FILE: test.txt ---\nHello",
                []
            )
            with patch('multi_agent_dashboard.llm_client._LITELLM_AVAILABLE', True):
                with patch('multi_agent_dashboard.llm_client._ChatLiteLLM'):
                    client = LLMClient()
                    agent = Mock()
                    agent.system_prompt = None
                    agent._provider_id = "openai"
                    agent._model = "gpt-4o"
                    agent._provider_features = {"image_inputs": False}
                    agent.invoke = Mock(return_value={"output": "response"})
                    
                    files = [{"filename": "test.txt", "content": b"Hello", "mime_type": "text/plain"}]
                    result = client.invoke_agent(
                        agent=agent,
                        prompt="Prompt",
                        files=files,
                        response_format=None,
                        stream=False,
                        context=None,
                    )
                    
                    mock_prepare.assert_called_once()
                    # Agent.invoke should have been called with string content
                    agent.invoke.assert_called_once()
                    call_args = agent.invoke.call_args
                    messages = call_args[0][0]['messages']
                    assert len(messages) == 1
                    role, content = _normalize_message(messages[0])
                    assert role in ('user', 'human')
                    assert content == "Prompt\n\n--- FILE: test.txt ---\nHello"


def test_legacy_concatenation_when_litellm_false():
    """
    When USE_LITELLM=False, the legacy concatenation path should be used
    (multimodal_handler not imported).
    """
    with patch.object(config, 'USE_LITELLM', False):
        # Ensure multimodal_handler is not imported (mock import error)
        with patch('multi_agent_dashboard.llm_client._LITELLM_AVAILABLE', False):
            client = LLMClient()
            # Mock agent (no provider_features needed)
            agent = Mock()
            agent.system_prompt = None
            agent._provider_id = "openai"
            agent._model = "gpt-4o"
            agent._provider_features = None
            agent.invoke = Mock(return_value={"output": "response"})
            
            # Spy on the fallback concatenation logic by checking the result
            files = [{"filename": "test.txt", "content": b"Hello", "mime_type": "text/plain"}]
            result = client.invoke_agent(
                agent=agent,
                prompt="Prompt",
                files=files,
                response_format=None,
                stream=False,
                context=None,
            )
            
            # The legacy concatenation should produce a string with the file appended
            agent.invoke.assert_called_once()
            call_args = agent.invoke.call_args
            messages = call_args[0][0]['messages']
            assert len(messages) == 1
            role, content = _normalize_message(messages[0])
            assert role in ('user', 'human')
            assert isinstance(content, str)
            assert "--- FILE: test.txt ---" in content
            assert "Hello" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])