"""
Integration tests for multimodal file handling.
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





def test_multimodal_fallback_when_no_vision():
    """
    When provider does not support vision (image_inputs=False), multimodal_handler
    should return a concatenated string.
    """
    with patch('multi_agent_dashboard.llm_client.multimodal.prepare_multimodal_content') as mock_prepare:
        mock_prepare.return_value = (
            "Prompt\n\n--- FILE: test.txt ---\nHello",
            []
        )
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


def test_legacy_concatenation_when_multimodal_handler_not_available():
    """
    When multimodal_handler cannot be imported, the legacy concatenation path should be used.
    """
    import sys
    # Temporarily remove the module to cause ImportError
    with patch.dict(sys.modules, {'multi_agent_dashboard.llm_client.multimodal': None}):
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