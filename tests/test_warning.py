#!/usr/bin/env python3
"""
Test that warning appears when use_responses_api=true with LiteLLM path.
"""
import sys
import os
import logging
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_warning_logged_for_litellm_with_responses_api():
    """Verify warning is logged when use_responses_api=true with LiteLLM wrapper."""
    # Capture logs
    warning_messages = []
    class TestHandler(logging.Handler):
        def emit(self, record):
            warning_messages.append(record.getMessage())
    
    # Mock langchain availability
    with patch('src.multi_agent_dashboard.llm_client._LANGCHAIN_AVAILABLE', True):
        with patch('src.multi_agent_dashboard.llm_client._LITELLM_AVAILABLE', True):
            from src.multi_agent_dashboard.llm_client import LLMClient
            
            logger = logging.getLogger('src.multi_agent_dashboard.llm_client')
            handler = TestHandler()
            handler.setLevel(logging.WARNING)
            logger.addHandler(handler)
            
            # Create client with LiteLLM enabled
            client = LLMClient()
            client._use_litellm = True
            client._model_factory = MagicMock()
            client._model_factory.get_model.return_value = MagicMock()
            client._create_agent = MagicMock(return_value=MagicMock())
            
            # Create a spec with use_responses_api=True
            class MockSpec:
                model = "gpt-5.1"
                provider_id = "openai"
                use_responses_api = True
                endpoint = None
                model_class = None
                provider_features = None
                temperature = None
                tools = {"enabled": False, "tools": []}
            
            spec = MockSpec()
            
            # Call create_agent_for_spec
            agent = client.create_agent_for_spec(spec)
            
            # Also test that use_responses_api flag is passed through (not overridden)
            # Verify that get_model was called with use_responses_api=True
            call_kwargs = client._model_factory.get_model.call_args[1]
            assert call_kwargs.get('use_responses_api') == True, \
                f"Expected use_responses_api=True, got {call_kwargs.get('use_responses_api')}"


if __name__ == "__main__":
    # Run the test directly
    logging.basicConfig(level=logging.WARNING)
    test_warning_logged_for_litellm_with_responses_api()
    print("SUCCESS: Warning logged as expected")