#!/usr/bin/env python3
"""
Test web search fixes for LiteLLM integration.
"""
import sys
import os
import logging
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_conversion_with_mocked_litellm():
    """Test convert_tools_for_litellm with mocked litellm detection."""
    from src.multi_agent_dashboard.tool_integration.litellm_tool_adapter import convert_tools_for_litellm
    
    # Mock litellm module
    with patch('src.multi_agent_dashboard.tool_integration.litellm_tool_adapter._LITELLM_AVAILABLE', True):
        with patch('src.multi_agent_dashboard.tool_integration.litellm_tool_adapter.litellm') as mock_litellm:
            # Scenario 1: GPT-5.1, web_search_options reported but native web search unsupported
            mock_litellm.get_supported_openai_params.return_value = ["tools", "web_search_options"]
            mock_litellm.supports_web_search.return_value = False
            
            tool_configs = {"enabled": True, "tools": ["web_search"]}
            result = convert_tools_for_litellm(tool_configs, "openai", "gpt-5.1", use_responses_api=False)
            print(f"Scenario 1 (GPT-5.1, use_responses_api=False): {result}")
            # Expected: web_search excluded because native web search unsupported
            assert result == {}, f"Expected empty dict, got {result}"
            
            # Scenario 2: GPT-5.1, use_responses_api=True (should still be excluded)
            result = convert_tools_for_litellm(tool_configs, "openai", "gpt-5.1", use_responses_api=True)
            print(f"Scenario 2 (GPT-5.1, use_responses_api=True): {result}")
            assert result == {}, f"Expected empty dict, got {result}"
            
            # Scenario 3: gpt-4o-search-preview, native web search supported, web_search_options supported
            mock_litellm.get_supported_openai_params.return_value = ["tools", "web_search_options"]
            mock_litellm.supports_web_search.return_value = True
            result = convert_tools_for_litellm(tool_configs, "openai", "gpt-4o-search-preview", use_responses_api=False)
            print(f"Scenario 3 (gpt-4o-search-preview, use_responses_api=False): {result}")
            # Should produce web_search_options dict
            assert "web_search_options" in result, f"Expected web_search_options, got {result}"
            
            # Scenario 4: same model, use_responses_api=True (should produce tools list)
            result = convert_tools_for_litellm(tool_configs, "openai", "gpt-4o-search-preview", use_responses_api=True)
            print(f"Scenario 4 (gpt-4o-search-preview, use_responses_api=True): {result}")
            # Should produce tools list with web_search type
            assert "tools" in result, f"Expected tools, got {result}"
            assert len(result["tools"]) == 1, f"Expected one tool, got {len(result['tools'])}"
            assert result["tools"][0]["type"] == "web_search", f"Expected web_search type, got {result['tools'][0]}"
            
            # Scenario 5: web_search_ddg tool (should produce function calling tool)
            tool_configs = {"enabled": True, "tools": ["web_search_ddg"]}
            # Ensure supports_tools_param is True
            mock_litellm.get_supported_openai_params.return_value = ["tools"]
            result = convert_tools_for_litellm(tool_configs, "openai", "gpt-4o", use_responses_api=False)
            print(f"Scenario 5 (web_search_ddg): {result}")
            assert "tools" in result, f"Expected tools for web_search_ddg, got {result}"
            # Should have function calling schema
            
            print("All conversion tests passed.")

def test_llm_client_override():
    """Test that LLMClient passes use_responses_api flag to tool adapter (no override)."""
    from src.multi_agent_dashboard.llm_client import LLMClient
    
    # Mock dependencies
    with patch('src.multi_agent_dashboard.llm_client._LANGCHAIN_AVAILABLE', True):
        with patch('src.multi_agent_dashboard.llm_client._LITELLM_AVAILABLE', True):
            with patch('src.multi_agent_dashboard.llm_client.ChatModelFactory') as MockFactory:
                mock_factory = MagicMock()
                mock_factory.get_model.return_value = MagicMock()
                MockFactory.return_value = mock_factory
                
                # Mock convert_tools_for_provider to capture arguments
                with patch('src.multi_agent_dashboard.llm_client.convert_tools_for_provider') as mock_convert:
                    mock_convert.return_value = {}
                    
                    # Create a mock spec with use_responses_api=True and tools enabled
                    class MockSpec:
                        model = "gpt-5.1"
                        provider_id = "openai"
                        use_responses_api = True
                        endpoint = None
                        model_class = None
                        provider_features = None
                        temperature = None
                        tools = {"enabled": True, "tools": ["web_search"]}
                    
                    # Test with LiteLLM enabled (set _use_litellm directly)
                    client = LLMClient()
                    client._use_litellm = True
                    
                    # Mock _create_agent to avoid actual agent creation
                    client._create_agent = MagicMock(return_value=MagicMock())
                    
                    # Create a dummy tool to trigger tool conversion
                    dummy_tool = MagicMock()
                    
                    # Call create_agent_for_spec with tools to trigger conversion
                    agent = client.create_agent_for_spec(MockSpec(), tools=[dummy_tool])
                    
                    # Verify that convert_tools_for_provider was called with use_responses_api=True
                    # (no longer overridden for LiteLLM path)
                    mock_convert.assert_called_once()
                    call_args = mock_convert.call_args
                    # The fourth argument is use_responses_api
                    call_use_responses_api = call_args[0][3]
                    print(f"convert_tools_for_provider called with use_responses_api={call_use_responses_api} (LiteLLM path)")
                    assert call_use_responses_api == True, f"Expected use_responses_api=True for LiteLLM path (no override), got {call_use_responses_api}"
                    
                    # Reset mock and test with LiteLLM disabled
                    mock_convert.reset_mock()
                    client._use_litellm = False
                    client._create_agent = MagicMock(return_value=MagicMock())
                    
                    agent = client.create_agent_for_spec(MockSpec(), tools=[dummy_tool])
                    # For non-LiteLLM path, conversion should NOT be called
                    assert mock_convert.call_count == 0, f"convert_tools_for_provider should not be called for non-LiteLLM path, but was called {mock_convert.call_count} times"
                    # Verify that use_responses_api flag was passed to model factory in the SECOND call (after LiteLLM disabled)
                    assert mock_factory.get_model.call_count >= 2, f"Expected at least 2 calls to get_model, got {mock_factory.get_model.call_count}"
                    second_call_args = mock_factory.get_model.call_args_list[1]
                    # second_call_args is a tuple of (args, kwargs)
                    call_kwargs = second_call_args[1]
                    factory_use_responses_api = call_kwargs.get('use_responses_api', None)
                    print(f"Model factory second call with use_responses_api={factory_use_responses_api} (non-LiteLLM)")
                    assert factory_use_responses_api == True, f"Expected use_responses_api=True for non-LiteLLM path in model factory, got {factory_use_responses_api}"
                    
                    print("LLMClient override test passed.")

if __name__ == "__main__":
    print("Running web search fix tests...")
    test_conversion_with_mocked_litellm()
    test_llm_client_override()
    print("All tests passed.")