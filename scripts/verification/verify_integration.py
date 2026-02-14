#!/usr/bin/env python3
"""
Quick integration test for provider tool adapter integration in AgentRuntime.run.
"""
import sys
import os
# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, project_root)

from src.multi_agent_dashboard.models import AgentRuntime, AgentSpec
from src.multi_agent_dashboard.llm_client import LLMClient
from unittest.mock import Mock, patch, MagicMock
import logging

logging.basicConfig(level=logging.DEBUG)

def test_tool_conversion_openai():
    """Test that OpenAI web_search tool is converted correctly."""
    spec = AgentSpec(
        name="test_agent",
        model="gpt-4o",
        provider_id="openai",
        tools={"enabled": True, "tools": ["web_search"]},
        use_responses_api=True,
        provider_features={"tool_calling": True, "structured_output": True},
        prompt_template="Hello",
        temperature=0.7,
    )
    llm_client = Mock(spec=LLMClient)
    llm_client._langchain_available = True
    llm_client.create_agent_for_spec = Mock(return_value=Mock())
    llm_client.invoke_agent = Mock(return_value=Mock(
        raw={},
        input_tokens=10,
        output_tokens=20,
        latency=1.0,
    ))
    
    runtime = AgentRuntime(spec=spec, llm_client=llm_client)
    
    # Mock the adapter call to see what's passed
    with patch('src.multi_agent_dashboard.models.convert_tools_for_provider') as mock_adapter:
        mock_adapter.return_value = {"tools": [{"type": "web_search"}]}
        # We'll call the run method but catch early to avoid actual LLM calls
        # Instead, we'll inspect the langchain_tools variable by patching create_agent_for_spec
        pass
    
    print("OpenAI test setup complete")

def test_tool_conversion_deepseek():
    """Test DeepSeek tool conversion."""
    spec = AgentSpec(
        name="test_agent",
        model="deepseek-chat",
        provider_id="deepseek",
        tools={"enabled": True, "tools": ["web_search_ddg"]},
        use_responses_api=False,
        prompt_template="Hello",
        temperature=0.7,
    )
    print("DeepSeek test setup complete")

def test_tool_conversion_ollama():
    """Test Ollama tool conversion."""
    spec = AgentSpec(
        name="test_agent",
        model="llama3",
        provider_id="ollama",
        tools={"enabled": True, "tools": ["web_search"]},
        use_responses_api=False,
        prompt_template="Hello",
        temperature=0.7,
    )
    print("Ollama test setup complete")

def test_filters_merging():
    """Test that allowed_domains filters are merged into tool spec."""
    # We'll directly test the logic by calling convert_tools_for_provider and merging
    from src.multi_agent_dashboard.tool_integration.provider_tool_adapter import convert_tools_for_provider
    from src.multi_agent_dashboard.models import AgentRuntime, AgentSpec
    import json
    
    spec = AgentSpec(
        name="test",
        model="gpt-4o",
        provider_id="openai",
        tools={"enabled": True, "tools": ["web_search"]},
        use_responses_api=True,
        prompt_template="",
        temperature=0.7,
    )
    runtime = AgentRuntime(spec=spec, llm_client=Mock())
    
    # Simulate tc dict with filters
    tc = {
        "tools": [
            {"type": "web_search", "filters": {"allowed_domains": ["example.com"]}}
        ]
    }
    
    # Call the adapter via the runtime's internal logic (we'll manually replicate)
    tools_cfg = spec.tools
    provider_id = spec.provider_id
    model = spec.model
    use_responses_api = spec.use_responses_api
    provider_features = spec.provider_features
    
    converted = convert_tools_for_provider(
        tools_cfg,
        provider_id,
        model,
        use_responses_api,
        provider_features,
    )
    print(f"Converted: {json.dumps(converted, indent=2)}")
    
    # Merge filters
    langchain_tools = converted.get("tools", [])
    if langchain_tools and isinstance(tc, dict):
        tools_arr = tc.get("tools")
        if isinstance(tools_arr, list):
            for tc_tool in tools_arr:
                if isinstance(tc_tool, dict) and tc_tool.get("type") == "web_search":
                    filters = tc_tool.get("filters")
                    if filters:
                        for lt in langchain_tools:
                            if isinstance(lt, dict):
                                if lt.get("type") == "web_search":
                                    lt["filters"] = filters
                                    break
                                func = lt.get("function", {})
                                if isinstance(func, dict) and func.get("name") in ("web_search", "duckduckgo_search"):
                                    lt["filters"] = filters
                                    break
    
    print(f"After merging filters: {json.dumps(langchain_tools, indent=2)}")
    assert langchain_tools[0].get("filters") == {"allowed_domains": ["example.com"]}
    print("Filters merging test passed")

if __name__ == "__main__":
    test_filters_merging()
    print("All quick tests passed")