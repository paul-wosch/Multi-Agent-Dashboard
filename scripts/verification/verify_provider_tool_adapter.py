#!/usr/bin/env python3
"""
Quick test for provider_tool_adapter.py
"""
import sys
import logging
sys.path.insert(0, "src")

logging.basicConfig(level=logging.DEBUG)

from multi_agent_dashboard.tool_integration.provider_tool_adapter import convert_tools_for_provider

def test_basic():
    print("=== Testing provider_tool_adapter ===")
    
    # Test 1: OpenAI with web_search, use_responses_api=True
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    result = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", use_responses_api=True)
    print(f"OpenAI gpt-4o web_search (Responses API): {result}")
    assert "tools" in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["type"] == "web_search"
    
    # Test 2: OpenAI with web_search, use_responses_api=False (Completions API)
    result = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", use_responses_api=False)
    print(f"OpenAI gpt-4o web_search (Completions API): {result}")
    assert "web_search_options" in result
    assert isinstance(result["web_search_options"], dict)
    
    # Test 3: OpenAI o1-preview (no native web search)
    result = convert_tools_for_provider(tool_configs, "openai", "o1-preview", use_responses_api=True)
    print(f"OpenAI o1-preview web_search: {result}")
    # Should fall back to generic function tool because native_web_search=False
    assert "tools" in result
    assert result["tools"][0]["type"] == "function"
    assert result["tools"][0]["function"]["name"] == "web_search"
    
    # Test 4: DeepSeek with web_search (no native web search)
    result = convert_tools_for_provider(tool_configs, "deepseek", "deepseek-chat", use_responses_api=False)
    print(f"DeepSeek web_search: {result}")
    assert "tools" in result
    assert result["tools"][0]["type"] == "function"
    
    # Test 5: Ollama with web_search (no tool calling advisory)
    result = convert_tools_for_provider(tool_configs, "ollama", "llama3", use_responses_api=False)
    print(f"Ollama llama3 web_search: {result}")
    # Ollama llama3 has tool_calling=True advisory, but native_web_search=False
    # Should produce function tool
    assert "tools" in result
    assert result["tools"][0]["type"] == "function"
    
    # Test 6: web_search_ddg tool (function tool)
    tool_configs_ddg = {"enabled": True, "tools": ["web_search_ddg"]}
    result = convert_tools_for_provider(tool_configs_ddg, "openai", "gpt-4o", use_responses_api=False)
    print(f"OpenAI web_search_ddg: {result}")
    assert "tools" in result
    assert result["tools"][0]["type"] == "function"
    assert result["tools"][0]["function"]["name"] == "duckduckgo_search"
    
    # Test 7: DeepSeek with web_search_ddg
    result = convert_tools_for_provider(tool_configs_ddg, "deepseek", "deepseek-chat", use_responses_api=False)
    print(f"DeepSeek web_search_ddg: {result}")
    assert "tools" in result
    
    # Test 8: Multiple tools
    tool_configs_multi = {"enabled": True, "tools": ["web_search", "web_search_ddg"]}
    result = convert_tools_for_provider(tool_configs_multi, "openai", "gpt-4o", use_responses_api=True)
    print(f"OpenAI multi-tools (Responses): {result}")
    # Should have both tools in list? Wait: web_search is native web search (type web_search) and web_search_ddg is function tool
    # But OpenAI Responses API can't mix web_search type with function tools? Actually, they can: tools param can include both.
    # Our adapter returns tools list with web_search type and function tool.
    # Let's check.
    if "tools" in result:
        print(f"  tools list length: {len(result['tools'])}")
        for t in result["tools"]:
            print(f"    - {t.get('type')}")
    
    # Test 9: Disabled tools
    tool_configs_disabled = {"enabled": False, "tools": ["web_search"]}
    result = convert_tools_for_provider(tool_configs_disabled, "openai", "gpt-4o", use_responses_api=True)
    print(f"Disabled tools: {result}")
    assert result == {}
    
    # Test 10: No tools list
    tool_configs_empty = {"enabled": True, "tools": []}
    result = convert_tools_for_provider(tool_configs_empty, "openai", "gpt-4o", use_responses_api=True)
    print(f"Empty tools list: {result}")
    assert result == {}
    
    print("\nAll basic tests passed!")

if __name__ == "__main__":
    test_basic()