#!/usr/bin/env python3
"""
Verify provider tool adapter outputs for different providers and configurations.
"""
import sys
import os
import json
# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, project_root)

from src.multi_agent_dashboard.tool_integration.provider_tool_adapter import convert_tools_for_provider

def test_scenario(name, tools_cfg, provider_id, model, use_responses_api, provider_features=None):
    print(f"\n=== {name} ===")
    print(f"tools_cfg: {tools_cfg}")
    print(f"provider_id: {provider_id}, model: {model}, use_responses_api: {use_responses_api}")
    if provider_features:
        print(f"provider_features: {provider_features}")
    try:
        result = convert_tools_for_provider(
            tools_cfg, provider_id, model, use_responses_api, provider_features
        )
        print(f"Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

# Test cases
# OpenAI with Responses API (web_search)
test_scenario(
    "OpenAI Responses API web_search",
    {"enabled": True, "tools": ["web_search"]},
    "openai", "gpt-4o", True,
    {"tool_calling": True, "structured_output": True}
)

# OpenAI without Responses API (should return web_search_options?)
test_scenario(
    "OpenAI Completions API web_search",
    {"enabled": True, "tools": ["web_search"]},
    "openai", "gpt-4o", False,
    {"tool_calling": True, "structured_output": True}
)

# OpenAI with web_search_ddg (function tool)
test_scenario(
    "OpenAI web_search_ddg",
    {"enabled": True, "tools": ["web_search_ddg"]},
    "openai", "gpt-4o", True,
)

# DeepSeek with web_search (advisory warning expected)
test_scenario(
    "DeepSeek web_search",
    {"enabled": True, "tools": ["web_search"]},
    "deepseek", "deepseek-chat", False,
)

# DeepSeek with web_search_ddg (function tool)
test_scenario(
    "DeepSeek web_search_ddg",
    {"enabled": True, "tools": ["web_search_ddg"]},
    "deepseek", "deepseek-chat", False,
)

# Ollama with web_search (advisory warning expected)
test_scenario(
    "Ollama web_search",
    {"enabled": True, "tools": ["web_search"]},
    "ollama", "llama3", False,
)

# Ollama with web_search_ddg (function tool)
test_scenario(
    "Ollama web_search_ddg",
    {"enabled": True, "tools": ["web_search_ddg"]},
    "ollama", "llama3", False,
)

# Disabled tools
test_scenario(
    "Disabled tools",
    {"enabled": False, "tools": ["web_search"]},
    "openai", "gpt-4o", True,
)

# No tools config
test_scenario(
    "Empty tools config",
    {},
    "openai", "gpt-4o", True,
)

# Multiple tools
test_scenario(
    "Multiple tools",
    {"enabled": True, "tools": ["web_search", "web_search_ddg"]},
    "openai", "gpt-4o", True,
)

print("\n=== All scenarios completed ===")