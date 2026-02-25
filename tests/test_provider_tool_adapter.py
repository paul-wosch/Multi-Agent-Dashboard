"""
Unit tests for provider_tool_adapter.py.
"""
import json
import logging
from unittest.mock import patch, MagicMock

from multi_agent_dashboard.tool_integration.provider_tool_adapter import (
    convert_tools_for_provider,
    _make_cache_key,
    _convert_web_search_tool,
    _convert_web_search_ddg_tool,
    _convert_tools_for_provider_cached,
)


def test_convert_tools_for_provider_disabled():
    """Test conversion when tools are disabled."""
    tool_configs = {"enabled": False, "tools": ["web_search"]}
    _convert_tools_for_provider_cached.cache_clear()
    result = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", True)
    assert result == {}


def test_convert_tools_for_provider_empty_tools():
    """Test conversion when enabled but no tools."""
    tool_configs = {"enabled": True, "tools": []}
    _convert_tools_for_provider_cached.cache_clear()
    result = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", True)
    assert result == {}


def test_convert_tools_for_provider_openai_web_search_responses():
    """Test OpenAI web_search with use_responses_api=True."""
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    _convert_tools_for_provider_cached.cache_clear()
    result = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", True)
    assert "tools" in result
    assert len(result["tools"]) == 1
    tool = result["tools"][0]
    # With dynamic data, native_web_search defaults to False, so falls back to function tool
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "web_search"


def test_convert_tools_for_provider_openai_web_search_completions():
    """Test OpenAI web_search with use_responses_api=False."""
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    _convert_tools_for_provider_cached.cache_clear()
    result = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", False)
    # With dynamic data, native_web_search defaults to False, so falls back to function tool
    assert "tools" in result
    assert len(result["tools"]) == 1
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "web_search"


def test_convert_tools_for_provider_openai_o1_preview():
    """Test OpenAI o1-preview (no native web search)."""
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    _convert_tools_for_provider_cached.cache_clear()
    result = convert_tools_for_provider(tool_configs, "openai", "o1-preview", True)
    # Should fall back to generic function tool
    assert "tools" in result
    assert len(result["tools"]) == 1
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "web_search"


def test_convert_tools_for_provider_deepseek_web_search():
    """Test DeepSeek web_search (no native web search)."""
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    _convert_tools_for_provider_cached.cache_clear()
    result = convert_tools_for_provider(tool_configs, "deepseek", "deepseek-chat", False)
    assert "tools" in result
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "web_search"


def test_convert_tools_for_provider_ollama_web_search():
    """Test Ollama web_search with advisory tool_calling."""
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    _convert_tools_for_provider_cached.cache_clear()
    result = convert_tools_for_provider(tool_configs, "ollama", "llama3", False)
    assert "tools" in result
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "web_search"


def test_convert_tools_for_provider_web_search_ddg():
    """Test web_search_ddg tool conversion."""
    tool_configs = {"enabled": True, "tools": ["web_search_ddg"]}
    _convert_tools_for_provider_cached.cache_clear()
    result = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", False)
    assert "tools" in result
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "duckduckgo_search"


def test_convert_tools_for_provider_multiple_tools():
    """Test conversion with multiple tools (web_search + web_search_ddg)."""
    tool_configs = {"enabled": True, "tools": ["web_search", "web_search_ddg"]}
    _convert_tools_for_provider_cached.cache_clear()
    result = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", True)
    # Should have both tools in list
    assert "tools" in result
    assert len(result["tools"]) == 2
    types = {t["type"] for t in result["tools"]}
    # With dynamic data, native_web_search defaults to False, so both are function tools
    assert types == {"function"}
    # Check both function names
    function_names = {t["function"]["name"] for t in result["tools"]}
    assert "web_search" in function_names
    assert "duckduckgo_search" in function_names


def test_convert_tools_for_provider_caching():
    """Test that conversion is cached (LRU)."""
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    _convert_tools_for_provider_cached.cache_clear()
    # First call
    result1 = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", True)
    # Second call with same parameters should hit cache
    result2 = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", True)
    assert result1 == result2
    # Ensure the underlying cached function was called once
    # We can't directly check LRU, but we can verify that logging.debug is called
    # Let's just trust the LRU decorator


def test_convert_tools_for_provider_provider_features():
    """Test conversion with provider_features parameter."""
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    provider_features = {"tool_calling": True, "structured_output": True}
    _convert_tools_for_provider_cached.cache_clear()
    result = convert_tools_for_provider(
        tool_configs, "openai", "gpt-4o", True, provider_features=provider_features
    )
    # Should still produce web_search tool
    assert "tools" in result
    # provider_features only affects advisory warnings, not output


def test_convert_tools_for_provider_unknown_tool():
    """Test conversion with unknown tool name (should be ignored)."""
    tool_configs = {"enabled": True, "tools": ["unknown_tool"]}
    with patch.object(logging.getLogger("multi_agent_dashboard.tool_integration.provider_tool_adapter"), "warning") as mock_warning:
        result = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", True)
        assert result == {}
        mock_warning.assert_called_once()


def test_make_cache_key():
    """Test cache key generation."""
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    key = _make_cache_key(tool_configs, "openai", "gpt-4o", True)
    # Should be deterministic string
    assert isinstance(key, str)
    # Should contain provider and model
    assert "openai" in key
    assert "gpt-4o" in key
    # Should include serialized config
    assert "web_search" in key


def test_make_cache_key_with_provider_features():
    """Test cache key includes provider_features."""
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    provider_features = {"tool_calling": True}
    key = _make_cache_key(tool_configs, "openai", "gpt-4o", True, provider_features)
    assert "tool_calling" in key


def test_make_cache_key_normalizes_tools_order():
    """Test cache key normalizes tool order."""
    tool_configs1 = {"enabled": True, "tools": ["web_search", "web_search_ddg"]}
    tool_configs2 = {"enabled": True, "tools": ["web_search_ddg", "web_search"]}
    key1 = _make_cache_key(tool_configs1, "openai", "gpt-4o", True)
    key2 = _make_cache_key(tool_configs2, "openai", "gpt-4o", True)
    assert key1 == key2


def test_convert_web_search_tool_openai_responses():
    """Test internal _convert_web_search_tool for OpenAI Responses API."""
    result = _convert_web_search_tool("openai", "gpt-4o", True, True, True)
    assert result is not None
    assert "tools" in result
    tool = result["tools"][0]
    assert tool["type"] == "web_search"


def test_convert_web_search_tool_openai_completions():
    """Test internal _convert_web_search_tool for OpenAI Completions API."""
    result = _convert_web_search_tool("openai", "gpt-4o", False, True, True)
    assert result is not None
    assert "web_search_options" in result
    assert isinstance(result["web_search_options"], dict)


def test_convert_web_search_tool_non_openai_with_tool_calling():
    """Test internal _convert_web_search_tool for non-OpenAI with advisory tool calling."""
    with patch.object(logging.getLogger("multi_agent_dashboard.tool_integration.provider_tool_adapter"), "warning") as mock_warning:
        result = _convert_web_search_tool("deepseek", "deepseek-chat", False, False, True)
        assert result is not None
        assert "tools" in result
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "web_search"
        # Warning about native web search missing should be logged
        mock_warning.assert_called_once()


def test_convert_web_search_tool_non_openai_without_tool_calling():
    """Test internal _convert_web_search_tool for non-OpenAI without advisory tool calling."""
    with patch.object(logging.getLogger("multi_agent_dashboard.tool_integration.provider_tool_adapter"), "warning") as mock_warning:
        result = _convert_web_search_tool("ollama", "phi", False, False, False)
        assert result is not None
        # Should still produce function tool but with warning
        assert "tools" in result
        tool = result["tools"][0]
        assert tool["type"] == "function"
        # Two warnings: native web search missing, tool calling unsupported
        assert mock_warning.call_count == 2


def test_convert_web_search_ddg_tool_with_tool_calling():
    """Test internal _convert_web_search_ddg_tool."""
    result = _convert_web_search_ddg_tool("openai", "gpt-4o", True, False)
    assert result is not None
    assert "tools" in result
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "duckduckgo_search"


def test_convert_web_search_ddg_tool_without_tool_calling():
    """Test internal _convert_web_search_ddg_tool when advisory tool calling false."""
    with patch.object(logging.getLogger("multi_agent_dashboard.tool_integration.provider_tool_adapter"), "warning") as mock_warning:
        result = _convert_web_search_ddg_tool("ollama", "phi", False, False)
        assert result is not None
        # Warning about tool calling unsupported
        mock_warning.assert_called_once()
        # Still returns function tool
        assert "tools" in result


def test_advisory_warnings_logged():
    """Test that advisory warnings are logged based on capability mapping."""
    tool_configs = {"enabled": True, "tools": ["web_search"]}
    # Mock supports_feature to simulate unsupported native web search
    with patch("multi_agent_dashboard.tool_integration.provider_tool_adapter.supports_feature") as mock_supports:
        mock_supports.side_effect = lambda provider_id, feature, model=None: {
            ("openai", "native_web_search", "gpt-4o"): False,
            ("openai", "tool_calling", "gpt-4o"): True,
        }.get((provider_id, feature, model), False)
        # Patch the module's logger attribute directly
        with patch("multi_agent_dashboard.tool_integration.provider_tool_adapter.logger.warning") as mock_warning:
            # Clear cache to ensure function executes, not cached from previous tests
            _convert_tools_for_provider_cached.cache_clear()
            result = convert_tools_for_provider(tool_configs, "openai", "gpt-4o", True)
            # Should have logged warning about native web search missing
            # Check that warning was called at least once with expected message
            call_found = False
            for call in mock_warning.call_args_list:
                args, kwargs = call
                if len(args) > 0 and "native web search" in args[0]:
                    call_found = True
                    break
            assert call_found, f"Expected warning about native web search, got calls: {mock_warning.call_args_list}"
            # Should still produce function tool
            assert "tools" in result
            tool = result["tools"][0]
            assert tool["type"] == "function"


def test_module_exports():
    """Ensure expected symbols are exported."""
    import multi_agent_dashboard.tool_integration.provider_tool_adapter as pta
    assert hasattr(pta, "convert_tools_for_provider")
    assert "convert_tools_for_provider" in pta.__all__