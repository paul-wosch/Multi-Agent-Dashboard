"""
Unit tests for provider_capabilities.py.
"""
import logging
from unittest.mock import patch

from multi_agent_dashboard.shared.provider_capabilities import (
    get_capabilities,
    supports_feature,
    PROVIDER_DEFAULT_CAPABILITIES,
    MODEL_CAPABILITIES,
)


def test_get_capabilities_openai_default():
    """Test get_capabilities for OpenAI without model."""
    caps = get_capabilities("openai")
    expected = PROVIDER_DEFAULT_CAPABILITIES["openai"]
    assert caps == expected


def test_get_capabilities_openai_with_model():
    """Test get_capabilities for OpenAI with specific model."""
    caps = get_capabilities("openai", "gpt-4o")
    # Should include provider defaults plus model-specific overrides
    assert caps["vision"] is True
    assert caps["image_inputs"] is True
    assert caps["max_input_tokens"] == 128000
    # Ensure other provider defaults still present
    assert caps["tool_calling"] is True
    assert caps["structured_output"] is True


def test_get_capabilities_openai_model_unknown():
    """Test get_capabilities for unknown model (should log debug)."""
    with patch.object(logging.getLogger("multi_agent_dashboard.shared.provider_capabilities"), "debug") as mock_debug:
        caps = get_capabilities("openai", "unknown-model")
        # Should return provider defaults unchanged
        expected = PROVIDER_DEFAULT_CAPABILITIES["openai"]
        assert caps == expected
        # Debug log should have been emitted
        mock_debug.assert_called_once()


def test_get_capabilities_deepseek_default():
    """Test get_capabilities for DeepSeek."""
    caps = get_capabilities("deepseek")
    expected = PROVIDER_DEFAULT_CAPABILITIES["deepseek"]
    assert caps == expected


def test_get_capabilities_deepseek_model():
    """Test get_capabilities for DeepSeek with model-specific overrides."""
    caps = get_capabilities("deepseek", "deepseek-reasoner")
    assert caps["reasoning"] is True
    assert caps["max_input_tokens"] == 128000
    # Ensure provider defaults still present
    assert caps["tool_calling"] is True
    assert caps["structured_output"] is True
    assert caps["image_inputs"] is False


def test_get_capabilities_ollama_default():
    """Test get_capabilities for Ollama."""
    caps = get_capabilities("ollama")
    expected = PROVIDER_DEFAULT_CAPABILITIES["ollama"]
    assert caps == expected


def test_get_capabilities_ollama_vision_model():
    """Test get_capabilities for Ollama vision-capable model."""
    caps = get_capabilities("ollama", "llava")
    assert caps["vision"] is True
    assert caps["image_inputs"] is True
    # Provider defaults still present
    assert caps["tool_calling"] is False  # default for ollama
    assert caps["structured_output"] is True


def test_get_capabilities_ollama_tool_calling_model():
    """Test get_capabilities for Ollama model with tool_calling advisory."""
    caps = get_capabilities("ollama", "llama3")
    assert caps["tool_calling"] is True
    assert caps["function_calling"] is True
    assert caps["tools"] is True


def test_get_capabilities_unknown_provider():
    """Test get_capabilities for unknown provider returns empty dict."""
    with patch.object(logging.getLogger("multi_agent_dashboard.shared.provider_capabilities"), "debug") as mock_debug:
        caps = get_capabilities("unknown_provider")
        assert caps == {}
        mock_debug.assert_called_once()


def test_get_capabilities_case_insensitive():
    """Test get_capabilities normalizes provider_id to lowercase."""
    caps1 = get_capabilities("OpenAI")
    caps2 = get_capabilities("openai")
    assert caps1 == caps2


def test_supports_feature_openai():
    """Test supports_feature for OpenAI."""
    assert supports_feature("openai", "tool_calling") is True
    assert supports_feature("openai", "vision") is True
    assert supports_feature("openai", "reasoning") is False  # default
    assert supports_feature("openai", "reasoning", model="o1-preview") is True


def test_supports_feature_deepseek():
    """Test supports_feature for DeepSeek."""
    assert supports_feature("deepseek", "tool_calling") is True
    assert supports_feature("deepseek", "vision") is False
    assert supports_feature("deepseek", "reasoning", model="deepseek-reasoner") is True
    assert supports_feature("deepseek", "reasoning", model="deepseek-chat") is False


def test_supports_feature_ollama():
    """Test supports_feature for Ollama."""
    assert supports_feature("ollama", "tool_calling") is False  # default
    assert supports_feature("ollama", "tool_calling", model="llama3") is True
    assert supports_feature("ollama", "vision", model="llava") is True
    assert supports_feature("ollama", "vision", model="phi") is False


def test_supports_feature_unknown_provider():
    """Test supports_feature for unknown provider returns False."""
    assert supports_feature("unknown", "tool_calling") is False


def test_supports_feature_unknown_feature():
    """Test supports_feature for unknown feature returns False."""
    assert supports_feature("openai", "unknown_feature") is False


def test_get_capabilities_model_case_insensitive():
    """Test get_capabilities lowercases model names."""
    caps1 = get_capabilities("openai", "GPT-4O")
    caps2 = get_capabilities("openai", "gpt-4o")
    assert caps1 == caps2
    # Should have vision capability (true for gpt-4o)
    assert caps1["vision"] is True


def test_supports_feature_model_case_insensitive():
    """Test supports_feature lowercases model names."""
    assert supports_feature("openai", "vision", model="GPT-4O") is True
    assert supports_feature("openai", "vision", model="gpt-4o") is True


def test_get_capabilities_empty_provider():
    """Test get_capabilities with empty provider string returns empty dict."""
    caps = get_capabilities("")
    assert caps == {}


def test_get_capabilities_empty_model():
    """Test get_capabilities with empty model string uses provider defaults."""
    caps = get_capabilities("openai", "")
    expected = PROVIDER_DEFAULT_CAPABILITIES["openai"]
    assert caps == expected


def test_get_capabilities_returns_copy():
    """Test that get_capabilities returns a mutable copy, not the original dict."""
    caps = get_capabilities("openai")
    caps["custom_key"] = True
    # Original should not be affected
    assert "custom_key" not in PROVIDER_DEFAULT_CAPABILITIES["openai"]


def test_supports_feature_int_capability():
    """Test supports_feature returns int value for max_input_tokens."""
    result = supports_feature("openai", "max_input_tokens")
    assert isinstance(result, int)
    assert result > 0


def test_module_exports():
    """Ensure expected symbols are exported."""
    import multi_agent_dashboard.shared.provider_capabilities as pc
    assert hasattr(pc, "get_capabilities")
    assert hasattr(pc, "supports_feature")
    assert hasattr(pc, "PROVIDER_DEFAULT_CAPABILITIES")
    assert hasattr(pc, "MODEL_CAPABILITIES")
    assert "get_capabilities" in pc.__all__
    assert "supports_feature" in pc.__all__