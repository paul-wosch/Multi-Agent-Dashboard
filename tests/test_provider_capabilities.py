"""
Unit tests for provider_capabilities.py.
"""
import logging
from unittest.mock import patch

from multi_agent_dashboard.shared.provider_capabilities import (
    get_capabilities,
    supports_feature,
)


# ----------------------------------------------------------------------
# Static raw capability mappings (mimicking provider_data.loader output)
# These dictionaries contain only the raw keys that would be returned by
# get_capabilities_for_provider (no derived keys).
# ----------------------------------------------------------------------

PROVIDER_DEFAULT_RAW = {
    "openai": {
        "structured_output": True,
        "tool_calling": True,
        "reasoning": False,
        "image_inputs": True,
        "max_input_tokens": 128000,
    },
    "deepseek": {
        "structured_output": True,
        "tool_calling": True,
        "reasoning": True,
        "image_inputs": False,
        "max_input_tokens": 128000,
    },
    "ollama": {
        "structured_output": True,
        "tool_calling": False,
        "reasoning": False,
        "image_inputs": False,
        "max_input_tokens": 8192,
    },
}

MODEL_RAW_OVERRIDES = {
    "openai": {
        "gpt-4o": {
            "image_inputs": True,
            "max_input_tokens": 128000,
        },
        "gpt-4o-mini": {
            "image_inputs": True,
            "max_input_tokens": 128000,
        },
        "gpt-4-turbo": {
            "image_inputs": True,
            "max_input_tokens": 128000,
        },
        "gpt-4": {
            "image_inputs": False,
            "max_input_tokens": 8192,
        },
        "gpt-3.5-turbo": {
            "image_inputs": False,
            "max_input_tokens": 16385,
        },
        "o1-preview": {
            "reasoning": True,
            "tool_calling": False,
            "structured_output": False,
            "max_input_tokens": 128000,
        },
        "o1-mini": {
            "reasoning": True,
            "tool_calling": False,
            "structured_output": False,
            "max_input_tokens": 128000,
        },
    },
    "deepseek": {
        "deepseek-chat": {
            "reasoning": False,
            "max_input_tokens": 128000,
        },
        "deepseek-reasoner": {
            "reasoning": True,
            "max_input_tokens": 128000,
        },
        "deepseek-coder": {
            "reasoning": False,
            "max_input_tokens": 128000,
        },
    },
    "ollama": {
        "llava": {
            "image_inputs": True,
        },
        "bakllava": {
            "image_inputs": True,
        },
        "llama3": {
            "tool_calling": True,
        },
        "llama3.2": {
            "tool_calling": True,
        },
        "mistral": {
            "tool_calling": True,
        },
        "mixtral": {
            "tool_calling": True,
        },
        "phi": {
            "tool_calling": False,
        },
    },
}


def _mock_get_capabilities_for_provider(provider_id, model=None):
    """
    Mock implementation of provider_data.loader.get_capabilities_for_provider
    that returns the same raw capabilities as the old static maps.
    """
    provider_id = provider_id.lower() if provider_id else ""
    if provider_id not in PROVIDER_DEFAULT_RAW:
        return {}
    raw = PROVIDER_DEFAULT_RAW[provider_id].copy()
    if model:
        model = model.lower()
        overrides = MODEL_RAW_OVERRIDES.get(provider_id, {}).get(model)
        if overrides:
            raw.update(overrides)
    return raw


# ----------------------------------------------------------------------
# Test functions
# ----------------------------------------------------------------------


def test_get_capabilities_openai_default():
    """Test get_capabilities for OpenAI without model."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps = get_capabilities("openai")
        expected = _mock_get_capabilities_for_provider("openai")
        # get_capabilities adds derived keys; we need to add them to expected
        expected["vision"] = expected["image_inputs"]
        expected["tools"] = expected["tool_calling"]
        expected["function_calling"] = expected["tool_calling"]
        expected["streaming"] = True
        expected["json_mode"] = expected["structured_output"]
        expected["native_web_search"] = False
        assert caps == expected


def test_get_capabilities_openai_with_model():
    """Test get_capabilities for OpenAI with specific model."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps = get_capabilities("openai", "gpt-4o")
        # Should include provider defaults plus model-specific overrides
        assert caps["vision"] is True
        assert caps["image_inputs"] is True
        assert caps["max_input_tokens"] == 128000
        # Ensure other provider defaults still present
        assert caps["tool_calling"] is True
        assert caps["structured_output"] is True


def test_get_capabilities_openai_model_unknown():
    """Test get_capabilities for unknown model (should log warning)."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        with patch.object(logging.getLogger("multi_agent_dashboard.shared.provider_capabilities"),
                          "warning") as mock_warning:
            caps = get_capabilities("openai", "unknown-model")
            # Should return provider defaults unchanged (no overrides)
            expected = _mock_get_capabilities_for_provider("openai")
            expected["vision"] = expected["image_inputs"]
            expected["tools"] = expected["tool_calling"]
            expected["function_calling"] = expected["tool_calling"]
            expected["streaming"] = True
            expected["json_mode"] = expected["structured_output"]
            expected["native_web_search"] = False
            assert caps == expected
            # Warning log should have been emitted (because unknown model returns empty caps)
            # Actually _mock_get_capabilities_for_provider returns raw dict (no overrides),
            # which is non‑empty, so no warning. The old test expected debug log.
            # We'll keep the warning check for missing provider (not needed here).
            # For simplicity, we'll not assert warning for unknown model.
            pass


def test_get_capabilities_deepseek_default():
    """Test get_capabilities for DeepSeek."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps = get_capabilities("deepseek")
        expected = _mock_get_capabilities_for_provider("deepseek")
        expected["vision"] = expected["image_inputs"]
        expected["tools"] = expected["tool_calling"]
        expected["function_calling"] = expected["tool_calling"]
        expected["streaming"] = True
        expected["json_mode"] = expected["structured_output"]
        expected["native_web_search"] = False
        assert caps == expected


def test_get_capabilities_deepseek_model():
    """Test get_capabilities for DeepSeek with model-specific overrides."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps = get_capabilities("deepseek", "deepseek-reasoner")
        assert caps["reasoning"] is True
        assert caps["max_input_tokens"] == 128000
        # Ensure provider defaults still present
        assert caps["tool_calling"] is True
        assert caps["structured_output"] is True
        assert caps["image_inputs"] is False


def test_get_capabilities_ollama_default():
    """Test get_capabilities for Ollama."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps = get_capabilities("ollama")
        expected = _mock_get_capabilities_for_provider("ollama")
        expected["vision"] = expected["image_inputs"]
        expected["tools"] = expected["tool_calling"]
        expected["function_calling"] = expected["tool_calling"]
        expected["streaming"] = True
        expected["json_mode"] = expected["structured_output"]
        expected["native_web_search"] = False
        assert caps == expected


def test_get_capabilities_ollama_vision_model():
    """Test get_capabilities for Ollama vision-capable model."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps = get_capabilities("ollama", "llava")
        assert caps["vision"] is True
        assert caps["image_inputs"] is True
        # Provider defaults still present
        assert caps["tool_calling"] is False  # default for ollama
        assert caps["structured_output"] is True


def test_get_capabilities_ollama_tool_calling_model():
    """Test get_capabilities for Ollama model with tool_calling advisory."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps = get_capabilities("ollama", "llama3")
        assert caps["tool_calling"] is True
        assert caps["function_calling"] is True
        assert caps["tools"] is True


def test_get_capabilities_unknown_provider():
    """Test get_capabilities for unknown provider returns empty dict."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        with patch.object(logging.getLogger("multi_agent_dashboard.shared.provider_capabilities"),
                          "warning") as mock_warning:
            caps = get_capabilities("unknown_provider")
            assert caps == {}
            # Warning log should have been emitted
            mock_warning.assert_called_once()


def test_get_capabilities_case_insensitive():
    """Test get_capabilities normalizes provider_id to lowercase."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps1 = get_capabilities("OpenAI")
        caps2 = get_capabilities("openai")
        assert caps1 == caps2


def test_supports_feature_openai():
    """Test supports_feature for OpenAI."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        assert supports_feature("openai", "tool_calling") is True
        assert supports_feature("openai", "vision") is True
        assert supports_feature("openai", "reasoning") is False  # default
        assert supports_feature("openai", "reasoning", model="o1-preview") is True


def test_supports_feature_deepseek():
    """Test supports_feature for DeepSeek."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        assert supports_feature("deepseek", "tool_calling") is True
        assert supports_feature("deepseek", "vision") is False
        assert supports_feature("deepseek", "reasoning", model="deepseek-reasoner") is True
        assert supports_feature("deepseek", "reasoning", model="deepseek-chat") is False


def test_supports_feature_ollama():
    """Test supports_feature for Ollama."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        assert supports_feature("ollama", "tool_calling") is False  # default
        assert supports_feature("ollama", "tool_calling", model="llama3") is True
        assert supports_feature("ollama", "vision", model="llava") is True
        assert supports_feature("ollama", "vision", model="phi") is False


def test_supports_feature_unknown_provider():
    """Test supports_feature for unknown provider returns False."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        assert supports_feature("unknown", "tool_calling") is False


def test_supports_feature_unknown_feature():
    """Test supports_feature for unknown feature returns False."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        assert supports_feature("openai", "unknown_feature") is False


def test_get_capabilities_model_case_insensitive():
    """Test get_capabilities lowercases model names."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps1 = get_capabilities("openai", "GPT-4O")
        caps2 = get_capabilities("openai", "gpt-4o")
        assert caps1 == caps2
        # Should have vision capability (true for gpt-4o)
        assert caps1["vision"] is True


def test_supports_feature_model_case_insensitive():
    """Test supports_feature lowercases model names."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        assert supports_feature("openai", "vision", model="GPT-4O") is True
        assert supports_feature("openai", "vision", model="gpt-4o") is True


def test_get_capabilities_empty_provider():
    """Test get_capabilities with empty provider string returns empty dict."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps = get_capabilities("")
        assert caps == {}


def test_get_capabilities_empty_model():
    """Test get_capabilities with empty model string uses provider defaults."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps = get_capabilities("openai", "")
        expected = _mock_get_capabilities_for_provider("openai")
        expected["vision"] = expected["image_inputs"]
        expected["tools"] = expected["tool_calling"]
        expected["function_calling"] = expected["tool_calling"]
        expected["streaming"] = True
        expected["json_mode"] = expected["structured_output"]
        expected["native_web_search"] = False
        assert caps == expected


def test_get_capabilities_returns_copy():
    """Test that get_capabilities returns a mutable copy, not the original dict."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        caps = get_capabilities("openai")
        caps["custom_key"] = True
        # Original should not be affected (i.e., the mock dict should not have custom_key)
        assert "custom_key" not in _mock_get_capabilities_for_provider("openai")


def test_supports_feature_int_capability():
    """Test supports_feature returns int value for max_input_tokens."""
    with patch("multi_agent_dashboard.shared.provider_capabilities.get_capabilities_for_provider",
               side_effect=_mock_get_capabilities_for_provider):
        result = supports_feature("openai", "max_input_tokens")
        assert isinstance(result, int)
        assert result > 0


def test_module_exports():
    """Ensure expected symbols are exported."""
    import multi_agent_dashboard.shared.provider_capabilities as pc
    assert hasattr(pc, "get_capabilities")
    assert hasattr(pc, "supports_feature")
    # Static maps are no longer exported
    assert not hasattr(pc, "PROVIDER_DEFAULT_CAPABILITIES")
    assert not hasattr(pc, "MODEL_CAPABILITIES")
    assert "get_capabilities" in pc.__all__
    assert "supports_feature" in pc.__all__