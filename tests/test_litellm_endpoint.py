"""
Test endpoint propagation and normalization in LiteLLM path.
"""
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multi_agent_dashboard.llm_client import _init_chat_model_with_litellm


def test_endpoint_from_kwargs():
    """Test that endpoint passed via kwargs is used as base_url."""
    with patch("multi_agent_dashboard.litellm_config.get_litellm_model_string") as mock_get_model:
        with patch("multi_agent_dashboard.litellm_config.get_provider_config") as mock_get_config:
            with patch("multi_agent_dashboard.llm_client._ChatLiteLLM") as mock_chat:
                # Setup mocks
                mock_get_model.return_value = "ollama/llama3"
                mock_get_config.return_value = {"base_url": None, "api_key": None}
                mock_chat.return_value = MagicMock()

                # Call with endpoint in kwargs
                _init_chat_model_with_litellm(
                    model="llama3",
                    model_provider="ollama",
                    base_url="http://192.168.178.103:11434"
                )

                # Verify ChatLiteLLM instantiated with correct base_url
                call_kwargs = mock_chat.call_args[1]
                assert "base_url" in call_kwargs
                assert call_kwargs["base_url"] == "http://192.168.178.103:11434"
                # Ensure provider config base_url not used
                mock_get_config.assert_called_once_with("ollama")


def test_endpoint_normalization_adds_scheme():
    """Test that endpoint without scheme gets http:// added."""
    with patch("multi_agent_dashboard.litellm_config.get_litellm_model_string") as mock_get_model:
        with patch("multi_agent_dashboard.litellm_config.get_provider_config") as mock_get_config:
            with patch("multi_agent_dashboard.llm_client._ChatLiteLLM") as mock_chat:
                # Setup mocks
                mock_get_model.return_value = "ollama/llama3"
                mock_get_config.return_value = {"base_url": None, "api_key": None}
                mock_chat.return_value = MagicMock()

                # Call with endpoint missing scheme
                _init_chat_model_with_litellm(
                    model="llama3",
                    model_provider="ollama",
                    base_url="192.168.178.103:11434"
                )

                call_kwargs = mock_chat.call_args[1]
                assert "base_url" in call_kwargs
                assert call_kwargs["base_url"] == "http://192.168.178.103:11434"


def test_endpoint_precedence_kwargs_over_config():
    """Test that kwargs base_url takes precedence over provider_config base_url."""
    with patch("multi_agent_dashboard.litellm_config.get_litellm_model_string") as mock_get_model:
        with patch("multi_agent_dashboard.litellm_config.get_provider_config") as mock_get_config:
            with patch("multi_agent_dashboard.llm_client._ChatLiteLLM") as mock_chat:
                # Setup mocks: provider config has a base_url
                mock_get_model.return_value = "ollama/llama3"
                mock_get_config.return_value = {"base_url": "http://localhost:11434", "api_key": None}
                mock_chat.return_value = MagicMock()

                # Call with different endpoint in kwargs
                _init_chat_model_with_litellm(
                    model="llama3",
                    model_provider="ollama",
                    base_url="http://custom:11434"
                )

                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["base_url"] == "http://custom:11434"  # kwargs wins


def test_endpoint_falls_back_to_config():
    """Test that if kwargs has no base_url, provider_config base_url is used."""
    with patch("multi_agent_dashboard.litellm_config.get_litellm_model_string") as mock_get_model:
        with patch("multi_agent_dashboard.litellm_config.get_provider_config") as mock_get_config:
            with patch("multi_agent_dashboard.llm_client._ChatLiteLLM") as mock_chat:
                # Setup mocks
                mock_get_model.return_value = "ollama/llama3"
                mock_get_config.return_value = {"base_url": "http://localhost:11434", "api_key": None}
                mock_chat.return_value = MagicMock()

                # Call without base_url in kwargs
                _init_chat_model_with_litellm(
                    model="llama3",
                    model_provider="ollama"
                )

                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["base_url"] == "http://localhost:11434"


def test_endpoint_preserves_https_scheme():
    """Test that endpoint with https:// is not changed."""
    with patch("multi_agent_dashboard.litellm_config.get_litellm_model_string") as mock_get_model:
        with patch("multi_agent_dashboard.litellm_config.get_provider_config") as mock_get_config:
            with patch("multi_agent_dashboard.llm_client._ChatLiteLLM") as mock_chat:
                # Setup mocks
                mock_get_model.return_value = "openai/gpt-4"
                mock_get_config.return_value = {"base_url": None, "api_key": None}
                mock_chat.return_value = MagicMock()

                _init_chat_model_with_litellm(
                    model="gpt-4",
                    model_provider="openai",
                    base_url="https://api.openai.com/v1"
                )

                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["base_url"] == "https://api.openai.com/v1"


if __name__ == "__main__":
    test_endpoint_from_kwargs()
    print("✓ test_endpoint_from_kwargs passed")
    test_endpoint_normalization_adds_scheme()
    print("✓ test_endpoint_normalization_adds_scheme passed")
    test_endpoint_precedence_kwargs_over_config()
    print("✓ test_endpoint_precedence_kwargs_over_config passed")
    test_endpoint_falls_back_to_config()
    print("✓ test_endpoint_falls_back_to_config passed")
    test_endpoint_preserves_https_scheme()
    print("✓ test_endpoint_preserves_https_scheme passed")
    print("All endpoint propagation tests passed!")