"""
Test LiteLLMClient token counting and usage extraction.
"""
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.multi_agent_dashboard.llm_client import LiteLLMClient, TextResponse


def test_litellm_client_usage_extraction():
    """Test that LiteLLMClient extracts usage correctly from mock response."""
    # Mock litellm availability
    with patch("src.multi_agent_dashboard.llm_client._LITELLM_AVAILABLE", True):
        with patch("src.multi_agent_dashboard.llm_client._litellm") as mock_litellm:
            with patch("src.multi_agent_dashboard.llm_client._litellm_completion") as mock_completion:
                # Create a simple mock object that mimics a LiteLLM response
                # with model_dump method (like Pydantic model)
                class MockResponse:
                    def __init__(self):
                        self.choices = [MockChoice()]
                        self.usage = {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        }
                    
                    def model_dump(self):
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": "Hello, world",
                                        "role": "assistant"
                                    }
                                }
                            ],
                            "usage": self.usage,
                        }
                
                class MockChoice:
                    def __init__(self):
                        self.message = MockMessage()
                
                class MockMessage:
                    content = "Hello, world"
                    role = "assistant"
                
                mock_response = MockResponse()
                mock_completion.return_value = mock_response
                
                client = LiteLLMClient()
                # Call invoke with minimal arguments
                response = client.invoke(
                    model="gpt-4o",
                    provider_id="openai",
                    messages=[{"role": "user", "content": "Hello"}],
                )
                
                # Verify token counts
                assert response.input_tokens == 10
                assert response.output_tokens == 20
                # Verify raw dict contains usage_metadata
                assert "usage_metadata" in response.raw
                assert response.raw["usage_metadata"]["prompt_tokens"] == 10
                assert response.raw["usage_metadata"]["completion_tokens"] == 20
                # Verify raw dict also contains usage (if present)
                assert "usage" in response.raw or "usage" in response.raw.get("usage_metadata", {})
                
                # Verify extra_body metadata was passed
                call_kwargs = mock_completion.call_args[1]
                assert "extra_body" in call_kwargs
                assert call_kwargs["extra_body"] == {"metadata": True}


def test_litellm_client_usage_extraction_dict_usage():
    """Test usage extraction when usage is a dict."""
    with patch("src.multi_agent_dashboard.llm_client._LITELLM_AVAILABLE", True):
        with patch("src.multi_agent_dashboard.llm_client._litellm_completion") as mock_completion:
            mock_response = {
                "choices": [{"message": {"content": "Test", "role": "assistant"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            }
            mock_completion.return_value = mock_response
            
            client = LiteLLMClient()
            response = client.invoke(
                model="ollama/llama3",
                messages=[{"role": "user", "content": "Hi"}],
            )
            
            assert response.input_tokens == 5
            assert response.output_tokens == 7
            assert "usage_metadata" in response.raw
            assert response.raw["usage_metadata"]["prompt_tokens"] == 5


def test_litellm_client_no_usage():
    """Test when response has no usage data."""
    with patch("src.multi_agent_dashboard.llm_client._LITELLM_AVAILABLE", True):
        with patch("src.multi_agent_dashboard.llm_client._litellm_completion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "No usage"
            mock_response.usage = None
            
            mock_completion.return_value = mock_response
            
            client = LiteLLMClient()
            response = client.invoke(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": "Test"}],
            )
            
            assert response.input_tokens is None
            assert response.output_tokens is None
            # usage_metadata should not be present
            assert "usage_metadata" not in response.raw


if __name__ == "__main__":
    test_litellm_client_usage_extraction()
    print("✓ test_litellm_client_usage_extraction passed")
    
    test_litellm_client_usage_extraction_dict_usage()
    print("✓ test_litellm_client_usage_extraction_dict_usage passed")
    
    test_litellm_client_no_usage()
    print("✓ test_litellm_client_no_usage passed")
    
    print("All tests passed!")