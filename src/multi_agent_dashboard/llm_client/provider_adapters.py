"""
Provider-specific adapters for structured output binding.

Each adapter encapsulates provider-specific logic for:
- Structured output method selection (json_schema, function_calling, json_mode)
- Schema extraction from provider-specific response_format wrappers
- Schema wrapping into provider-specific response_format payloads
"""

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ProviderAdapter:
    """
    Abstract interface for provider-specific structured output handling.
    """

    def get_structured_output_method(self, model_name: str) -> str:
        """
        Return the appropriate method for with_structured_output for this provider.
        """
        raise NotImplementedError

    def extract_schema(self, response_format: Any, spec: Any) -> Tuple[Any, Optional[str]]:
        """
        Extract schema dict and optional schema name from provider-specific response_format.
        Returns (schema, schema_name).
        """
        raise NotImplementedError

    def wrap_schema(self, schema: Dict[str, Any], schema_name: Optional[str]) -> Any:
        """
        Wrap a plain JSON schema dict into provider-specific response_format payload.
        """
        raise NotImplementedError


class OpenAIAdapter(ProviderAdapter):
    """
    Adapter for OpenAI's JSON Schema structured output format.
    """

    def get_structured_output_method(self, model_name: str) -> str:
        return "json_schema"

    def extract_schema(self, response_format: Any, spec: Any) -> Tuple[Any, Optional[str]]:
        schema = None
        schema_name = None
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema_obj = response_format.get("json_schema", {})
            schema = json_schema_obj.get("schema")
            schema_name = json_schema_obj.get("name")
        else:
            schema = response_format
        return schema, schema_name

    def wrap_schema(self, schema: Dict[str, Any], schema_name: Optional[str]) -> Any:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name or "schema",
                "schema": schema,
            },
        }


class DeepSeekAdapter(ProviderAdapter):
    """
    Adapter for DeepSeek's function-calling / json_mode structured output.
    """

    def get_structured_output_method(self, model_name: str) -> str:
        # Reasoner models often reject tool_choice; prefer json_mode
        if "reasoner" in model_name.lower():
            return "json_mode"
        return "function_calling"

    def extract_schema(self, response_format: Any, spec: Any) -> Tuple[Any, Optional[str]]:
        schema = None
        schema_name = None
        if isinstance(response_format, dict):
            if "parameters" in response_format:
                schema = response_format["parameters"]
                schema_name = response_format.get("name")
            else:
                schema = response_format
        else:
            schema = response_format
        return schema, schema_name

    def wrap_schema(self, schema: Dict[str, Any], schema_name: Optional[str]) -> Any:
        # DeepSeek uses function-calling; wrap plain JSON schema as a tool definition.
        if isinstance(schema, dict) and "parameters" not in schema:
            return {
                "name": schema_name or "schema",
                "description": "Structured output schema",
                "parameters": schema,
            }
        return schema


class OllamaAdapter(ProviderAdapter):
    """
    Adapter for Ollama's raw JSON schema format.
    """

    def get_structured_output_method(self, model_name: str) -> str:
        return "json_schema"

    def extract_schema(self, response_format: Any, spec: Any) -> Tuple[Any, Optional[str]]:
        # Ollama expects plain schema dict
        return response_format, None

    def wrap_schema(self, schema: Dict[str, Any], schema_name: Optional[str]) -> Any:
        # Ollama uses raw schema dict, no wrapper
        return schema


class DefaultAdapter(ProviderAdapter):
    """
    Fallback adapter for unknown providers (structured output disabled).
    """

    def get_structured_output_method(self, model_name: str) -> str:
        return "json_schema"

    def extract_schema(self, response_format: Any, spec: Any) -> Tuple[Any, Optional[str]]:
        return response_format, None

    def wrap_schema(self, schema: Dict[str, Any], schema_name: Optional[str]) -> Any:
        # Unknown provider: structured output disabled (return None)
        return None


def get_adapter(provider_id: str) -> ProviderAdapter:
    """
    Factory function returning appropriate adapter for given provider_id.
    """
    provider_id = provider_id.lower()
    if provider_id == "openai":
        return OpenAIAdapter()
    elif provider_id == "deepseek":
        return DeepSeekAdapter()
    elif provider_id == "ollama":
        return OllamaAdapter()
    else:
        return DefaultAdapter()