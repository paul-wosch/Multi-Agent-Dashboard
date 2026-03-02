"""
Structured output binding and schema management for LLM providers.

This module provides the core logic for binding JSON schemas to LLM providers
for structured output generation. It handles provider-specific schema extraction,
response format wrapping, and structured output method selection.

Key functionality:
- Provider-specific schema extraction from response_format wrappers
- Structured output method selection (JSON Schema, function calling, JSON mode)
- Schema validation and transformation for provider compatibility
- Integration with provider adapters for consistent behavior

The module enables reliable structured output generation across different
LLM providers while maintaining schema integrity and validation.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from .provider_adapters import get_adapter
from .wrappers import StructuredOutputWrapper

logger = logging.getLogger(__name__)


class StructuredOutputBinder:
    """
    Handles provider-specific schema extraction and structured output binding.
    """

    def __init__(self, client):
        self._client = client

    def extract_schema(self, response_format, provider_id, spec):
        """
        Extract schema from provider-specific response_format wrapper.
        Returns (schema, schema_name).
        """
        adapter = get_adapter(provider_id)
        schema, schema_name = adapter.extract_schema(response_format, spec)
        
        if schema is None:
            schema = response_format  # ultimate fallback

        # Ensure the schema dict has title and description for LangChain's with_structured_output
        if isinstance(schema, dict):
            # Create a copy to avoid modifying the original
            schema = schema.copy()
            if "title" not in schema:
                schema["title"] = schema_name or getattr(spec, "schema_name", None) or "schema"
            if "description" not in schema:
                schema["description"] = "Structured output schema"

        return schema, schema_name

    def bind_structured_output(self, spec, model_instance, response_format, provider_id, model, tools=None, strict=True):
        """
        Bind structured output to model instance, optionally with tools (unified binding).
        Returns (model_instance, effective_response_format).
        """
        try:
            adapter = get_adapter(provider_id)
            structured_output_method = adapter.get_structured_output_method(model)
            schema, schema_name = self.extract_schema(response_format, provider_id, spec)
            
            # Apply binding
            if tools is not None:
                # Unified binding with tools
                unified_model = model_instance.with_structured_output(
                    schema,
                    method=structured_output_method,
                    include_raw=True,
                    tools=tools,
                )
                model_instance = StructuredOutputWrapper.wrap(unified_model)
                effective_response_format = None
                logger.info(f"Applied unified tools+structured_output binding for {provider_id}")
            else:
                # Structured output only
                structured_model = model_instance.with_structured_output(
                    schema,
                    method=structured_output_method,
                    include_raw=True,
                    strict=strict,
                )
                model_instance = StructuredOutputWrapper.wrap(structured_model)
                effective_response_format = None
                logger.info(f"Applied provider-specific structured output binding for {provider_id}")
            return model_instance, effective_response_format
        except Exception as e:
            logger.warning(f"Structured output binding failed: {e}")
            # Return unchanged model and original response_format
            return model_instance, response_format