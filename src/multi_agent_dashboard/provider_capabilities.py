"""
Static advisory capability mapping for LLM providers.

This module provides purely advisory capability information for known providers and models.
It is used only for generating warnings and setting UI defaults, NOT for runtime decisions.
The primary source of truth for capabilities is the agent configuration (`provider_features`).
"""

import logging
from typing import Dict, Any, Optional, Set

logger = logging.getLogger(__name__)

# ============================================================================
# Static Advisory Capability Maps
# ============================================================================

# Provider-level default capabilities (when model not specified)
PROVIDER_DEFAULT_CAPABILITIES: Dict[str, Dict[str, bool]] = {
    "openai": {
        "structured_output": True,
        "tool_calling": True,
        "reasoning": False,          # Only specific models (e.g., o1) support reasoning
        "image_inputs": True,        # Vision-capable models (gpt-4o, gpt-4-turbo)
        "max_input_tokens": 128000,  # Default for gpt-4o
        "streaming": True,
        "json_mode": True,
        "function_calling": True,
        "vision": True,
        "tools": True,
        "native_web_search": True,
    },
    "deepseek": {
        "structured_output": True,
        "tool_calling": True,
        "reasoning": True,           # DeepSeek-Reasoner models
        "image_inputs": False,       # Text-only API as of 2026-02
        "max_input_tokens": 128000,  # DeepSeek chat context window
        "streaming": True,
        "json_mode": True,
        "function_calling": True,
        "vision": False,
        "tools": True,
        "native_web_search": False,
    },
    "ollama": {
        "structured_output": True,
        "tool_calling": False,       # Some models may support; treat as advisory only
        "reasoning": False,
        "image_inputs": False,       # Vision-capable models (llava, bakllava) override
        "max_input_tokens": 8192,    # Typical context window for Llama models
        "streaming": True,
        "json_mode": True,
        "function_calling": False,
        "vision": False,
        "tools": False,
        "native_web_search": False,
    },
}

# Model-specific capability overrides (additive, not exhaustive)
MODEL_CAPABILITIES: Dict[str, Dict[str, Dict[str, bool]]] = {
    "openai": {
        "gpt-4o": {
            "vision": True,
            "image_inputs": True,
            "max_input_tokens": 128000,
        },
        "gpt-4o-mini": {
            "vision": True,
            "image_inputs": True,
            "max_input_tokens": 128000,
        },
        "gpt-4-turbo": {
            "vision": True,
            "image_inputs": True,
            "max_input_tokens": 128000,
        },
        "gpt-4": {
            "vision": False,         # Original GPT-4 lacks native vision
            "image_inputs": False,
            "max_input_tokens": 8192,
        },
        "gpt-3.5-turbo": {
            "vision": False,
            "image_inputs": False,
            "max_input_tokens": 16385,
        },
        "o1-preview": {
            "reasoning": True,
            "tool_calling": False,   # o1 does not support tool calling
            "structured_output": False,  # o1 does not support JSON Schema
            "max_input_tokens": 128000,
            "native_web_search": False,
        },
        "o1-mini": {
            "reasoning": True,
            "tool_calling": False,
            "structured_output": False,
            "max_input_tokens": 128000,
            "native_web_search": False,
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
            "vision": True,
            "image_inputs": True,
        },
        "bakllava": {
            "vision": True,
            "image_inputs": True,
        },
        "llama3": {
            "tool_calling": True,    # Some Llama 3 models support tool calling
            "function_calling": True,
            "tools": True,
        },
        "llama3.2": {
            "tool_calling": True,
            "function_calling": True,
            "tools": True,
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

# ============================================================================
# Advisory Functions
# ============================================================================

def get_capabilities(
    provider_id: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get advisory capabilities for a provider (and optionally a specific model).

    This function returns a dictionary of capability hints that can be used for
    warnings and UI defaults. It does NOT override user configuration.

    Args:
        provider_id: Provider identifier ("openai", "deepseek", "ollama")
        model: Optional model name (e.g., "gpt-4o", "llava"). If provided,
               model-specific overrides are applied.

    Returns:
        Dictionary with capability keys and boolean/int values. If provider is
        unknown, returns an empty dict.
    """
    provider_id = provider_id.lower() if provider_id else ""
    if provider_id not in PROVIDER_DEFAULT_CAPABILITIES:
        logger.debug(f"Unknown provider '{provider_id}' in capability lookup")
        return {}

    # Start with provider defaults
    caps = PROVIDER_DEFAULT_CAPABILITIES[provider_id].copy()

    # Apply model-specific overrides if model given and known
    if model:
        model = model.lower()
        model_overrides = MODEL_CAPABILITIES.get(provider_id, {}).get(model)
        if model_overrides:
            caps.update(model_overrides)
        else:
            # Log that model is not in our advisory map (not an error)
            logger.debug(
                f"No advisory capability overrides for model '{model}' "
                f"under provider '{provider_id}'"
            )

    return caps


def supports_feature(
    provider_id: str,
    feature: str,
    model: Optional[str] = None
) -> bool:
    """
    Check if a provider (and optionally a specific model) is likely to support a feature.

    This is an advisory check only, used for logging warnings and setting UI defaults.
    It does NOT restrict what features can be enabled by the user.

    Args:
        provider_id: Provider identifier ("openai", "deepseek", "ollama")
        feature: Feature name (e.g., "structured_output", "tool_calling", "vision")
        model: Optional model name (e.g., "gpt-4o", "llava").

    Returns:
        True if the advisory capability mapping indicates support; False otherwise.
        Unknown providers return False.
    """
    caps = get_capabilities(provider_id, model)
    return caps.get(feature, False)


# ============================================================================
# Module Export
# ============================================================================

__all__ = [
    "get_capabilities",
    "supports_feature",
    "PROVIDER_DEFAULT_CAPABILITIES",
    "MODEL_CAPABILITIES",
]