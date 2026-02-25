"""
Static advisory capability mapping for LLM providers.

This module provides purely advisory capability information for known providers and models.
It is used only for generating warnings and setting UI defaults, NOT for runtime decisions.
The primary source of truth for capabilities is the agent configuration.
"""

import logging
from typing import Dict, Any, Optional, Set
from multi_agent_dashboard.provider_data.loader import get_capabilities_for_provider

logger = logging.getLogger(__name__)


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
    caps = get_capabilities_for_provider(provider_id, model)
    
    if not caps:
        logger.warning(
            f"No capability data for provider '{provider_id}'"
            + (f" model '{model}'" if model else "")
        )
    
    # Add derived keys for backward compatibility
    # vision maps to image_inputs
    if "image_inputs" in caps:
        caps["vision"] = caps["image_inputs"]
    # tools maps to tool_calling
    if "tool_calling" in caps:
        caps["tools"] = caps["tool_calling"]
        caps["function_calling"] = caps["tool_calling"]
    # streaming assumed True for all known providers (advisory only)
    caps["streaming"] = True
    # json_mode maps to structured_output
    if "structured_output" in caps:
        caps["json_mode"] = caps["structured_output"]
    # native_web_search defaults to False (advisory only)
    caps["native_web_search"] = False
    
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
]