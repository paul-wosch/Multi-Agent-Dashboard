"""
LiteLLM configuration and provider mapping for the Multi-Agent Dashboard.

This module provides a unified interface to LiteLLM, translating provider IDs
(openai, ollama, deepseek) into LiteLLM model strings and managing provider-specific
environment variables.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Provider ID to LiteLLM model prefix mapping
# These prefixes are used to construct LiteLLM model strings: {prefix}/{model_name}
PROVIDER_TO_LITELLM_PREFIX = {
    "openai": "openai",
    "ollama": "ollama",
    "deepseek": "deepseek",
}

# Default model names for each provider (used when only provider is specified)
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "ollama": "llama3",
    "deepseek": "deepseek-chat",
}

# Environment variable names for each provider's API key/base URL
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "ollama": "OLLAMA_HOST",  # Not an API key, but host URL
    "deepseek": "DEEPSEEK_API_KEY",
}


def get_litellm_model_string(provider_id: str, model_name: Optional[str] = None) -> str:
    """
    Convert a provider ID and optional model name to a LiteLLM model string.
    
    Examples:
        get_litellm_model_string("openai", "gpt-4o") -> "openai/gpt-4o"
        get_litellm_model_string("ollama") -> "ollama/llama3"
    
    Args:
        provider_id: Provider identifier (openai, ollama, deepseek)
        model_name: Specific model name (e.g., "gpt-4o", "llama3", "deepseek-reasoner")
    
    Returns:
        LiteLLM-compatible model string.
    """
    provider_id = provider_id.lower() if provider_id else ""
    if provider_id not in PROVIDER_TO_LITELLM_PREFIX:
        raise ValueError(f"Unknown provider ID: {provider_id}. Supported: {list(PROVIDER_TO_LITELLM_PREFIX.keys())}")
    
    prefix = PROVIDER_TO_LITELLM_PREFIX[provider_id]
    if not model_name:
        model_name = DEFAULT_MODELS[provider_id]
    
    return f"{prefix}/{model_name}"


def get_provider_config(provider_id: str) -> Dict[str, Any]:
    """
    Get provider-specific configuration for LiteLLM.
    
    Returns a dictionary with api_key, base_url, and other provider-specific
    settings extracted from environment variables and config module.
    
    Args:
        provider_id: Provider identifier (openai, ollama, deepseek)
    
    Returns:
        Dictionary of LiteLLM configuration parameters.
    """
    provider_id = provider_id.lower() if provider_id else ""
    config = {}
    
    # Get API key or host from environment or config module
    env_var = PROVIDER_ENV_VARS.get(provider_id)
    if env_var:
        value = None
        
        # Try config module first (reads from .env file)
        if provider_id == "openai":
            from multi_agent_dashboard import config as app_config
            value = getattr(app_config, "OPENAI_API_KEY", None)
        elif provider_id == "deepseek":
            from multi_agent_dashboard import config as app_config
            value = getattr(app_config, "DEEPSEEK_API_KEY", None)
        elif provider_id == "ollama":
            # OLLAMA_HOST is not in config module, read from os.environ
            value = os.getenv(env_var)
        
        if value:
            # For Ollama, the host goes into base_url
            if provider_id == "ollama":
                config["base_url"] = value
            else:
                config["api_key"] = value
        else:
            # Environment variable is missing or empty
            if provider_id == "ollama":
                logger.warning(f"OLLAMA_HOST not set; default endpoint is localhost:11434 (can be overridden per agent). Make sure Ollama is running.")
            elif provider_id in ["openai", "deepseek"]:
                logger.warning(f"{env_var} environment variable is not set. API calls will fail.")
    
    # Provider-specific defaults
    if provider_id == "ollama":
        # Ensure base_url has scheme if missing
        base_url = config.get("base_url")
        if base_url and "://" not in base_url:
            config["base_url"] = f"http://{base_url}"
        elif not base_url:
            # Set default base_url for Ollama if not provided
            config["base_url"] = "http://localhost:11434"
            logger.debug("Using default Ollama base_url: http://localhost:11434")
    
    if provider_id == "deepseek":
        # Set default base_url for DeepSeek if not provided
        base_url = config.get("base_url")
        if not base_url:
            # Use environment variable or default to official v1 endpoint
            base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
            config["base_url"] = base_url
            logger.debug("Using DeepSeek base_url: %s", base_url)
        elif "://" not in base_url:
            # Ensure scheme for custom base_url
            config["base_url"] = f"https://{base_url}"

    return config


def normalize_model_and_provider(
    model: str, 
    provider_id: Optional[str] = None
) -> Tuple[str, str]:
    """
    Normalize model and provider into a consistent (provider_id, model_name) tuple.
    
    Handles cases where:
    1. model is already a LiteLLM string (e.g., "openai/gpt-4o")
    2. model is a plain name and provider_id is given
    3. Only provider_id is given (use default model)
    
    Args:
        model: Model identifier (could be "gpt-4o" or "openai/gpt-4o")
        provider_id: Optional provider identifier
    
    Returns:
        Tuple of (provider_id, model_name) where model_name is without provider prefix.
    """
    if not model and not provider_id:
        raise ValueError("Either model or provider_id must be provided")
    
    # Case 1: model already contains provider prefix (e.g., "openai/gpt-4o")
    if "/" in model:
        parts = model.split("/", 1)
        detected_provider = parts[0]
        model_name = parts[1]
        
        # If provider_id was also given, ensure consistency
        if provider_id and provider_id != detected_provider:
            logger.warning(
                f"Provider mismatch: model suggests '{detected_provider}', "
                f"but provider_id is '{provider_id}'. Using '{detected_provider}'."
            )
        return detected_provider, model_name
    
    # Case 2: model is plain name, use provider_id or default to openai
    if not provider_id:
        provider_id = "openai"
    
    return provider_id, model


def get_litellm_completion_kwargs(
    provider_id: str,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    **extra_kwargs,
) -> Dict[str, Any]:
    """
    Build keyword arguments for litellm.completion() calls.
    
    Args:
        provider_id: Provider identifier
        endpoint: Custom endpoint URL (overrides environment-based base_url)
        api_key: Custom API key (overrides environment)
        base_url: Custom base URL (overrides environment)
        timeout: Request timeout in seconds
        **extra_kwargs: Additional parameters to pass to litellm.completion
    
    Returns:
        Dictionary of kwargs for litellm.completion.
    """
    kwargs = {}
    
    # Apply provider config from environment
    provider_config = get_provider_config(provider_id)
    
    # API key precedence: explicit > environment > none
    if api_key is not None:
        kwargs["api_key"] = api_key
    elif "api_key" in provider_config:
        kwargs["api_key"] = provider_config["api_key"]
    
    # Base URL precedence: endpoint > base_url > environment > none
    if endpoint:
        kwargs["base_url"] = endpoint
    elif base_url is not None:
        kwargs["base_url"] = base_url
    elif "base_url" in provider_config:
        kwargs["base_url"] = provider_config["base_url"]
    
    # Timeout
    if timeout is not None:
        kwargs["timeout"] = timeout
    
    # Merge extra kwargs
    kwargs.update(extra_kwargs)
    
    return kwargs


# LiteLLM feature support detection (can be extended)
SUPPORTED_FEATURES = {
    "openai": ["json_mode", "function_calling", "vision", "streaming", "tools"],
    "ollama": ["json_mode", "streaming"],  # Ollama may support function calling in some models
    "deepseek": ["json_mode", "function_calling", "streaming", "tools"],
}


def supports_feature(provider_id: str, feature: str) -> bool:
    """
    Check if a provider supports a specific feature.
    
    Note: This is a simplistic implementation. In production, you might want
    to query LiteLLM's capability detection or test dynamically.
    
    Args:
        provider_id: Provider identifier
        feature: Feature name (json_mode, function_calling, vision, streaming, tools)
    
    Returns:
        True if the feature is supported.
    """
    provider_id = provider_id.lower() if provider_id else ""
    
    # Try LiteLLM's dynamic detection first
    try:
        import litellm
        
        # Map feature names to LiteLLM detection methods
        if feature == "json_mode":
            # Check if provider supports response schema (JSON mode)
            if hasattr(litellm, "supports_response_schema"):
                if litellm.supports_response_schema(provider_id):
                    return True
                # If detection returns False, continue to fallback
        
        # For vision, check if vision is in supported OpenAI params
        if feature == "vision":
            if hasattr(litellm, "get_supported_openai_params"):
                params = litellm.get_supported_openai_params(provider_id)
                if params and "vision" in params:
                    return True
        
        # For function_calling and tools, check if tools param supported
        if feature in ("function_calling", "tools"):
            if hasattr(litellm, "get_supported_openai_params"):
                params = litellm.get_supported_openai_params(provider_id)
                if params and "tools" in params:
                    return True
        
        # For streaming, most providers support it; default True
        if feature == "streaming":
            return True
            
    except Exception as e:
        logger.debug(f"LiteLLM dynamic feature detection failed for {provider_id}.{feature}: {e}")
    
    # Fallback to static mapping
    return feature in SUPPORTED_FEATURES.get(provider_id, [])


# Export public interface
__all__ = [
    "get_litellm_model_string",
    "get_provider_config",
    "normalize_model_and_provider",
    "get_litellm_completion_kwargs",
    "supports_feature",
    "PROVIDER_TO_LITELLM_PREFIX",
    "DEFAULT_MODELS",
    "PROVIDER_ENV_VARS",
]