"""
LiteLLM Tool Adapter for Multi-Agent Dashboard.

Converts agent tool configurations to LiteLLM-compatible tool definitions,
dynamically selecting between Responses API (`tools`) and Completions API
(`web_search_options`) based on provider/model capabilities.
"""

import functools
import logging
import os
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try to import LiteLLM (optional)
try:
    import litellm
    # Drop unsupported parameters to avoid errors with providers like GPT‑5
    litellm.drop_params = True

    _LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    _LITELLM_AVAILABLE = False
from multi_agent_dashboard.litellm_config import register_model_with_litellm, normalize_model_and_provider


def convert_tools_for_litellm(
    tool_configs: Dict[str, Any],
    provider_id: str,
    model: str,
    use_responses_api: bool,
) -> Dict[str, Any]:
    """
    Convert agent tool configurations to LiteLLM-compatible tool definitions.
    
    Args:
        tool_configs: AgentSpec.tools dictionary with "enabled" and "tools" keys.
            Example: {"enabled": True, "tools": ["web_search"]}
        provider_id: Provider identifier (openai, ollama, deepseek).
        model: Model name (e.g., "gpt-4o", "llama3").
        use_responses_api: Whether to prefer the Responses API (True) or
            Completions API (False) for native web search.
    
    Returns:
        Dictionary with either "tools" (list) or "web_search_options" (dict)
        suitable for passing to LiteLLM completion calls. If no tools are
        applicable, returns an empty dict.
    
    Raises:
        ValueError: If tool_configs is malformed.
    
    Notes:
        - For `web_search`: uses native web search where supported.
        - For `web_search_ddg`: converts to a function‑calling tool with JSON Schema.
        - Unsupported tools are logged and excluded.
        - Caching is applied per (provider_id, model, use_responses_api, tool_configs_hash).
    """
    if not tool_configs.get("enabled", False):
        logger.debug("Tool conversion skipped: tool_configs.enabled=False")
        return {}
    
    enabled_tools = tool_configs.get("tools", [])
    if not enabled_tools:
        logger.debug("Tool conversion skipped: no tools enabled")
        return {}
    
    # Compute cache key (immutable representation of inputs)
    cache_key = _make_cache_key(tool_configs, provider_id, model, use_responses_api)
    return _convert_tools_for_litellm_cached(
        tuple(enabled_tools), provider_id, model, use_responses_api, cache_key
    )


@functools.lru_cache(maxsize=128)
def _convert_tools_for_litellm_cached(
    enabled_tools: Tuple[str],
    provider_id: str,
    model: str,
    use_responses_api: bool,
    cache_key: str,
) -> Dict[str, Any]:
    """
    Cached implementation of tool conversion.
    
    The cache key includes the hash of tool_configs (via cache_key) to avoid
    re‑computation for identical inputs. The LRU cache size is 128 entries.
    """
    # Determine which API(s) the model supports
    supports_tools_param = False
    supports_web_search_options = False
    native_web_search_supported = False
    
    if _LITELLM_AVAILABLE:
        full_model = f"{provider_id}/{model}"
        # Register model with LiteLLM to prevent spam messages
        _, model_name = normalize_model_and_provider(model or "", provider_id)
        register_model_with_litellm(provider_id, model_name)
        detection_succeeded = False
        try:
            # Get supported OpenAI parameters for this model
            supported_params = litellm.get_supported_openai_params(full_model)
            # Check native web search support
            native_web_search_supported = litellm.supports_web_search(full_model)
            detection_succeeded = True
        except Exception as e:
            logger.debug(f"LiteLLM capability detection failed for {provider_id}/{model}: {e}")
            # Fall back to static detection via supports_feature
            pass
        
        if detection_succeeded and supported_params is not None:
            # Only use dynamic detection if we got valid parameters
            supports_tools_param = "tools" in supported_params
            supports_web_search_options = "web_search_options" in supported_params
        else:
            # Dynamic detection failed or returned None, fall back to static detection
            detection_succeeded = False
    
    # If LiteLLM not available or detection failed, use static feature mapping (from litellm_config)
    if not _LITELLM_AVAILABLE or (_LITELLM_AVAILABLE and not detection_succeeded):
        # Import here to avoid circular imports
        from multi_agent_dashboard.litellm_config import supports_feature
        supports_tools_param = supports_feature(provider_id, "tools", model)
        # web_search_options is currently only for OpenAI Responses API
        supports_web_search_options = supports_tools_param and provider_id == "openai"
        native_web_search_supported = supports_web_search_options
    
    # Correct detection conflicts: if native web search is not supported, web_search_options is likely outdated
    logger.debug(f"Conflict resolution: supports_web_search_options={supports_web_search_options}, native_web_search_supported={native_web_search_supported}")
    if supports_web_search_options and not native_web_search_supported:
        logger.warning(f"Detection conflict for {provider_id}/{model}: web_search_options reported as supported but native web search unsupported. Treating web_search_options as unsupported.")
        supports_web_search_options = False
        logger.debug(f"After conflict resolution: supports_web_search_options={supports_web_search_options}")
    
    # Prefer tools over web_search_options when both are supported but native web search unsupported
    if supports_tools_param and supports_web_search_options and not native_web_search_supported:
        logger.info(f"Both tools and web_search_options reported for {provider_id}/{model}; preferring tools due to missing native web search support.")
        supports_web_search_options = False

    logger.info(
        f"Tool conversion for {provider_id}/{model}: "
        f"supports_tools_param={supports_tools_param}, "
        f"supports_web_search_options={supports_web_search_options}, "
        f"native_web_search_supported={native_web_search_supported}, "
        f"use_responses_api={use_responses_api}"
    )
    
    # Process each enabled tool
    tools_list: List[Dict[str, Any]] = []
    web_search_options: Optional[Dict[str, Any]] = None
    
    for tool_name in enabled_tools:
        if tool_name == "web_search":
            result = _convert_web_search_tool(
                provider_id,
                model,
                use_responses_api,
                supports_tools_param,
                supports_web_search_options,
                native_web_search_supported,
            )
            if result is None:
                logger.warning(f"Tool 'web_search' is not supported for {provider_id}/{model}; excluded.")
                continue
            
            if isinstance(result, dict) and "web_search_options" in result:
                # Completions API style
                if web_search_options is None:
                    web_search_options = result["web_search_options"]
                else:
                    # Merge? Only one web search tool is allowed; log warning.
                    logger.warning("Multiple web search configurations; using first.")
            else:
                # Responses API style (tools list)
                tools_list.extend(result.get("tools", []))
        
        elif tool_name == "web_search_ddg":
            result = _convert_web_search_ddg_tool(
                provider_id,
                model,
                supports_tools_param,
            )
            if result is None:
                logger.warning(f"Tool 'web_search_ddg' is not supported for {provider_id}/{model}; excluded.")
                continue
            tools_list.extend(result.get("tools", []))
        
        else:
            logger.warning(f"Unknown tool '{tool_name}'; excluded.")
            continue
    
    # Decide final output format
    if web_search_options is not None:
        # Prefer web_search_options when available (Completions API)
        logger.info(f"Using web_search_options: {web_search_options}")
        return {"web_search_options": web_search_options}
    elif tools_list:
        logger.info(f"Using tools list with {len(tools_list)} tool(s)")
        return {"tools": tools_list}
    else:
        logger.debug("No tools applicable after conversion")
        return {}


def _convert_web_search_tool(
    provider_id: str,
    model: str,
    use_responses_api: bool,
    supports_tools_param: bool,
    supports_web_search_options: bool,
    native_web_search_supported: bool,
) -> Optional[Dict[str, Any]]:
    """
    Convert a native web_search tool to LiteLLM format.
    
    Returns either {"web_search_options": {...}} for Completions API or
    {"tools": [{"type": "web_search", ...}]} for Responses API.
    Returns None if web search is not supported.
    """
    # If native web search is not supported, web search tool cannot be used
    if not native_web_search_supported:
        logger.debug(f"Native web search not supported for {provider_id}/{model}; web_search tool excluded.")
        return None

    # Determine which API to use based on capabilities and flag
    use_completions_api = False
    use_responses_api_flag = False
    
    if supports_web_search_options and not use_responses_api:
        # Model supports web_search_options and caller prefers Completions API
        use_completions_api = True
    elif supports_tools_param and use_responses_api:
        # Model supports tools param and caller prefers Responses API
        use_responses_api_flag = True
    elif native_web_search_supported:
        # Fallback: if native web search is supported, use whichever API is available
        if supports_web_search_options:
            use_completions_api = True
        elif supports_tools_param:
            use_responses_api_flag = True
    
    if not (use_completions_api or use_responses_api_flag):
        # No supported API found
        return None
    
    # Build filters (allowed_domains) from tool_configs if available
    # Currently, domain filtering is handled by the caller via state.
    # We'll pass an empty dict; the caller can add filters later.
    # For now, we just return the basic tool structure.
    
    if use_completions_api:
        # Completions API: web_search_options dict
        web_search_options: Dict[str, Any] = {}
        # Note: allowed_domains can be added later by the caller
        return {"web_search_options": web_search_options}
    else:
        # Responses API: tools list with web_search type
        tool_obj: Dict[str, Any] = {"type": "web_search"}
        # filters can be added later
        return {"tools": [tool_obj]}


def _convert_web_search_ddg_tool(
    provider_id: str,
    model: str,
    supports_tools_param: bool,
) -> Optional[Dict[str, Any]]:
    """
    Convert a DuckDuckGo web search tool to a function‑calling tool.
    
    Returns {"tools": [function_tool_dict]} or None if tool calling unsupported.
    """
    if not supports_tools_param:
        logger.debug(f"Tool calling not supported for {provider_id}/{model}; skipping web_search_ddg.")
        return None
    
    # JSON Schema for the DuckDuckGo search function
    function_schema: Dict[str, Any] = {
        "name": "duckduckgo_search",
        "description": "Search the web using DuckDuckGo. Returns relevant web pages with snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "domain_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of domains to restrict results to",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
    
    # Build the function tool dict (OpenAI tool format)
    tool_obj = {
        "type": "function",
        "function": function_schema,
    }
    
    return {"tools": [tool_obj]}


def _make_cache_key(
    tool_configs: Dict[str, Any],
    provider_id: str,
    model: str,
    use_responses_api: bool,
) -> str:
    """
    Create a deterministic cache key for the given inputs.
    
    The key is a string that uniquely identifies the combination of
    tool_configs (sorted), provider_id, model, and use_responses_api.
    """
    import json
    # Normalize tool_configs to a sorted JSON string
    normalized = {
        "enabled": tool_configs.get("enabled", False),
        "tools": sorted(tool_configs.get("tools", [])),
    }
    # Include any extra keys that might affect conversion (e.g., filters)
    # For now, we ignore filters because they are added later via state.
    config_str = json.dumps(normalized, sort_keys=True)
    return f"{provider_id}:{model}:{use_responses_api}:{config_str}"


# Export public interface
__all__ = ["convert_tools_for_litellm"]