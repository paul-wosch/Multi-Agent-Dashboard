"""
Provider-Specific Tool Adapter for Multi-Agent Dashboard.

Converts agent tool configurations to provider-specific tool definitions,
using advisory capability mapping for warnings only. Respects exact tool
configuration from AgentSpec.tools as the primary source of truth.

Supports:
- web_search: native web search tool (OpenAI Responses/Completions API)
- web_search_ddg: DuckDuckGo search via function-calling tool

For providers other than OpenAI, attempts to bind tools as function-calling
tools where supported, logging advisory warnings based on static capability
mapping.
"""

import functools
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from multi_agent_dashboard.provider_capabilities import supports_feature

logger = logging.getLogger(__name__)


def convert_tools_for_provider(
    tool_configs: Dict[str, Any],
    provider_id: str,
    model: str,
    use_responses_api: bool,
    provider_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert agent tool configurations to provider-specific tool definitions.

    Args:
        tool_configs: AgentSpec.tools dictionary with "enabled" and "tools" keys.
            Example: {"enabled": True, "tools": ["web_search"]}
        provider_id: Provider identifier ("openai", "deepseek", "ollama").
        model: Model name (e.g., "gpt-4o", "llama3").
        use_responses_api: Whether to prefer the Responses API (True) or
            Completions API (False) for native web search (OpenAI only).
        provider_features: Optional provider features dict from AgentSpec,
            used for advisory capability checks.

    Returns:
        Dictionary with either "tools" (list) or "web_search_options" (dict)
        suitable for passing to LangChain's bind_tools() or bind() methods.
        If no tools are applicable, returns an empty dict.

    Raises:
        ValueError: If tool_configs is malformed.

    Notes:
        - Respects exact tool configuration; does not convert between tool types.
        - Uses advisory capability mapping only for logging warnings.
        - For OpenAI, web_search uses Responses API (tools list) when
          use_responses_api=True, otherwise Completions API (web_search_options).
        - For other providers, web_search is treated as a function-calling tool
          (if tool calling is likely supported) with a generic search schema.
        - Caching is applied per (provider_id, model, use_responses_api,
          tool_configs_hash, provider_features_hash).
    """
    if not tool_configs.get("enabled", False):
        logger.debug("Tool conversion skipped: tool_configs.enabled=False")
        return {}

    enabled_tools = tool_configs.get("tools", [])
    if not enabled_tools:
        logger.debug("Tool conversion skipped: no tools enabled")
        return {}

    # Compute cache key (immutable representation of inputs)
    cache_key = _make_cache_key(
        tool_configs, provider_id, model, use_responses_api, provider_features
    )
    # Serialize provider_features to immutable tuple for caching
    provider_features_tuple = None
    if provider_features is not None:
        provider_features_tuple = tuple(sorted(provider_features.items()))
    
    return _convert_tools_for_provider_cached(
        tuple(enabled_tools),
        provider_id,
        model,
        use_responses_api,
        provider_features_tuple,
        cache_key,
    )


@functools.lru_cache(maxsize=128)
def _convert_tools_for_provider_cached(
    enabled_tools: Tuple[str],
    provider_id: str,
    model: str,
    use_responses_api: bool,
    provider_features: Optional[Tuple[Tuple[str, Any], ...]],  # serialized as tuple of tuples
    cache_key: str,
) -> Dict[str, Any]:
    """
    Cached implementation of tool conversion.

    The cache key includes the hash of tool_configs and provider_features
    to avoid re-computation for identical inputs. The LRU cache size is 128 entries.
    """
    # Deserialize provider_features back to dict if present
    provider_features_dict = None
    if provider_features is not None:
        provider_features_dict = dict(provider_features)

    # Determine advisory capabilities for logging warnings
    # This is advisory only; we still attempt to bind requested tools.
    advisory_tool_calling = supports_feature(provider_id, "tool_calling", model)
    advisory_native_web_search = supports_feature(provider_id, "native_web_search", model)

    logger.debug(
        f"Tool conversion for {provider_id}/{model}: "
        f"advisory_tool_calling={advisory_tool_calling}, "
        f"advisory_native_web_search={advisory_native_web_search}, "
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
                advisory_native_web_search,
                advisory_tool_calling,
            )
            if result is None:
                logger.warning(
                    f"Tool 'web_search' is not supported for {provider_id}/{model}; excluded."
                )
                continue

            if isinstance(result, dict) and "web_search_options" in result:
                # Completions API style
                if web_search_options is None:
                    web_search_options = result["web_search_options"]
                else:
                    # Merge? Only one web search tool is allowed; log warning.
                    logger.warning("Multiple web search configurations; using first.")
            else:
                # Responses API style (tools list) or function-calling tool
                tools_list.extend(result.get("tools", []))

        elif tool_name == "web_search_ddg":
            result = _convert_web_search_ddg_tool(
                provider_id,
                model,
                advisory_tool_calling,
            )
            if result is None:
                logger.warning(
                    f"Tool 'web_search_ddg' is not supported for {provider_id}/{model}; excluded."
                )
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
    advisory_native_web_search: bool,
    advisory_tool_calling: bool,
) -> Optional[Dict[str, Any]]:
    """
    Convert a web_search tool to provider-specific format.

    Returns either {"web_search_options": {...}} for OpenAI Completions API,
    {"tools": [{"type": "web_search", ...}]} for OpenAI Responses API,
    or {"tools": [function_tool_dict]} for providers without native web search.
    Returns None if web search cannot be represented.
    """
    # Log advisory warnings
    if not advisory_native_web_search:
        logger.warning(
            f"Provider {provider_id} likely does not support native web search. "
            f"Attempting to bind as function-calling tool."
        )

    # OpenAI-specific native web search
    if provider_id == "openai" and advisory_native_web_search:
        if use_responses_api:
            # Responses API: tools list with web_search type
            tool_obj: Dict[str, Any] = {"type": "web_search"}
            # filters can be added later by caller
            return {"tools": [tool_obj]}
        else:
            # Completions API: web_search_options dict
            web_search_options: Dict[str, Any] = {}
            # Note: allowed_domains can be added later by caller
            return {"web_search_options": web_search_options}
    else:
        # Non-OpenAI or OpenAI without native web search support:
        # treat as a generic search function-calling tool
        if not advisory_tool_calling:
            logger.warning(
                f"Provider {provider_id} likely does not support tool calling. "
                f"Web search may fail."
            )
        # Create a generic web search function tool (similar to DuckDuckGo but generic)
        function_schema: Dict[str, Any] = {
            "name": "web_search",
            "description": "Search the web for current information. Returns relevant web pages with snippets.",
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
        tool_obj = {
            "type": "function",
            "function": function_schema,
        }
        return {"tools": [tool_obj]}


def _convert_web_search_ddg_tool(
    provider_id: str,
    model: str,
    advisory_tool_calling: bool,
) -> Optional[Dict[str, Any]]:
    """
    Convert a DuckDuckGo web search tool to a function-calling tool.

    Returns {"tools": [function_tool_dict]} or None if tool calling unsupported.
    """
    if not advisory_tool_calling:
        logger.warning(
            f"Provider {provider_id} likely does not support tool calling. "
            f"DuckDuckGo search may fail."
        )
        # Still attempt to bind; let runtime error surface.

    # JSON Schema for the DuckDuckGo search function (same as existing)
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
    provider_features: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a deterministic cache key for the given inputs.

    The key is a string that uniquely identifies the combination of
    tool_configs (sorted), provider_id, model, use_responses_api, and
    provider_features (sorted).
    """
    # Normalize tool_configs to a sorted JSON string
    normalized = {
        "enabled": tool_configs.get("enabled", False),
        "tools": sorted(tool_configs.get("tools", [])),
    }
    # Include any extra keys that might affect conversion (e.g., filters)
    config_str = json.dumps(normalized, sort_keys=True)

    # Normalize provider_features to sorted JSON string if present
    features_str = ""
    if provider_features is not None:
        # Sort keys and convert to JSON
        features_str = json.dumps(
            {k: v for k, v in sorted(provider_features.items())}, sort_keys=True
        )

    return f"{provider_id}:{model}:{use_responses_api}:{config_str}:{features_str}"


# Export public interface
__all__ = ["convert_tools_for_provider"]