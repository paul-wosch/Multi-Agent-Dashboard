# Tool configuration logic for agent runtime
from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional, Union
from multi_agent_dashboard.tool_integration.provider_tool_adapter import convert_tools_for_provider

logger = logging.getLogger(__name__)


def get_allowed_domains(spec: Any, state: Dict[str, Any]) -> Optional[List[str]]:
    """
    Extract allowed domains for this agent from state.
    Returns list of domains or None if no restrictions.
    """
    # 1) Per-agent domains, if provided
    per_agent = state.get("allowed_domains_by_agent")
    if isinstance(per_agent, dict):
        maybe = per_agent.get(spec.name)
        if isinstance(maybe, list) and maybe:
            return maybe

    # 2) Global domains as fallback
    global_domains = state.get("allowed_domains")
    if isinstance(global_domains, list) and global_domains:
        return global_domains

    return None


def build_tools_config(spec: Any, state: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Build tools/tool_choice/include args for OpenAI Responses API
    based on agent spec tools and state (allowed domains).
    Supports:
      - state["allowed_domains_by_agent"][agent_name]
      - state["allowed_domains"] as a global fallback.
    """
    tools_cfg = spec.tools or {}
    if not tools_cfg.get("enabled"):
        return None

    enabled_tools = tools_cfg.get("tools") or []
    tools_array: List[Dict[str, Any]] = []
    allowed_domains = get_allowed_domains(spec, state)

    for tool_name in enabled_tools:
        if tool_name == "web_search":
            tool_obj: Dict[str, Any] = {"type": "web_search"}
            if allowed_domains:
                tool_obj["filters"] = {"allowed_domains": allowed_domains}
            tools_array.append(tool_obj)
        elif tool_name == "web_search_ddg":
            tool_obj: Dict[str, Any] = {
                "type": "function",
                "function": {"name": "duckduckgo_search"}
            }
            if allowed_domains:
                tool_obj["filters"] = {"allowed_domains": allowed_domains}
            tools_array.append(tool_obj)
        elif tool_name == "web_fetch":
            tool_obj: Dict[str, Any] = {
                "type": "function",
                "function": {"name": "web_fetch"}
            }
            # web_fetch does not use domain filters
            tools_array.append(tool_obj)
        # Other tools could be added here later

    if not tools_array:
        return None

    return {
        "tools": tools_array,
        "tool_choice": "required",
        "include": ["web_search_call.action.sources"],
    }


def build_reasoning_config(spec: Any) -> Dict[str, Any] | None:
    effort = spec.reasoning_effort
    summary = spec.reasoning_summary

    if not effort and not summary:
        return None

    reasoning: Dict[str, Any] = {}
    if effort and effort != "none":
        reasoning["effort"] = effort
    # For summary, "none" means do not request it
    if summary and summary != "none":
        reasoning["summary"] = summary

    if not reasoning:
        return None
    return reasoning


def prepare_tools_for_agent(
    spec: Any,
    state: Dict[str, Any],
    provider_id: str,
    model: str,
    use_responses_api: bool,
    provider_features: Optional[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert configured tools to LangChain-compatible tool specs using provider adapter.
    Merges allowed_domains filters from tools_config into tool specs.
    Returns list of LangChain tool dicts or None if no tools.
    """
    tools_cfg = spec.tools or {}
    # Build tools_config and allowed_domains
    tc = build_tools_config(spec, state)
    allowed_domains = get_allowed_domains(spec, state)
    logger.debug("Agent %s tools_config=%r allowed_domains=%r", spec.name, tc, allowed_domains)

    # Convert using provider adapter (supports OpenAI, DeepSeek, Ollama)
    try:
        converted = convert_tools_for_provider(
            tools_cfg,
            provider_id,
            model,
            use_responses_api,
            provider_features,
        )
        
        langchain_tools = None
        if "tools" in converted:
            langchain_tools = converted["tools"]
        elif "web_search_options" in converted:
            # Completions API style - cannot be passed as tools list
            # Log warning and treat as no tools (binding must happen elsewhere)
            logger.debug(
                f"Agent {spec.name}: web_search_options returned by adapter; "
                f"binding will be handled by LLMClient."
            )
            langchain_tools = None
        # else empty dict -> langchain_tools stays None
        
        # Merge allowed_domains filters from tc into tool specs
        if langchain_tools and isinstance(tc, dict):
            tools_arr = tc.get("tools")
            if isinstance(tools_arr, list):
                for tc_tool in tools_arr:
                    if isinstance(tc_tool, dict) and (tc_tool.get("type") == "web_search" or (tc_tool.get("type") == "function" and tc_tool.get("function", {}).get("name") == "duckduckgo_search")):
                        filters = tc_tool.get("filters")
                        if filters:
                            # Find matching web_search tool in langchain_tools
                            for lt in langchain_tools:
                                if isinstance(lt, dict):
                                    # Match by type "web_search" or function name "web_search"/"duckduckgo_search"
                                    if lt.get("type") == "web_search":
                                        lt["filters"] = filters
                                        break
                                    func = lt.get("function", {})
                                    if isinstance(func, dict) and func.get("name") in ("web_search", "duckduckgo_search"):
                                        # For function tools, set default domain_filter in schema
                                        # and keep filters as extra metadata for compatibility
                                        lt["filters"] = filters
                                        # Set default domain_filter in parameters if present
                                        allowed_domains = filters.get("allowed_domains")
                                        if allowed_domains and isinstance(allowed_domains, list):
                                            params = func.get("parameters", {})
                                            if isinstance(params, dict):
                                                props = params.get("properties", {})
                                                if isinstance(props, dict) and "domain_filter" in props:
                                                    # Set default value in schema (optional, will be hidden)
                                                    props["domain_filter"]["default"] = allowed_domains
                                                    # Hide domain_filter from LLM by removing the property
                                                    del props["domain_filter"]
                                                    # Also remove from required list if present
                                                    required = params.get("required")
                                                    if isinstance(required, list) and "domain_filter" in required:
                                                        required.remove("domain_filter")
                                        break
        return langchain_tools
    except Exception:
        logger.debug("Tool conversion failed for agent %s", spec.name, exc_info=True)
        return None