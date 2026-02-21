# multi_agent_dashboard.llm_client.tool_binder
# Tool conversion, binding, and tool instance retrieval.

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Conditional import for DuckDuckGoSearchTool (may not be available if LangChain missing)
try:
    from multi_agent_dashboard.tool_integration.search import DuckDuckGoSearchTool
    _DUCKDUCKGO_TOOL_AVAILABLE = True
except ImportError:
    DuckDuckGoSearchTool = None
    _DUCKDUCKGO_TOOL_AVAILABLE = False

# Import required dependencies
from multi_agent_dashboard.tool_integration.provider_tool_adapter import convert_tools_for_provider
from multi_agent_dashboard.tool_integration.registry import get_registry
from .structured_output import StructuredOutputBinder

class ToolBinder:
    """
    Handles tool conversion, binding, and tool instance retrieval.
    """

    def __init__(self, client):
        self._client = client

    def process_tools(self, spec, model_instance, response_format, provider_id, tools_param):
        """
        Convert tool configs, bind tools to model, retrieve tool instances.
        
        Returns:
            tuple: (model_instance, tools_list, unified_binding_applied, effective_response_format)
        """
        # Recompute provider_id from spec for consistency
        provider_id = (getattr(spec, "provider_id", None) or "openai").lower()
        model = getattr(spec, "model", "")
        use_responses_api = getattr(spec, "use_responses_api", False)
        tool_configs = getattr(spec, "tools", {})
        enabled_tools = tool_configs.get("tools", []) if tool_configs.get("enabled", False) else []
        logger.debug("Converting tools for provider; tools param=%s, spec.tools=%s", tools_param, tool_configs)
        # Convert tool configs to provider-specific format
        provider_features = getattr(spec, "provider_features", None)
        converted_tools = convert_tools_for_provider(
            tool_configs, provider_id, model, use_responses_api, provider_features
        )
        # Bind tools to model instance if applicable
        # Check if we should attempt unified binding with both structured output and tools
        unified_binding_applied = False
        effective_response_format = response_format  # start with original
        if response_format is not None and converted_tools and "tools" in converted_tools:
            structured_binder = StructuredOutputBinder(self._client)
            model_instance, effective_response_format = structured_binder.bind_structured_output(
                spec, model_instance, response_format, provider_id, model,
                tools=converted_tools["tools"], strict=False
            )
            if effective_response_format is None:
                unified_binding_applied = True
                logger.info(f"Applied unified tools+structured_output binding for {provider_id}")
            else:
                logger.warning("Unified binding failed, falling back to sequential")

        if converted_tools:
            if "tools" in converted_tools:
                # Bind function tools only if unified binding wasn't applied
                if not unified_binding_applied:
                    model_instance = model_instance.bind_tools(converted_tools["tools"])
                    logger.info(f"Bound {len(converted_tools['tools'])} function tool(s) to model")
                # Retrieve tool instances from registry for create_agent
                tool_instances = []
                # Build mapping from tool name to filters from converted tool specs
                tool_filters = {}
                for tool_dict in converted_tools["tools"]:
                    if isinstance(tool_dict, dict):
                        tool_type = tool_dict.get("type")
                        if tool_type == "web_search":
                            # Native web_search tool
                            filters = tool_dict.get("filters")
                            if filters:
                                tool_filters["web_search"] = filters.get("allowed_domains")
                        elif tool_type == "function":
                            func = tool_dict.get("function", {})
                            func_name = func.get("name")
                            if func_name == "duckduckgo_search":
                                filters = tool_dict.get("filters")
                                if filters:
                                    tool_filters["web_search_ddg"] = filters.get("allowed_domains")
                            elif func_name == "web_search":
                                # Generic web search function tool for non-OpenAI providers
                                filters = tool_dict.get("filters")
                                if filters:
                                    tool_filters["web_search"] = filters.get("allowed_domains")
                for tool_name in enabled_tools:
                    tool_instance = get_registry().get_tool(tool_name)
                    if tool_instance is not None:
                        # Apply domain filters to DuckDuckGoSearchTool instance (hidden from LLM)
                        allowed_domains = tool_filters.get(tool_name)
                        if allowed_domains and _DUCKDUCKGO_TOOL_AVAILABLE and isinstance(tool_instance, DuckDuckGoSearchTool):
                            tool_instance._domain_filter = allowed_domains
                            logger.debug(f"Set domain filter on DuckDuckGoSearchTool: {allowed_domains}")
                        tool_instances.append(tool_instance)
                    else:
                        # Tool not in registry (e.g., native web_search); skip
                        logger.debug(f"Tool '{tool_name}' not found in registry; assuming native tool")
                # Replace tools list with tool instances
                tools_list = tool_instances
            elif "web_search_options" in converted_tools:
                # Bind web search options (Completions API)
                model_instance = model_instance.bind(web_search_options=converted_tools["web_search_options"])
                logger.info(f"Bound web search options to model")
                # Clear tools list since web_search_options is being used instead
                tools_list = []
        else:
            # No tools applicable after conversion
            tools_list = []

        return model_instance, tools_list, unified_binding_applied, effective_response_format