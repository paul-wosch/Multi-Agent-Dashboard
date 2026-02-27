"""
Primary DuckDuckGo search tool for Multi-Agent Dashboard.

This is the custom tool that wraps patched LangChain DuckDuckGo utilities
and adds domain filtering capability. Registered as "web_search_ddg" in the
tool registry, it is the actual tool invoked by the engine when web search
with domain filtering is requested.

The tool integrates with the patched DuckDuckGoSearchResults from
duckduckgo_base.py and applies optional domain filters either from UI
configuration or tool call parameters.
"""

import logging
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)

# Try to import LangChain BaseTool (optional)
try:
    from langchain.tools import BaseTool
    _LANGCHAIN_TOOLS_AVAILABLE = True
except ImportError:
    # Fallback for environments without LangChain
    class BaseTool:  # type: ignore
        pass
    _LANGCHAIN_TOOLS_AVAILABLE = False

# Import our patched DuckDuckGo classes
try:
    from .duckduckgo_base import DuckDuckGoSearchRun, DuckDuckGoSearchResults
    _DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DuckDuckGoSearchRun = None      # type: ignore
    DuckDuckGoSearchResults = None  # type: ignore
    _DUCKDUCKGO_AVAILABLE = False

from ..registry import register_tool


# JSON Schema matching the adapter's expected format
# This must match the schema in provider_tool_adapter.py:_convert_web_search_ddg_tool()
TOOL_SCHEMA: Dict[str, Any] = {
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


if _LANGCHAIN_TOOLS_AVAILABLE and _DUCKDUCKGO_AVAILABLE:
    @register_tool(
        name="web_search_ddg",
        description="Search the web using DuckDuckGo. Returns relevant web pages with snippets.",
        schema=TOOL_SCHEMA,
        tags=["search", "web", "duckduckgo"],
    )
    class DuckDuckGoSearchTool(BaseTool):
        """DuckDuckGo Search Tool with optional domain filtering.
        
        This is the primary search tool used by the engine, registered as
        "web_search_ddg". It wraps patched LangChain DuckDuckGoSearchResults
        and applies domain filters from UI configuration or tool parameters.
        """
        
        name: str = "duckduckgo_search"
        description: str = "Search the web using DuckDuckGo. Returns relevant web pages with snippets."
        args_schema: Type = None  # Will use JSON schema from registration
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._search_tool = DuckDuckGoSearchResults(output_format="list", backend="duckduckgo")
        
        def _run(
            self,
            query: str,
            domain_filter: Optional[list] = None,
            **kwargs: Any,
        ) -> str:
            """
            Execute DuckDuckGo search.
            
            Args:
                query: Search query
                domain_filter: Optional list of domains to restrict results to
                **kwargs: Additional arguments
            
            Returns:
                Search results as text
            """

            # Use instance attribute if set (from UI domain filters), else fallback to parameter
            effective_filter = getattr(self, '_domain_filter', None) or domain_filter
            if effective_filter:
                logger.info(f"DuckDuckGo search filter: {effective_filter}")
                # Normalize domain filter with multiple domains for query
                domain_clauses = [f"site:{domain}" for domain in effective_filter]
                domain_clauses = [f"OR {clause}" if i > 0 else clause for i, clause in enumerate(domain_clauses)]
                # Inject domain filter to query
                domain_filter_string = " ".join(domain_clauses)
                query = f"{query} {domain_filter_string}"
                logger.info(f"DuckDuckGo search query: {query}")

            try:
                result = self._search_tool.invoke(query)
                return str(result)
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")
                return f"Error performing DuckDuckGo search: {str(e)}"
        
        async def _arun(self, *args, **kwargs):
            """Async version (not implemented)."""
            raise NotImplementedError("Async DuckDuckGo search not supported")
        
else:
    # Create a dummy class for environments without required dependencies
    class DuckDuckGoSearchTool:
        """Dummy class when dependencies are not available."""
        
        def __init__(self, **kwargs):
            raise ImportError(
                "DuckDuckGoSearchTool requires langchain.tools and langchain_community.tools. "
                "Install with: pip install langchain langchain-community"
            )


# Export for easy import
__all__ = ["DuckDuckGoSearchTool", "TOOL_SCHEMA"]