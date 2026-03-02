"""
Search tool integration with DuckDuckGo and domain filtering.

This subpackage provides web search capabilities for LLM agents through
DuckDuckGo integration. It includes patched LangChain components with
enhanced functionality and domain filtering support.

Key components:
- `duckduckgo_base.py`: Patched LangChain DuckDuckGo utilities with backend
  parameter support and enhanced search results
- `duckduckgo_tool.py`: Custom DuckDuckGo search tool with domain filtering
  and registry integration

The search tools support both text-based search results and domain-restricted
searching for controlled web research tasks.
"""

from .duckduckgo_tool import DuckDuckGoSearchTool, TOOL_SCHEMA

__all__ = ["DuckDuckGoSearchTool", "TOOL_SCHEMA"]