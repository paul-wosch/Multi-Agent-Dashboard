"""
Tool integration and provider adaptation for Multi-Agent Dashboard.

This package provides a comprehensive tool ecosystem for LLM agents, including:
- Tool registry with decorator-based registration for LangChain tools
- Provider-specific tool adapters for converting tool configurations across
  different LLM APIs (OpenAI, DeepSeek, Ollama)
- Built-in tools: web search (DuckDuckGo) and web content fetching
- Domain filtering and security controls for web-based tools

The package enables agent tool calling while maintaining provider compatibility
and respecting agent configuration as the primary source of truth. Advisory
capability warnings are provided based on dynamic provider data.
"""

from .registry import ToolRegistry, register_tool, get_registry, register_tool_instance

# Import tool implementations to trigger registration
try:
    from . import search
except ImportError:
    # Tools may not be available if dependencies are missing
    pass

# Import web fetch tool
try:
    from . import web_fetch_tool
except ImportError:
    # Tool may not be available if dependencies are missing
    pass

__all__ = [
    "ToolRegistry",
    "register_tool",
    "get_registry",
    "register_tool_instance",
]