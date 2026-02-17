"""
Tool integration module for Multi-Agent Dashboard.

Provides a registry for LangChain tools and adapters for provider-specific tool calling.
"""

from .registry import ToolRegistry, register_tool, get_registry, register_tool_instance

# Import tool implementations to trigger registration
try:
    from . import search
except ImportError:
    # Tools may not be available if dependencies are missing
    pass

__all__ = [
    "ToolRegistry",
    "register_tool",
    "get_registry",
    "register_tool_instance",
]