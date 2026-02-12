"""
Tool integration module for Multi-Agent Dashboard.

Provides a registry for LangChain tools and adapters for LiteLLM tool calling.
"""

from .registry import ToolRegistry, register_tool, get_registry, register_tool_instance
from .litellm_tool_adapter import convert_tools_for_litellm

__all__ = [
    "ToolRegistry",
    "register_tool",
    "get_registry",
    "register_tool_instance",
    "convert_tools_for_litellm",
]