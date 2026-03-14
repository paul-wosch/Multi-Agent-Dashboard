"""
LLM Client package for the Multi-Agent Dashboard.

This package provides a provider-agnostic LLM client interface with support for
multiple LLM providers (OpenAI, DeepSeek, Ollama), structured output, tool calling,
multimodal file handling, and comprehensive instrumentation.

Key components:
- LLMClient: Main client class with provider-agnostic interface
- Core modules: Modular implementation of client internals
- Provider adapters: Provider-specific structured output and tool handling
- Instrumentation: Middleware for metrics, logging, and observability
- Response normalization: Standardization of provider-specific responses

The package uses LangChain's unified init_chat_model interface internally while
exposing a simplified, consistent API to the engine and UI layers.
"""
# multi_agent_dashboard.llm_client package
# Exports the main LLM client interface (instrumentation middleware is exported as None for compatibility).

import sys

from .core import LLMClient, TextResponse, LLMError
from .core import availability as _availability
from .core.availability import (
    LANGCHAIN_AVAILABLE,
    LANGFUSE_AVAILABLE,
    DUCKDUCKGO_TOOL_AVAILABLE,
    get_SystemMessage,
    get_HumanMessage,
    get_AIMessage,
    get_init_chat_model,
    get_create_agent,
    get_AgentMiddleware,
    is_langfuse_enabled,
    get_langfuse_handler,
    DuckDuckGoSearchTool,
)
from .instrumentation import INSTRUMENTATION_MIDDLEWARE

# Provide legacy module path multi_agent_dashboard.llm_client.availability
sys.modules.setdefault(__name__ + ".availability", _availability)

__all__ = [
    "LLMClient",
    "TextResponse",
    "LLMError",
    "INSTRUMENTATION_MIDDLEWARE",
    "LANGCHAIN_AVAILABLE",
    "LANGFUSE_AVAILABLE",
    "DUCKDUCKGO_TOOL_AVAILABLE",
    "get_SystemMessage",
    "get_HumanMessage",
    "get_AIMessage",
    "get_init_chat_model",
    "get_create_agent",
    "get_AgentMiddleware",
    "is_langfuse_enabled",
    "get_langfuse_handler",
    "DuckDuckGoSearchTool",
]
