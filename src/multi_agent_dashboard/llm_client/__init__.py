# multi_agent_dashboard.llm_client package
# Exports the main LLM client interface and instrumentation middleware.

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
