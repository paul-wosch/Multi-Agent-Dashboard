"""
Centralized conditional import detection for the LLM client subpackage.

This module provides lazy-loaded references to optional dependencies
(LangChain, Langfuse, DuckDuckGoSearchTool) and boolean flags indicating
their availability.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = [
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

# ------------------------------------------------------------------------------
# DuckDuckGoSearchTool (optional, depends on LangChain)
# ------------------------------------------------------------------------------
try:
    from multi_agent_dashboard.tool_integration.search import DuckDuckGoSearchTool
    DUCKDUCKGO_TOOL_AVAILABLE = True
except ImportError:
    DuckDuckGoSearchTool = None
    DUCKDUCKGO_TOOL_AVAILABLE = False

# ------------------------------------------------------------------------------
# Langfuse observability (optional)
# ------------------------------------------------------------------------------
try:
    from multi_agent_dashboard.observability import is_langfuse_enabled, get_langfuse_handler
    LANGFUSE_AVAILABLE = True
except ImportError:
    is_langfuse_enabled = None
    get_langfuse_handler = None
    LANGFUSE_AVAILABLE = False

# ------------------------------------------------------------------------------
# LangChain components (optional)
# ------------------------------------------------------------------------------
# Global availability flag
LANGCHAIN_AVAILABLE = False

# Lazy references – will be set to the actual classes if import succeeds,
# otherwise remain None.
_SystemMessage = None
_HumanMessage = None
_AIMessage = None
_init_chat_model = None
_create_agent = None
_AgentMiddleware = None

try:
    from langchain.chat_models import init_chat_model  # type: ignore
    from langchain.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
    from langchain.agents import create_agent  # type: ignore
    from langchain.agents.middleware import AgentMiddleware  # type: ignore

    LANGCHAIN_AVAILABLE = True
    _SystemMessage = SystemMessage
    _HumanMessage = HumanMessage
    _AIMessage = AIMessage
    _init_chat_model = init_chat_model
    _create_agent = create_agent
    _AgentMiddleware = AgentMiddleware
except Exception:
    # Keep resilience when LangChain is not installed or partial environments.
    LANGCHAIN_AVAILABLE = False
    _SystemMessage = None
    _HumanMessage = None
    _AIMessage = None
    _init_chat_model = None
    _create_agent = None
    _AgentMiddleware = None

# ------------------------------------------------------------------------------
# Public getter functions (defer import errors)
# ------------------------------------------------------------------------------
def get_SystemMessage():
    """Return SystemMessage class if LangChain available, else None."""
    return _SystemMessage

def get_HumanMessage():
    """Return HumanMessage class if LangChain available, else None."""
    return _HumanMessage

def get_AIMessage():
    """Return AIMessage class if LangChain available, else None."""
    return _AIMessage

def get_init_chat_model():
    """Return init_chat_model function if LangChain available, else None."""
    return _init_chat_model

def get_create_agent():
    """Return create_agent function if LangChain available, else None."""
    return _create_agent

def get_AgentMiddleware():
    """Return AgentMiddleware class if LangChain available, else None."""
    return _AgentMiddleware