# multi_agent_dashboard.llm_client package
# Exports the main LLM client interface and instrumentation middleware.

from .core import LLMClient, TextResponse, LLMError
from .instrumentation import INSTRUMENTATION_MIDDLEWARE

__all__ = ["LLMClient", "TextResponse", "LLMError", "INSTRUMENTATION_MIDDLEWARE"]