"""
Core sub-package for the LLM client implementation.

This sub-package contains the modular implementation of the LLM client,
decomposing the monolithic LLMClient class into focused, single-responsibility
components. This architecture improves maintainability, testability, and
extensibility while preserving the original public API.

Key components:
- LLMClient: Main facade class coordinating all components
- AgentCreationFacade: Coordinates agent creation with extracted components
- RequestBuilder: Constructs agent invocation states with file handling
- ExecutionEngine: Executes agent invocations with observability and retries
- ResponseProcessor: Processes and normalizes agent responses
- Availability: Conditional import detection and lazy loading

The core sub-package follows the facade pattern, where LLMClient serves as
the public interface while delegating to specialized internal components.
"""
# Core subpackage facade for llm_client
from .client import LLMClient, TextResponse, LLMError
from ..instrumentation import INSTRUMENTATION_MIDDLEWARE

__all__ = [
    "LLMClient",
    "TextResponse",
    "LLMError",
    "INSTRUMENTATION_MIDDLEWARE",
]
