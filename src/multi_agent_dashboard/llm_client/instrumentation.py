"""
Instrumentation middleware for LangChain agents with comprehensive metrics collection.

This module provides middleware for collecting detailed metrics from LangChain
agent executions, including token usage, latency, costs, and execution traces.
It integrates with Langfuse for observability and supports graceful fallback
when LangChain dependencies are unavailable.

Key capabilities:
- Real-time metrics collection (tokens, latency, costs)
- Langfuse integration for distributed tracing
- Provider-specific cost calculation using dynamic pricing data
- Fallback compatibility for environments without LangChain
- Structured logging with context preservation

The middleware is designed to be transparent to agent execution while providing
comprehensive observability for debugging and optimization.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)

# Try to import LangChain AgentMiddleware (optional)
_AgentMiddleware = None
try:
    from langchain.agents.middleware import AgentMiddleware  # type: ignore
    _AgentMiddleware = AgentMiddleware
except Exception:
    # Keep resilience when LangChain is not installed or partial environments.
    _AgentMiddleware = None

# If AgentMiddleware import failed, provide a minimal fallback so that tests and
# environments without langchain.agents.middleware can still use LangChain's middleware
# interface (instrumentation middleware is no longer defined). This fallback
# intentionally mirrors the minimal method signatures used by LangChain's middleware system.
if _AgentMiddleware is None:
    class _FallbackAgentMiddleware:
        """
        Minimal fallback middleware base compatible with LangChain's AgentMiddleware interface.
        This allows tests and environments without langchain.agents.middleware to still
        use LangChain's middleware interface (instrumentation middleware is no longer defined).
        """

        # Provide multiple common hook names across LangChain minor versions
        def before_model(self, state: Dict[str, Any], runtime: Any) -> Any:
            return None

        def modify_model_request(self, state: Dict[str, Any], runtime: Any) -> Any:
            return None

        def after_model(self, state: Dict[str, Any], runtime: Any) -> Any:
            return None

        # Some versions use wrap_model_call semantics (return-through handler)
        def wrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
            return handler(request)

    _AgentMiddleware = _FallbackAgentMiddleware





_INSTRUMENTATION_MIDDLEWARE: Optional[type] = None



INSTRUMENTATION_MIDDLEWARE = _INSTRUMENTATION_MIDDLEWARE

class InstrumentationManager:
    """
    Previously handled instrumentation middleware detection, instantiation, and attachment.
    Now only normalizes middleware lists (instrumentation middleware is no longer attached).
    """
    

    
    @staticmethod
    def prepare(middleware: Optional[List[Any]], spec) -> Tuple[List[Any], bool, Optional[str]]:
        """
        Normalize middleware list (instrumentation middleware is no longer attached).
        
        Returns:
            tuple: (middleware_list, instrumentation_attached, instrumentation_attach_error)
                  instrumentation_attached is always False, instrumentation_attach_error is always None.
        """
        logger = logging.getLogger(__name__)
        
        # Normalize middleware list and instantiate classes when provided.
        middleware_list: List[Any] = []
        for mw in (middleware or []):
            try:
                # If a class was passed instead of an instance, try to instantiate.
                if isinstance(mw, type):
                    try:
                        mw_inst = mw()
                        middleware_list.append(mw_inst)
                    except Exception:
                        # Could not instantiate - append the class unchanged (some libs accept classes)
                        middleware_list.append(mw)
                else:
                    middleware_list.append(mw)
            except Exception:
                middleware_list.append(mw)
        
        # Instrumentation middleware is no longer attached
        return middleware_list, False, None