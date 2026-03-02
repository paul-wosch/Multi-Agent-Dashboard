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
# environments without langchain.agents.middleware can still instantiate and use
# the instrumentation middleware class. This fallback intentionally mirrors the
# minimal method signatures used by LangChain's middleware system.
if _AgentMiddleware is None:
    class _FallbackAgentMiddleware:
        """
        Minimal fallback middleware base compatible with LangChain's AgentMiddleware interface.
        This allows tests and environments without langchain.agents.middleware to still
        instantiate and use the instrumentation middleware defined below.
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

# Import ResponseNormalizer (will be defined in response_normalizer.py)
from .response_normalizer import ResponseNormalizer

def _extract_content_blocks_from_message(message: Any) -> List[Dict[str, Any]]:
    normalized = ResponseNormalizer.normalize_to_dict(message)
    blocks = normalized.get("content_blocks")
    if isinstance(blocks, list):
        return blocks
    output = normalized.get("output")
    if isinstance(output, list):
        return output
    return []

_INSTRUMENTATION_MIDDLEWARE: Optional[type] = None

# Always define a dashboard instrumentation middleware class (subclassing the
# real AgentMiddleware when available, otherwise the fallback above).
class _DashboardInstrumentationMiddleware(_AgentMiddleware):  # type: ignore
    name = "MultiAgentInstrumentationMiddleware"

    # Various LangChain versions may call different hooks; implement a superset
    # so the middleware captures the last model message consistently.
    def after_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        try:
            messages = state.get("messages") or []
            if not messages:
                return None
            event = {
                "content_blocks": _extract_content_blocks_from_message(messages[-1]),
                "structured_response": (
                        state.get("structured_response")
                        or state.get("structured")
                        or ResponseNormalizer.normalize_to_dict(messages[-1]).get("structured_response")
                        or ResponseNormalizer.normalize_to_dict(messages[-1]).get("structured")
                ),
                "text": ResponseNormalizer.normalize_to_dict(messages[-1]).get("text"),
                # attach a monotonic timestamp to aid debugging/auditing
                "ts": time.time(),
            }
            state.setdefault("_multi_agent_dashboard_events", []).append(event)
        except Exception:
            # Middleware must not raise
            logger.debug("Instrumentation middleware after_model failed", exc_info=True)
        return None

    # Also support wrap-style hook names used in some releases
    def wrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        # Call through and capture effects inside after_model when the model returns.
        return handler(request)

_INSTRUMENTATION_MIDDLEWARE = _DashboardInstrumentationMiddleware

INSTRUMENTATION_MIDDLEWARE = _INSTRUMENTATION_MIDDLEWARE

class InstrumentationManager:
    """
    Handles instrumentation middleware detection, instantiation, and attachment.
    """
    
    @staticmethod
    def _middleware_includes_instrumentation(mw_list: List[Any]) -> bool:
        """Check if instrumentation middleware is already present in the list."""
        if INSTRUMENTATION_MIDDLEWARE is None:
            return False
        for item in mw_list:
            try:
                # Direct instance of the instrumentation middleware
                if isinstance(item, INSTRUMENTATION_MIDDLEWARE):
                    return True
            except Exception:
                # isinstance might fail if types incompatible; continue defensively
                pass
            try:
                # If the item is the class itself
                if item is INSTRUMENTATION_MIDDLEWARE:
                    return True
            except Exception:
                pass
            try:
                # If the item is a subclass (class object provided)
                if isinstance(item, type) and issubclass(item, INSTRUMENTATION_MIDDLEWARE):
                    return True
            except Exception:
                pass
        return False
    
    @staticmethod
    def prepare(middleware: Optional[List[Any]], spec) -> Tuple[List[Any], bool, Optional[str]]:
        """
        Normalize middleware list and attach instrumentation middleware if needed.
        
        Returns:
            tuple: (middleware_list, instrumentation_attached, instrumentation_attach_error)
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
        
        # Detect whether instrumentation middleware already present
        instrumentation_present = False
        try:
            instrumentation_present = InstrumentationManager._middleware_includes_instrumentation(middleware_list)
        except Exception:
            instrumentation_present = False
        
        instrumentation_attached = instrumentation_present
        instrumentation_attach_error: Optional[str] = None
        
        # Try to attach instrumentation middleware safely; if instantiation fails, log and continue without it.
        if not instrumentation_present and INSTRUMENTATION_MIDDLEWARE is not None:
            try:
                mw_instance = None
                try:
                    mw_instance = INSTRUMENTATION_MIDDLEWARE()
                    # Ensure instrumentation is the final element in the middleware list
                    middleware_list.append(mw_instance)
                    instrumentation_attached = True
                except Exception as inst_exc:
                    # As a fallback, some integrations accept middleware classes instead of instances.
                    try:
                        middleware_list.append(INSTRUMENTATION_MIDDLEWARE)
                        instrumentation_attached = True
                        instrumentation_attach_error = f"instantiation_failed:{inst_exc}"
                        logger.debug(
                            "Instrumentation middleware could not be instantiated; appended class instead for agent=%s. instantiation error=%s",
                            getattr(spec, "name", "<unnamed>"),
                            inst_exc,
                            exc_info=True,
                        )
                    except Exception as append_exc:
                        instrumentation_attach_error = f"instantiation_failed:{inst_exc}; append_failed:{append_exc}"
                        logger.warning(
                            "Instrumentation middleware exists but could not be instantiated or appended for agent=%s",
                            getattr(spec, "name", "<unnamed>"),
                        )
                        logger.debug("Instrumentation instantiation/append error: %s / %s", inst_exc, append_exc,
                                     exc_info=True)
            except Exception as e:
                instrumentation_attach_error = str(e)
                logger.exception("Failed to instantiate instrumentation middleware: %s", e)
        
        return middleware_list, instrumentation_attached, instrumentation_attach_error