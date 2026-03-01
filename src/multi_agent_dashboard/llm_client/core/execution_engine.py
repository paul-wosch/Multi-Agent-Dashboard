"""
Execution engine for LLM client.

This module encapsulates the logic for executing agent invocations with
optional Langfuse observability and retry/backoff handling.
"""

import time
import logging
from typing import Any, Dict, Optional, Callable, Tuple

from ..observability.langfuse_integration import build_langfuse_config

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Executes agent.invoke calls with observability and retry support.
    
    Handles Langfuse configuration and latency measurement.
    """
    
    def __init__(
        self,
        *,
        langfuse_enabled: bool,
        max_retries: int,
        backoff_base: float,
        on_rate_limit: Optional[Callable[[int], None]],
    ):
        """
        Args:
            langfuse_enabled: Whether Langfuse observability is enabled.
            max_retries: Maximum number of retry attempts (currently unused).
            backoff_base: Exponential backoff base factor (currently unused).
            on_rate_limit: Optional callback for rate‑limit events (currently unused).
        """
        self._langfuse_enabled = langfuse_enabled
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._on_rate_limit = on_rate_limit
    
    def execute(
        self,
        agent: Any,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, float]:
        """
        Execute agent.invoke with Langfuse tracing and latency measurement.
        
        Returns (result, latency).
        """
        start_ts = time.perf_counter()

        # Build invocation config with Langfuse callback if enabled
        invoke_config = build_langfuse_config(
            agent,
            context=context,
            langfuse_enabled=self._langfuse_enabled,
        )

        # agent.invoke may accept context parameter in v1 Agents API
        try:
            if context is not None:
                if invoke_config:
                    result = agent.invoke(state, context=context, config=invoke_config)
                else:
                    result = agent.invoke(state, context=context)
            else:
                if invoke_config:
                    result = agent.invoke(state, config=invoke_config)
                else:
                    result = agent.invoke(state)
        except Exception as e:
            logger.debug("agent.invoke failed: %s", e, exc_info=True)
            raise

        end_ts = time.perf_counter()
        latency = end_ts - start_ts
        
        return result, latency