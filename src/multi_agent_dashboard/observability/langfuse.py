"""
Langfuse observability integration (optional).

This module is only active when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.
"""

import atexit
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Singleton Langfuse client
_langfuse_client = None
_langfuse_handler_class = None

def _try_import_langfuse():
    """Conditionally import Langfuse components."""
    global _langfuse_client, _langfuse_handler_class
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler  # v3.x.x import path
        _langfuse_client = get_client()
        # Register automatic flush on program exit
        atexit.register(flush_langfuse)
        _langfuse_handler_class = CallbackHandler
        return True
    except ImportError as e:
        logger.warning("Langfuse not available: %s", e)
        return False

def is_langfuse_enabled() -> bool:
    """Return True if Langfuse environment variables are set and SDK is importable."""
    from multi_agent_dashboard.config import LANGFUSE_ENABLED
    if not LANGFUSE_ENABLED:
        return False
    return _try_import_langfuse()

def get_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: str = "multi_agent_dashboard",
    tags: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Create a Langfuse CallbackHandler for a specific invocation.

    Args:
        session_id: Langfuse session ID (maps to run_id).
        user_id: Static user identifier (default: "multi_agent_dashboard").
        tags: List of tags (e.g., ["agent:my_agent", "pipeline:my_pipeline"]).
        metadata: Additional custom metadata.

    Returns:
        A CallbackHandler instance, or None if Langfuse is disabled.
    """
    if not is_langfuse_enabled():
        return None

    if _langfuse_handler_class is None:
        logger.warning("Langfuse handler class not available")
        return None

    try:
        # Langfuse v3.x.x expects metadata via config, not handler constructor
        # We'll attach metadata later in the invocation config
        handler = _langfuse_handler_class()
        # Note: session_id, user_id, tags are passed via config["metadata"]
        return handler
    except Exception as e:
        logger.error("Failed to create Langfuse handler: %s", e)
        return None

def flush_langfuse():
    """Ensure all pending traces are sent to Langfuse (called automatically on exit)."""
    if is_langfuse_enabled() and _langfuse_client is not None:
        try:
            _langfuse_client.flush()
        except Exception as e:
            logger.warning("Failed to flush Langfuse client: %s", e)