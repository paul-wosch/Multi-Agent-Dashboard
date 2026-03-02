"""
Langfuse observability integration for the Multi-Agent Dashboard.

This module provides optional integration with Langfuse (https://langfuse.com) for
distributed tracing of LLM calls, tool invocations, and agent reasoning steps.
It enables detailed observability including latency breakdown, token usage,
provider costs, and execution metadata.

Key Features:
- Optional dependency with graceful fallback when not configured
- Singleton client with lazy initialization
- Automatic flush on program exit via atexit
- Environment-based configuration
- Support for Langfuse v3.x.x SDK

Architecture:
- Uses singleton pattern for Langfuse client
- Lazy initialization on first use
- Automatic environment variable setup from config
- CallbackHandler creation for individual invocations

Configuration Requirements:
    LANGFUSE_PUBLIC_KEY: Public API key from Langfuse dashboard
    LANGFUSE_SECRET_KEY: Secret API key from Langfuse dashboard
    LANGFUSE_BASE_URL: Optional, for self-hosted instances (default: cloud.langfuse.com)
    LANGFUSE_ENABLED: Optional, set to 'false' to explicitly disable

Integration Points:
    - LLMClient uses get_langfuse_handler() for tracing
    - Automatic flush ensures all traces are sent on program exit
    - Environment variables are automatically set from config

Note: This module is completely optional. If Langfuse credentials are not set,
the integration remains inactive with zero overhead.
"""

import atexit
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Singleton Langfuse client
_langfuse_client = None
_langfuse_handler_class = None
_import_attempted = False
_atexit_registered = False
_langfuse_enabled_cache = None

def _ensure_langfuse_initialized() -> bool:
    """Initialize Langfuse client and handler class exactly once.
    
    Returns True if Langfuse is enabled and successfully imported.
    """
    global _langfuse_client, _langfuse_handler_class, _import_attempted, _atexit_registered
    
    if _import_attempted:
        # Already attempted import, return cached success state
        return _langfuse_handler_class is not None
    
    _import_attempted = True

    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler  # v3.x.x import path
        from multi_agent_dashboard.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
        
        # Ensure environment variables are set for Langfuse SDK
        # Langfuse expects LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
        # We already have them in config; set them in os.environ if not already present
        if LANGFUSE_PUBLIC_KEY:
            os.environ.setdefault('LANGFUSE_PUBLIC_KEY', LANGFUSE_PUBLIC_KEY)
        if LANGFUSE_SECRET_KEY:
            os.environ.setdefault('LANGFUSE_SECRET_KEY', LANGFUSE_SECRET_KEY)
        if LANGFUSE_BASE_URL:
            os.environ.setdefault('LANGFUSE_HOST', LANGFUSE_BASE_URL)
        
        # get_client() reads from environment variables or global singleton
        _langfuse_client = get_client()
        _langfuse_handler_class = CallbackHandler
        
        # Register automatic flush on program exit (only once)
        if not _atexit_registered:
            atexit.register(flush_langfuse)
            _atexit_registered = True
            
        return True
    except ImportError as e:
        logger.debug("Langfuse not available: %s", e)
        return False
    except Exception as e:
        logger.warning("Failed to initialize Langfuse client: %s", e)
        return False

def is_langfuse_enabled() -> bool:
    """Return True if Langfuse environment variables are set and SDK is importable."""
    from multi_agent_dashboard.config import LANGFUSE_ENABLED
    if not LANGFUSE_ENABLED:
        return False
    
    # Cache the result after first evaluation
    global _langfuse_enabled_cache
    if _langfuse_enabled_cache is None:
        _langfuse_enabled_cache = _ensure_langfuse_initialized()
    return _langfuse_enabled_cache

def get_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: str = "multi_agent_dashboard",
    tags: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Create a Langfuse CallbackHandler for a specific invocation.

    Args:
        session_id: Langfuse session ID (maps to pipeline name or "Ad‑Hoc").
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
        # Langfuse v3.x.x CallbackHandler reads from environment variables
        # Metadata is passed via config["metadata"] and config["run_name"]
        handler = _langfuse_handler_class()
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