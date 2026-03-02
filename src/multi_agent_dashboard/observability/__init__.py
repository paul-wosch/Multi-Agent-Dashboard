"""
Observability integrations for the Multi-Agent Dashboard.

This package provides optional observability features for tracing, monitoring, and
debugging agent executions. Currently supports Langfuse integration for distributed
tracing of LLM calls, tool invocations, and agent reasoning steps.

Key Features:
- Langfuse integration for detailed tracing and cost tracking
- Optional dependency (graceful fallback when not configured)
- Automatic flush on program exit
- Environment-based configuration

Usage:
    from multi_agent_dashboard.observability import (
        is_langfuse_enabled,
        get_langfuse_handler,
        flush_langfuse,
    )
    
    if is_langfuse_enabled():
        handler = get_langfuse_handler(session_id="my_pipeline")
        # Pass handler to LLM client for tracing

Configuration:
    Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env file.
    Optionally set LANGFUSE_BASE_URL for self-hosted instances.
    Set LANGFUSE_ENABLED=false to explicitly disable even if keys are present.
"""

from .langfuse import (
    is_langfuse_enabled,
    get_langfuse_handler,
    flush_langfuse,
)

__all__ = [
    "is_langfuse_enabled",
    "get_langfuse_handler",
    "flush_langfuse",
]