# src/multi_agent_dashboard/runtime_hooks.py
"""
Runtime hook registry for agent-change notifications.

This module provides a small, environment-safe registration surface that
backend code (DB/services) can call to notify the running UI process that
agents have changed in a way that requires cache invalidation and engine
reloads.

Design goals:
- Keep DB/services UI-agnostic (no Streamlit imports here).
- Allow the UI to register concrete invalidation / reload handlers at
  runtime (bootstrap time).
- Make on_agent_change() a best-effort call (exceptions are logged and swallowed).
"""
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)

# Hooks (callables). Set by the UI at bootstrap time.
invalidate_agents_fn: Optional[Callable[[], None]] = None  # type: ignore
reload_agents_fn: Optional[Callable[[], None]] = None  # type: ignore


def register_agent_change_handlers(
    invalidate_agents_fn: Optional[Callable[[], None]],
    reload_agents_fn: Optional[Callable[[], None]],
) -> None:
    """
    Register two functions that will be called when agent metadata changes.

    - invalidate_agents_fn: should clear caches related to agents (e.g., cached_load_agents)
    - reload_agents_fn: should rebuild/load agents into the active engine/runtime

    Either argument may be None; registration simply stores them for later use.
    """
    global _invalidate_agents_fn, _reload_agents_fn
    _invalidate_agents_fn = invalidate_agents_fn
    _reload_agents_fn = reload_agents_fn
    logger.debug(
        "Registered agent-change handlers: invalidate=%s reload=%s",
        bool(_invalidate_agents_fn),
        bool(_reload_agents_fn),
    )


def clear_agent_change_handlers() -> None:
    """Clear any registered handlers (useful for tests)."""
    global _invalidate_agents_fn, _reload_agents_fn
    _invalidate_agents_fn = None
    _reload_agents_fn = None
    logger.debug("Cleared agent-change handlers")


def on_agent_change() -> None:
    """
    Best-effort notification: call the registered invalidation and reload functions.

    Callers (service layer) should call this function when provider-related metadata
    for an agent is created or changed.
    """
    try:
        if _invalidate_agents_fn:
            try:
                _invalidate_agents_fn()
            except Exception as e:
                logger.exception("invalidate_agents_fn raised during on_agent_change: %s", e)
        if _reload_agents_fn:
            try:
                _reload_agents_fn()
            except Exception as e:
                logger.exception("reload_agents_fn raised during on_agent_change: %s", e)
    except Exception:
        # Never let notification break DB/service flows
        logger.exception("on_agent_change encountered an unexpected error")
