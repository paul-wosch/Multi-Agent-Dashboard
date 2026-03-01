"""
Langfuse integration for LLM client observability.

Provides a function to build invocation configuration for Langfuse tracing.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import Langfuse utilities from centralized availability module
from ..core.availability import get_langfuse_handler


def build_langfuse_config(
    agent: Any,
    context: Optional[Dict[str, Any]] = None,
    langfuse_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Build invocation config dictionary for Langfuse tracing.

    Args:
        agent: LangChain agent instance.
        context: Optional context dict with pipeline_name, run_id.
        langfuse_enabled: Whether Langfuse is enabled for this client.

    Returns:
        invoke_config dict (empty if Langfuse disabled or handler unavailable).
    """
    invoke_config = {}
    if not langfuse_enabled:
        return invoke_config

    # Extract metadata from context and agent spec
    pipeline_name = None
    run_id = None
    tags = []
    metadata = {}

    if context is not None:
        pipeline_name = context.get("pipeline_name")
        run_id = context.get("run_id")

    # Always add pipeline tag (pipeline name or "Ad‑Hoc" for ad‑hoc runs)
    pipeline_tag_value = pipeline_name or "Ad‑Hoc"
    tags.append(f"pipeline:{pipeline_tag_value}")
    
    if run_id:
        tags.append(f"run:{run_id}")

    # Agent name (from agent._name, agent.name, or spec)
    agent_name = getattr(agent, "_name", None)
    if not agent_name:
        agent_name = getattr(agent, "name", None)
    if not agent_name:
        # Fallback: try to get from agent spec if available
        agent_spec = getattr(agent, "_agent_spec", None)
        if agent_spec and hasattr(agent_spec, "name"):
            agent_name = agent_spec.name
    
    # Default agent name if still None
    if not agent_name:
        agent_name = "unknown"
    
    # Add agent tag
    tags.append(f"agent:{agent_name}")

    # Session ID: pipeline name or "Ad‑Hoc" for ad‑hoc runs
    session_id = pipeline_name or "Ad‑Hoc"

    # Create Langfuse handler (no constructor parameters needed)
    if get_langfuse_handler is not None:
        handler = get_langfuse_handler()
        if handler is not None:
            invoke_config["callbacks"] = [handler]
            # Trace name (appears as "Name" in Langfuse UI)
            invoke_config["run_name"] = agent_name
            # Langfuse‑specific metadata keys
            invoke_config["metadata"] = {
                "langfuse_session_id": session_id,
                "langfuse_user_id": "multi_agent_dashboard",
                "langfuse_tags": tags,
                **metadata,
            }

    return invoke_config