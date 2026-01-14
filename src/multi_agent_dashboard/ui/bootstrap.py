# ui/bootstrap.py
from __future__ import annotations

from typing import Dict

import streamlit as st
import logging

from multi_agent_dashboard.config import DB_FILE_PATH, OPENAI_API_KEY, configure_logging, AGENT_SNAPSHOT_PRUNE_AUTO, AGENT_SNAPSHOT_PRUNE_KEEP
from multi_agent_dashboard.db.db import init_db
from multi_agent_dashboard.engine import MultiAgentEngine
from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard.models import AgentSpec

from multi_agent_dashboard.ui.cache import cached_load_agents, get_agent_service, invalidate_agents
from multi_agent_dashboard.ui.logging_ui import attach_streamlit_log_handler


# Default agent templates (moved from app.py for bootstrap)
# Each agent now has an explicit system_prompt_template (developer/system role)
# and a prompt_template (user/task-facing content).
default_agents = {
    "planner": {
        "model": "gpt-4.1-nano",
        "system_prompt_template": "You are the Planner Agent. Refrain from producing the actual solution. Only clarify the task and produce steps.",
        "prompt_template": """
Task:
{task}

Output:
- Clarified Task
- Plan
""",
        "role": "planner",
        "input_vars": ["task"],
        "output_vars": ["plan"],
    },
    "solver": {
        "model": "gpt-4.1-nano",
        "system_prompt_template": "You are the Solver Agent. Focus on producing a correct and complete answer using the provided plan.",
        "prompt_template": """
Use the plan to produce an answer.

Plan:
{plan}

Output:
- Answer
""",
        "role": "solver",
        "input_vars": ["plan"],
        "output_vars": ["answer"],
    },
    "critic": {
        "model": "gpt-4.1-nano",
        "system_prompt_template": "You are the Critic Agent. Your role is to verify that the answer fully and correctly satisfies the user’s task. Provide concise, actionable critique of the provided answer.",
        "prompt_template": """
Evaluate the answer.

Answer:
{answer}

Output:
- Issues
- Improvements
""",
        "role": "critic",
        "input_vars": ["task", "answer"],
        "output_vars": ["critique"],
    },
    "finalizer": {
        "model": "gpt-4.1-nano",
        "system_prompt_template": "You are the Finalizer Agent. Use the critique to revise and improve the answer; return the improved final answer only.",
        "prompt_template": """
You must revise the original answer using the critique.

Original Answer:
{answer}

Critique:
{critique}

Your task:
Return the improved final answer only.
""",
        "role": "finalizer",
        "input_vars": ["answer", "critique"],
        "output_vars": ["final"],
    },
}


def create_openai_client(api_key: str):
    """Factory to create an OpenAI client. Allows tests to replace this factory or pass fake client."""
    # Import here to keep optional dependency local to factory
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key)


def bootstrap_default_agents(defaults: Dict[str, dict]):
    """
    Ensure default agents exist in the DB.

    Implementation notes:
    - Use the AgentService directly to avoid reading a cached empty list from
      cached_load_agents() on first run (which would mask newly-inserted rows).
    - Use save_agent_atomic to persist agent metadata and create a prompt version
    in a single transaction.
    - Invalidate the agents cache after inserting so subsequent cached reads
    (reload_agents_into_engine) will pick up the inserted agents.
    """
    svc = get_agent_service()
    # Read DB directly (bypass st.cache_data) to detect empty DB reliably.
    existing = svc.list_agents()
    if existing:
        return

    for name, data in defaults.items():
        svc.save_agent_atomic(
            name,
            data.get("model", "gpt-4.1-nano"),
            data.get("prompt_template", ""),
            data.get("role", ""),
            data.get("input_vars", []),
            data.get("output_vars", []),
            save_prompt_version=True,
            metadata={"role": data.get("role", "")},
            system_prompt=data.get("system_prompt_template", None),
        )

    # Ensure caches are invalidated so the new agents are visible immediately.
    invalidate_agents()


def reload_agents_into_engine():
    """Helper to reload agents into the engine from DB."""
    engine = st.session_state.engine
    engine.agents.clear()

    stored_agents = cached_load_agents()
    for a in stored_agents:
        spec = AgentSpec(
            name=a["agent_name"],
            model=a["model"],
            prompt_template=a["prompt_template"],
            role=a["role"],
            input_vars=a["input_vars"],
            output_vars=a["output_vars"],
            color=a.get("color") or "#6b7280",
            symbol=a.get("symbol") or "·",
            tools=a.get("tools") or {},
            reasoning_effort=a.get("reasoning_effort"),
            reasoning_summary=a.get("reasoning_summary"),
            system_prompt_template=a.get("system_prompt_template"),
            # Provider metadata (Phase 1)
            provider_id=a.get("provider_id"),
            model_class=a.get("model_class"),
            endpoint=a.get("endpoint"),
            use_responses_api=bool(a.get("use_responses_api")),
            provider_features=a.get("provider_features") or {},
        )
        engine.add_agent(spec)


def app_start():
    """
    Explicit application bootstrap. Call this once when running the Streamlit app.
    - initializes DB and migrations
    - creates an OpenAI client via factory
    - creates the MultiAgentEngine with the injected client
    - bootstraps default agents if DB empty
    - loads agents from DB into the engine
    - stores engine into st.session_state
    """
    # Attach Streamlit log handler once per session (and load historic logs)
    attach_streamlit_log_handler(capacity=500)

    # Initialize DB and apply migrations
    init_db(DB_FILE_PATH)

    # create OpenAI client (factory)
    openai_client = create_openai_client(OPENAI_API_KEY)
    llm_client = LLMClient(openai_client)

    # create engine with injected client
    engine = MultiAgentEngine(llm_client=llm_client)
    st.session_state.engine = engine

    # Ensure default agents exist in DB (only if table empty)
    bootstrap_default_agents(default_agents)

    # Load agents from DB into engine
    reload_agents_into_engine()

    # Optionally keep client in session state for custom use
    st.session_state.llm_client = llm_client

    # Optional automatic pruning of old snapshots at startup (configurable)
    if AGENT_SNAPSHOT_PRUNE_AUTO:
        try:
            from multi_agent_dashboard.db.infra.maintenance import prune_agent_snapshots

            deleted = prune_agent_snapshots(agent_name=None, keep=AGENT_SNAPSHOT_PRUNE_KEEP)
            logging.getLogger(__name__).info(
                "Automatic startup prune: removed %d snapshots (keep=%d)",
                deleted,
                AGENT_SNAPSHOT_PRUNE_KEEP,
            )
        except Exception:
            logging.getLogger(__name__).exception("Failed to run automatic snapshot pruning at startup")
