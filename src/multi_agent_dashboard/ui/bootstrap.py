# ui/bootstrap.py
from __future__ import annotations

from typing import Dict

import streamlit as st

from multi_agent_dashboard.config import DB_FILE_PATH, OPENAI_API_KEY, configure_logging
from multi_agent_dashboard.db.db import init_db
from multi_agent_dashboard.engine import MultiAgentEngine
from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard.models import AgentSpec

from multi_agent_dashboard.ui.cache import cached_load_agents, get_agent_service
from multi_agent_dashboard.ui.logging_ui import attach_streamlit_log_handler


# Default agent templates (moved from app.py for bootstrap)
default_agents = {
    "planner": {
        "model": "gpt-4.1-nano",
        "prompt_template": """
You are the Planner Agent.
Refrain from producing the actual solution.
Only clarify the task and produce steps.

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
        "prompt_template": """
You are the Solver Agent.
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
        "prompt_template": """
You are the Critic Agent.
Evaluate the answer.

Answer:
{answer}

Output:
- Issues
- Improvements
""",
        "role": "critic",
        "input_vars": ["answer"],
        "output_vars": ["critique"],
    },
    "finalizer": {
        "model": "gpt-4.1-nano",
        "prompt_template": """
You are the Finalizer Agent.
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
    existing = cached_load_agents()
    if existing:
        return
    svc = get_agent_service()
    for name, data in defaults.items():
        svc.save_agent(
            name,
            data.get("model", "gpt-4.1-nano"),
            data.get("prompt_template", ""),
            data.get("role", ""),
            data.get("input_vars", []),
            data.get("output_vars", []),
        )
        # also save a versioned prompt snapshot
        svc.save_prompt_version(
            name,
            data.get("prompt_template", ""),
            metadata={"role": data.get("role", "")},
        )


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
