# ui/cache.py
from __future__ import annotations

from typing import List, Tuple, Optional

import streamlit as st

from multi_agent_dashboard.config import DB_FILE_PATH
from multi_agent_dashboard.db.services import AgentService, PipelineService, RunService

# Lazy service singletons (created on first use)
_agent_svc: Optional[AgentService] = None
_pipeline_svc: Optional[PipelineService] = None
_run_svc: Optional[RunService] = None


def get_agent_service() -> AgentService:
    global _agent_svc
    if _agent_svc is None:
        _agent_svc = AgentService(DB_FILE_PATH)
    return _agent_svc


def get_pipeline_service() -> PipelineService:
    global _pipeline_svc
    if _pipeline_svc is None:
        _pipeline_svc = PipelineService(DB_FILE_PATH)
    return _pipeline_svc


def get_run_service() -> RunService:
    global _run_svc
    if _run_svc is None:
        _run_svc = RunService(DB_FILE_PATH)
    return _run_svc


# Cached loaders (originally in app.py)
@st.cache_data(ttl=60)
def cached_load_agents() -> List[dict]:
    svc = get_agent_service()
    return svc.list_agents()


@st.cache_data(ttl=60)
def cached_load_pipelines() -> List[dict]:
    svc = get_pipeline_service()
    return svc.list_pipelines()


@st.cache_data(ttl=30)
def cached_load_runs() -> List[dict]:
    svc = get_run_service()
    return svc.list_runs()


@st.cache_data(ttl=30)
def cached_load_run_details(run_id: int) -> Tuple[dict, List[dict], List[dict], List[dict], List[dict]]:
    svc = get_run_service()
    return svc.get_run_details(run_id)


@st.cache_data(ttl=60)
def cached_load_agent_snapshots(agent_name: str) -> List[dict]:
    """
    Cached loader for agent snapshots. Returns [] on error to keep UI resilient
    if migrations have not yet been applied.
    """
    try:
        svc = get_agent_service()
        return svc.list_snapshots(agent_name)
    except Exception:
        # If the agent_snapshots table doesn't exist yet (older DB), don't blow up the UI
        return []


# Cache invalidation helpers (moved from app.py)
def invalidate_caches(*names: str):
    """
    Centralized cache invalidation helper.
    Allowed names:
      - agents
      - pipelines
      - runs
      - run_details
      - snapshots
      - all
    """
    if "all" in names:
        cached_load_agents.clear()
        cached_load_pipelines.clear()
        cached_load_runs.clear()
        cached_load_run_details.clear()
        cached_load_agent_snapshots.clear()
        return

    for name in names:
        if name == "agents":
            cached_load_agents.clear()
        elif name == "pipelines":
            cached_load_pipelines.clear()
        elif name == "runs":
            cached_load_runs.clear()
        elif name == "run_details":
            cached_load_run_details.clear()
        elif name == "snapshots":
            cached_load_agent_snapshots.clear()


def invalidate_agents():
    """Invalidate caches related to agents and snapshots."""
    invalidate_caches("agents", "snapshots")


def invalidate_pipelines():
    """Invalidate caches related to pipelines."""
    invalidate_caches("pipelines")


def invalidate_runs():
    """Invalidate caches related to runs and run details."""
    invalidate_caches("runs", "run_details")
