# ui/exports.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Dict, List

import streamlit as st

from multi_agent_dashboard.engine import EngineResult, MultiAgentEngine


def export_pipeline_agents_as_json(pipeline_name: str, steps: List[str]) -> str:
    """
    Export all agents used in a pipeline as a JSON string.
    Reads the engine from st.session_state to preserve existing call sites.
    """
    engine: MultiAgentEngine = st.session_state.engine

    agents_payload = []
    for agent_name in steps:
        agent = engine.agents.get(agent_name)
        if not agent:
            continue
        agents_payload.append(asdict(agent.spec))

    export = {
        "pipeline": pipeline_name,
        "exported_at": datetime.now(UTC).isoformat(),
        "agents": agents_payload,
    }

    return json.dumps(export, indent=2)


def build_export_from_engine_result(
    result: EngineResult,
    steps: List[str],
    task: str,
) -> dict:
    """
    Build a JSON-serializable export for the current (live) run,
    mirroring the structure used for stored runs in History.

    Reads the engine from st.session_state to preserve the previous behavior.
    """
    engine: MultiAgentEngine = st.session_state.engine

    # Build metrics_by_agent from result.agent_metrics
    metrics_by_agent: Dict[str, dict] = {}
    for name, m in (result.agent_metrics or {}).items():
        if not m:
            continue
        metrics_copy = dict(m)
        metrics_copy.pop("agent_name", None)
        metrics_by_agent[name] = metrics_copy

    # Tool usages by agent (already structured that way in result)
    tool_usages_export_by_agent: Dict[str, List[dict]] = {}
    for name, entries in (result.tool_usages or {}).items():
        tool_usages_export_by_agent[name] = [dict(e) for e in (entries or [])]

    export_agents: Dict[str, dict] = {}
    for name in steps:
        runtime = engine.agents.get(name)
        model = runtime.spec.model if runtime else None
        output = result.memory.get(name, "")

        agent_config = {
            "model": model,
            "role": getattr(runtime.spec, "role", None) if runtime else None,
            "tools": getattr(runtime.spec, "tools", None),
            "reasoning": {
                "effort": getattr(runtime.spec, "reasoning_effort", None)
                if runtime
                else None,
                "summary": getattr(runtime.spec, "reasoning_summary", None)
                if runtime
                else None,
            },
        }

        export_agents[name] = {
            "output": {
                "output": output,
                "is_json": False,  # no explicit JSON flag for live outputs
                "model": model,
            },
            "metrics": metrics_by_agent.get(name),
            "config": agent_config,
            "tool_usages": tool_usages_export_by_agent.get(name) or [],
        }

    # Totals for export
    metrics_iter = list((result.agent_metrics or {}).values())
    total_latency = sum((m.get("latency") or 0.0) for m in metrics_iter)
    total_cost = sum((m.get("cost") or 0.0) for m in metrics_iter)
    total_input_cost = sum((m.get("input_cost") or 0.0) for m in metrics_iter)
    total_output_cost = sum((m.get("output_cost") or 0.0) for m in metrics_iter)

    final_model = (
        engine.agents.get(result.final_agent).spec.model
        if result.final_agent and engine.agents.get(result.final_agent)
        else None
    )

    return {
        "run_id": None,  # live run has no DB id
        "timestamp": datetime.now(UTC).isoformat(),
        "pipeline_summary": {
            "total_latency": round(total_latency, 5),
            "total_input_cost": round(total_input_cost, 6),
            "total_output_cost": round(total_output_cost, 6),
            "total_cost": round(total_cost, 6),
        },
        "task_input": task,
        "final_output": {
            "output": result.final_output,
            "is_json": False,
            "model": final_model,
        },
        "agents": export_agents,
    }
