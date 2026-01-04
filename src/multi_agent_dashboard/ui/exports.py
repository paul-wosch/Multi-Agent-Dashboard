# ui/exports.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import streamlit as st

from multi_agent_dashboard.engine import EngineResult, MultiAgentEngine


def _agent_spec_to_dict_safe(spec) -> dict:
    """
    Safely convert an AgentSpec-like object to a plain dict suitable for JSON export.
    Prefer dataclasses.asdict, but fall back to an explicit attribute extraction to
    remain robust if AgentSpec is not a dataclass or contains non-serializable fields.

    Important: always expose 'prompt_template' and 'system_prompt_template' keys
    explicitly in the exported dict (even if None) so downstream consumers can
    rely on their presence.
    """
    try:
        # Preferred if AgentSpec is a dataclass
        d = asdict(spec)
        # Ensure the two prompt keys exist explicitly
        d.setdefault("prompt_template", getattr(spec, "prompt_template", None))
        d.setdefault("system_prompt_template", getattr(spec, "system_prompt_template", None))
        return d
    except Exception:
        # Fall back: extract common attributes used in the UI
        out = {
            "name": getattr(spec, "name", None),
            "model": getattr(spec, "model", None),
            "prompt_template": getattr(spec, "prompt_template", None),
            "role": getattr(spec, "role", None),
            "input_vars": getattr(spec, "input_vars", None),
            "output_vars": getattr(spec, "output_vars", None),
            "color": getattr(spec, "color", None),
            "symbol": getattr(spec, "symbol", None),
            "tools": getattr(spec, "tools", None),
            "reasoning_effort": getattr(spec, "reasoning_effort", None),
            "reasoning_summary": getattr(spec, "reasoning_summary", None),
            "system_prompt_template": getattr(spec, "system_prompt_template", None),
        }
        # Keep prompt keys even if None; drop other keys that are None to keep payload compact
        final = {}
        for k, v in out.items():
            if v is None and k not in ("prompt_template", "system_prompt_template"):
                continue
            final[k] = v
        return final


def export_pipeline_agents_as_json(
    pipeline_name: str,
    steps: List[str],
    engine: Optional[MultiAgentEngine] = None,
) -> str:
    """
    Export all agents used in a pipeline as a JSON string.

    Backwards-compatible: if `engine` is not provided, read engine from st.session_state
    (same behavior as before). Providing an explicit `engine` makes the function easier
    to test in isolation.
    """
    if engine is None:
        if "engine" not in st.session_state:
            raise ValueError("No engine available in session state; provide `engine` argument.")
        engine = st.session_state.engine

    agents_payload = []
    for agent_name in steps:
        agent = engine.agents.get(agent_name)
        if not agent:
            continue
        # Use safe conversion to avoid hard failure if AgentSpec is not a dataclass
        agents_payload.append(_agent_spec_to_dict_safe(agent.spec))

    export = {
        "pipeline": pipeline_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "agents": agents_payload,
    }

    return json.dumps(export, indent=2)


def build_export_from_engine_result(
    result: EngineResult,
    steps: List[str],
    task: str,
    engine: Optional[MultiAgentEngine] = None,
) -> dict:
    """
    Build a JSON-serializable export for the current (live) run,
    mirroring the structure used for stored runs in History.

    Backwards-compatible: if `engine` is None, the function reads the engine from
    st.session_state (previous behavior). Optionally pass an engine explicitly for
    testing or isolation.

    This export now exposes both prompt_template and system_prompt_template for
    each agent under the agent's config.
    """
    if engine is None:
        if "engine" not in st.session_state:
            raise ValueError("No engine available in session state; provide `engine` argument.")
        engine = st.session_state.engine

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
            # Explicitly expose both prompt templates
            "prompt_template": getattr(runtime.spec, "prompt_template", None) if runtime else None,
            "system_prompt_template": getattr(runtime.spec, "system_prompt_template", None) if runtime else None,
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
