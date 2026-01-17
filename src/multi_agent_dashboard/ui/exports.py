# ui/exports.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

import streamlit as st

from multi_agent_dashboard.engine import EngineResult, MultiAgentEngine


def _parse_endpoint_host(endpoint: Optional[str]) -> Optional[str]:
    """
    Parse endpoint string and return a host[:port] representation when possible.
    Returns None when endpoint is falsy or not parseable.
    """
    if not endpoint:
        return None
    try:
        p = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
        host = p.hostname
        port = p.port
        if host is None:
            return None
        if port:
            return f"{host}:{port}"
        return host
    except Exception:
        return None


def _provider_friendly_name(provider_id: Optional[str]) -> str:
    mapping = {
        "openai": "OpenAI",
        "azure_openai": "Azure OpenAI",
        "ollama": "Ollama (local)",
        "anthropic": "Anthropic",
        "custom": "Custom",
        "": "OpenAI",
        None: "OpenAI",
    }
    pid = (provider_id or "").strip().lower()
    return mapping.get(pid, pid or "Unknown")


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

        # Normalize provider object to make downstream reconstruction easier
        provider_id = d.get("provider_id")
        endpoint = d.get("endpoint")
        provider_obj = {
            "id": provider_id,
            "name": _provider_friendly_name(provider_id),
            "endpoint": endpoint,
            "host": _parse_endpoint_host(endpoint),
            "features": d.get("provider_features") or {},
        }
        d["provider_snapshot"] = provider_obj

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
            # Provider metadata
            "provider_id": getattr(spec, "provider_id", None),
            "model_class": getattr(spec, "model_class", None),
            "endpoint": getattr(spec, "endpoint", None),
            "use_responses_api": getattr(spec, "use_responses_api", None),
            "provider_features": getattr(spec, "provider_features", None),
        }
        # Build provider_snapshot
        provider_obj = {
            "id": out.get("provider_id"),
            "name": _provider_friendly_name(out.get("provider_id")),
            "endpoint": out.get("endpoint"),
            "host": _parse_endpoint_host(out.get("endpoint")),
            "features": out.get("provider_features") or {},
        }
        out["provider_snapshot"] = provider_obj

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

    Exports include provider snapshot information (id, name, host, endpoint, features)
    to allow downstream systems to reconstruct the original environment.
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
        spec_dict = _agent_spec_to_dict_safe(agent.spec)
        agents_payload.append(spec_dict)

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

    This export now exposes provider metadata (id, name, host), content_blocks,
    instrumentation events, and low-level reasoning/tools config so downstream
    consumers can fully reconstruct the runtime environment.
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

        # Prefer provider snapshot from the per-run agent_configs (if available), else fallback to spec-level snapshot
        run_cfg = (result.agent_configs or {}).get(name, {}) or {}

        # Derive provider snapshot (explicit fields)
        provider_snapshot = {
            "provider_id": run_cfg.get("provider_id") or (getattr(runtime.spec, "provider_id", None) if runtime else None),
            "provider_name": _provider_friendly_name(run_cfg.get("provider_id") or (getattr(runtime.spec, "provider_id", None) if runtime else None)),
            "model_class": run_cfg.get("model_class") or (getattr(runtime.spec, "model_class", None) if runtime else None),
            "endpoint": run_cfg.get("endpoint") or (getattr(runtime.spec, "endpoint", None) if runtime else None),
            "host": _parse_endpoint_host(run_cfg.get("endpoint") or (getattr(runtime.spec, "endpoint", None) if runtime else None)),
            "use_responses_api": bool(run_cfg.get("use_responses_api")) if run_cfg else (getattr(runtime.spec, "use_responses_api", None) if runtime else None),
            "provider_features": run_cfg.get("provider_features") if run_cfg and run_cfg.get("provider_features") is not None else (getattr(runtime.spec, "provider_features", None) if runtime else None),
        }

        # Determine effective tools configuration, and capture any allowed_domains present in runtime state
        spec_tools = getattr(runtime.spec, "tools", None) if runtime else None
        # Derive allowed_domains for this agent from run state (result.state)
        allowed_domains_agent = None
        try:
            if isinstance(result.state, dict):
                adb = result.state.get("allowed_domains_by_agent")
                if isinstance(adb, dict):
                    allowed_domains_agent = adb.get(name)
                else:
                    global_allowed = result.state.get("allowed_domains")
                    if isinstance(global_allowed, list) and global_allowed:
                        allowed_domains_agent = global_allowed
        except Exception:
            allowed_domains_agent = None

        tools_snapshot = dict(spec_tools) if isinstance(spec_tools, dict) else (spec_tools or {})
        if allowed_domains_agent:
            # include the effective allowed_domains used during the run for auditing/export
            tools_snapshot = dict(tools_snapshot)
            tools_snapshot["allowed_domains"] = allowed_domains_agent

        # Low-level configs derived from run snapshot (if present)
        tools_config = run_cfg.get("tools_config")
        reasoning_config = run_cfg.get("reasoning_config")

        # Extra from the per-run snapshot (these often include content_blocks, detected_provider_profile, instrumentation events)
        extra = run_cfg.get("extra") or {}

        # Extract content_blocks & instrumentation events explicitly when available
        content_blocks = extra.get("content_blocks") if isinstance(extra, dict) else None
        instrumentation_events = extra.get("instrumentation_events") if isinstance(extra, dict) else None
        structured_response = extra.get("structured_response") if isinstance(extra, dict) else None

        agent_config = {
            "model": model,
            "role": getattr(runtime.spec, "role", None) if runtime else None,
            "tools": tools_snapshot,
            "reasoning": {
                "effort": (run_cfg.get("reasoning_effort") if run_cfg and run_cfg.get("reasoning_effort") is not None else (getattr(runtime.spec, "reasoning_effort", None) if runtime else None)),
                "summary": (run_cfg.get("reasoning_summary") if run_cfg and run_cfg.get("reasoning_summary") is not None else (getattr(runtime.spec, "reasoning_summary", None) if runtime else None)),
                "config": reasoning_config,
            },
            # Provider snapshot (prefer per-run derived snapshot)
            "provider": provider_snapshot,
            # Explicitly expose both prompt templates
            "prompt_template": (run_cfg.get("prompt_template") if run_cfg and run_cfg.get("prompt_template") is not None else (getattr(runtime.spec, "prompt_template", None) if runtime else None)),
            "system_prompt_template": (run_cfg.get("system_prompt_template") if run_cfg and run_cfg.get("system_prompt_template") is not None else (getattr(runtime.spec, "system_prompt_template", None) if runtime else None)),
            # Low-level tools config
            "tools_config": tools_config,
            # Structured/extra metadata surfaced from LangChain middleware / responses
            "content_blocks": content_blocks,
            "instrumentation_events": instrumentation_events,
            "structured_response": structured_response,
            # Raw extra for auditing / debugging
            "raw_extra": extra,
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
