# ui/history_mode.py
from __future__ import annotations

from typing import Dict, List

import streamlit as st

from multi_agent_dashboard.ui.cache import cached_load_runs, cached_load_run_details
from multi_agent_dashboard.ui.view_models import metrics_view_from_db_rows, config_view_from_db_rows
from multi_agent_dashboard.ui.metrics_view import render_cost_latency_section
from multi_agent_dashboard.ui.tools_view import render_agent_config_section, build_tool_calls_overview
from multi_agent_dashboard.ui.utils import parse_json_field

logger = __import__("logging").getLogger(__name__)

from urllib.parse import urlparse


def _provider_friendly_name(provider_id: str | None) -> str:
    mapping = {
        "openai": "OpenAI",
        "azure_openai": "Azure OpenAI",
        "ollama": "Ollama (local)",
        "anthropic": "Anthropic",
        "custom": "Custom",
        None: "OpenAI",
        "": "OpenAI",
    }
    pid = (provider_id or "").strip().lower()
    return mapping.get(pid, pid or "Unknown")


def _parse_endpoint_host(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    try:
        p = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
        host = p.hostname
        port = p.port
        if host is None:
            return None
        return f"{host}:{port}" if port else host
    except Exception:
        return None


def render_history_mode():
    st.header("📜 Past Runs")

    runs = cached_load_runs()

    def abbreviate_task(text: str, max_words: int = 16) -> str:
        if not text:
            return ""
        words = text.strip().split()
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words]) + "…"

    def get_agent_names_for_run(run_id: int) -> str:
        try:
            _, agents, _, _, _ = cached_load_run_details(run_id)
        except Exception:
            logger.exception(
                "Failed to load agent names for run %s", run_id
            )
            return ""

        # Preserve order but deduplicate
        seen = set()
        ordered_names: List[str] = []
        for a in agents:
            name = a.get("agent_name")
            if not name or name in seen:
                continue
            seen.add(name)
            ordered_names.append(name)

        return ", ".join(ordered_names)

    options: Dict[str, int] = {}

    for r in runs:
        run_id = r["id"]
        ts = r["timestamp"]

        task_text = r.get("task_input", "") if isinstance(r, dict) else ""
        task_abbrev = abbreviate_task(task_text)

        agent_names = get_agent_names_for_run(run_id)

        label_parts = [f"Run {run_id}", str(ts)]
        if agent_names:
            label_parts.append(agent_names)
        if task_abbrev:
            label_parts.append(f"[{task_abbrev}]")

        label = " — ".join(label_parts)
        options[label] = run_id

    selected = st.selectbox(
        "Select Run",
        ["None"] + list(options.keys()),
    )

    if selected == "None":
        return

    run_id = options[selected]
    run, agents, metrics, tool_usages, agent_run_configs = cached_load_run_details(
        run_id
    )

    ts = run["timestamp"]
    task = run["task_input"]
    final = run["final_output"]
    final_is_json = run["final_is_json"]
    final_model = run["final_model"]

    # Run-level badge for strict schema exit
    if run.get("strict_schema_exit"):
        st.warning("This run exited early due to strict schema validation.")

    # Shared cost & latency rendering for stored metrics
    # Pass agent_run_configs so we can populate the Model column for historic runs.
    metrics_view = metrics_view_from_db_rows(metrics, agent_run_configs)
    if metrics_view:
        render_cost_latency_section(metrics_view, title_suffix="Stored")

    # ---------- Tools & Advanced Configuration (historic, per run snapshot) ----------
    agent_run_cfg_by_name: Dict[str, dict] = {}
    for cfg in agent_run_configs or []:
        agent_run_cfg_by_name[cfg["agent_name"]] = cfg

    tool_usages_by_agent: Dict[str, List[dict]] = {}
    for t in tool_usages or []:
        tool_usages_by_agent.setdefault(t["agent_name"], []).append(t)

    config_view = config_view_from_db_rows(agents, agent_run_configs or [])
    render_agent_config_section(
        config_view,
        tool_usages_by_agent,
        title_suffix="stored",
        is_historic=True,
    )

    # Per-call Tool Usage Overview for stored runs
    st.markdown("---")
    st.subheader("Tool Usage Overview (per call, stored)")
    df_calls_stored = build_tool_calls_overview(
        tool_usages_by_agent,
        is_historic=True,
    )
    if not df_calls_stored.empty:
        st.dataframe(df_calls_stored, width="stretch")
    else:
        st.info("No tool calls recorded for this run.")

    st.markdown("---")
    st.subheader(f"Run {run_id}")
    st.code(task)

    header = "Final Output"
    if final_model:
        header += f" · {final_model}"

    # Final output viewer
    from multi_agent_dashboard.ui.run_mode import render_output_block

    render_output_block(
        header,
        final,
        is_json_hint=bool(final_is_json),
        key_prefix=f"hist_run_{run_id}_final",
    )

    # Per-agent outputs
    for a in agents:
        name = a["agent_name"]
        output = a["output"]
        is_json = a["is_json"]
        model = a["model"]
        header = f"{name}"
        if model:
            header += f" · {model}"

        render_output_block(
            header,
            output,
            is_json_hint=bool(is_json),
            key_prefix=f"hist_run_{run_id}_{name}",
        )

    # ---------- Build per-agent details for export ----------
    metrics_by_agent: Dict[str, dict] = {}
    for m in metrics or []:
        agent_name = m.get("agent_name")
        if agent_name:
            m_copy = dict(m)
            m_copy.pop("agent_name", None)
            metrics_by_agent[agent_name] = m_copy

    tool_usages_export_by_agent: Dict[str, List[dict]] = {}
    for t in tool_usages or []:
        name = t.get("agent_name")
        if not name:
            continue
        t_copy = dict(t)
        t_copy.pop("agent_name", None)
        tool_usages_export_by_agent.setdefault(name, []).append(t_copy)

    agent_run_cfg_by_name = {cfg["agent_name"]: cfg for cfg in agent_run_configs or []}

    export_agents: Dict[str, dict] = {}

    for a in agents:
        name = a["agent_name"]

        agent_output = {
            "output": a["output"],
            "is_json": bool(a["is_json"]),
            "model": a["model"],
        }

        cfg = agent_run_cfg_by_name.get(name, {})
        tools_json = parse_json_field(cfg.get("tools_json"), {})
        tools_cfg_json = parse_json_field(
            cfg.get("tools_config_json"), {}
        )
        reasoning_cfg_json = parse_json_field(
            cfg.get("reasoning_config_json"), {}
        )
        extra_cfg_json = parse_json_field(
            cfg.get("extra_config_json"), {}
        )
        provider_feats = parse_json_field(cfg.get("provider_features_json"), {})

        # Extract allowed domains from the low-level stored tools_config_json (if present)
        allowed_domains = None
        try:
            # helper from view_models normally does this; inline here for export completeness
            if isinstance(tools_cfg_json, dict):
                tools_low = tools_cfg_json.get("tools")
                if isinstance(tools_low, list):
                    for tcfg in tools_low:
                        if not isinstance(tcfg, dict):
                            continue
                        if tcfg.get("type") != "web_search":
                            continue
                        filters = tcfg.get("filters") or {}
                        if isinstance(filters, dict) and "allowed_domains" in filters:
                            allowed = filters["allowed_domains"]
                            if isinstance(allowed, list):
                                allowed_domains = [str(d) for d in allowed]
                            else:
                                allowed_domains = [str(allowed)]
                            break
        except Exception:
            allowed_domains = None

        tools_snapshot = {
            "enabled": bool(tools_json.get("enabled")),
            "tools": tools_json.get("tools") or [],
        }
        if allowed_domains:
            tools_snapshot["allowed_domains"] = allowed_domains

        # Provider host/name convenience fields (explicit)
        endpoint = cfg.get("endpoint") or None
        provider_id = cfg.get("provider_id") or None

        agent_config = {
            "model": cfg.get("model") or a.get("model") or "unknown",
            "role": cfg.get("role") or "–",
            "tools": tools_snapshot,
            "reasoning": {
                "effort": cfg.get("reasoning_effort") or "default",
                "summary": cfg.get("reasoning_summary") or "none",
            },
            # Explicitly expose prompt templates from the per-run snapshot
            "prompt_template": cfg.get("prompt_template") or None,
            "system_prompt_template": cfg.get("system_prompt_template") or None,
            "raw": {
                "tools_json": tools_json or None,
                "tools_config_json": tools_cfg_json or None,
                "reasoning_config_json": reasoning_cfg_json or None,
                "extra_config_json": extra_cfg_json or None,
            },
            # Provider snapshot (captured at run time) including friendly name and host
            "provider": {
                "provider_id": provider_id or None,
                "provider_name": _provider_friendly_name(provider_id),
                "model_class": cfg.get("model_class") or None,
                "endpoint": endpoint or None,
                "host": _parse_endpoint_host(endpoint),
                "use_responses_api": bool(cfg.get("use_responses_api")),
                "provider_features": provider_feats or None,
            },
            "schema_validation_failed": bool(a.get("schema_validation_failed")),
            "strict_schema_validation": bool(cfg.get("strict_schema_validation")),
        }

        # Also explicitly expose content_blocks and instrumentation events if present in extra_config_json:
        try:
            if isinstance(extra_cfg_json, dict):
                if "content_blocks" in extra_cfg_json:
                    agent_config["content_blocks"] = extra_cfg_json.get("content_blocks")
                if "instrumentation_events" in extra_cfg_json:
                    agent_config["instrumentation_events"] = extra_cfg_json.get("instrumentation_events")
                if "structured_response" in extra_cfg_json:
                    agent_config["structured_response"] = extra_cfg_json.get("structured_response")
        except Exception:
            # Non-fatal: leave as-is if parsing/extraction fails
            pass

        export_agents[name] = {
            "output": agent_output,
            "metrics": metrics_by_agent.get(name) or None,
            "config": agent_config,
            "tool_usages": tool_usages_export_by_agent.get(name) or [],
        }

    # Totals for export (match earlier summary; recompute to avoid dependency)
    total_latency = sum((m.get("latency") or 0.0) for m in metrics or [])
    total_cost = sum((m.get("cost") or 0.0) for m in metrics or [])
    total_input_cost = sum((m.get("input_cost") or 0.0) for m in metrics or [])
    total_output_cost = sum((m.get("output_cost") or 0.0) for m in metrics or [])

    export = {
        "run_id": run_id,
        "timestamp": ts,
        "pipeline_summary": {
            "total_latency": round(total_latency, 5),
            "total_input_cost": round(total_input_cost, 6),
            "total_output_cost": round(total_output_cost, 6),
            "total_cost": round(total_cost, 6),
        },
        "strict_schema_exit": bool(run.get("strict_schema_exit")),
        "task_input": task,
        "final_output": {
            "output": final,
            "is_json": bool(final_is_json),
            "model": final_model,
        },
        "agents": export_agents,
    }

    st.download_button(
        "Download Run as JSON",
        data=__import__("json").dumps(export, indent=2),
        file_name=f"run_{run_id}.json",
        mime="application/json",
    )
