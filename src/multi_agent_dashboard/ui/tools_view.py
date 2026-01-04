# ui/tools_view.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from multi_agent_dashboard.ui.utils import parse_json_field
from multi_agent_dashboard.ui.view_models import AgentConfigView


def _extract_allowed_domains_from_tool_entry(
    entry: dict,
    is_historic: bool,
) -> List[str]:
    """
    Normalize extraction of allowed_domains from a tool usage entry.
    Handles both live (action dict) and historic (args_json) forms.
    """
    domains: List[str] = []

    if not is_historic:
        action = entry.get("action") or {}
        if not isinstance(action, dict):
            return domains
    else:
        args = parse_json_field(entry.get("args_json"), {})
        action = args.get("action") if isinstance(args, dict) else None
        if not isinstance(action, dict):
            return domains

    filters = action.get("filters") or {}
    if not isinstance(filters, dict):
        return domains

    allowed = filters.get("allowed_domains")
    if isinstance(allowed, list):
        domains.extend(str(d) for d in allowed)
    elif allowed:
        domains.append(str(allowed))

    return domains


def render_agent_config_section(
    config_view: List[AgentConfigView],
    tool_usages_by_agent: Dict[str, List[dict]],
    title_suffix: str = "",
    *,
    is_historic: bool,
):
    """
    Shared renderer for per-agent configuration + tool usage.

    Differences between live vs historic data (e.g. raw JSON configs)
    are handled via the is_historic flag and the presence of raw_* fields.
    """
    if not config_view:
        st.info("No agent configuration available.")
        return

    header = "Per-Agent Configuration"
    if title_suffix:
        header += f" ({title_suffix})"
    st.subheader(header)

    # Build a lookup for per-agent configured allowed domains
    configured_domains_by_agent: Dict[str, Optional[List[str]]] = {
        cfg.agent_name: cfg.web_search_allowed_domains
        for cfg in config_view
    }

    for cfg in config_view:
        with st.expander(f"{cfg.agent_name} — Configuration", expanded=False):
            # Model & role
            st.markdown(f"**Model:** `{cfg.model}`")
            st.markdown(f"**Role:** {cfg.role}")

            st.markdown("**System prompt (system/developer role):**")
            if cfg.system_prompt_template:
                st.code(cfg.system_prompt_template, language=None)
            else:
                st.markdown("–")

            # --- Prompts (user & system) ---
            st.markdown("**User prompt (prompt_template):**")
            if cfg.prompt_template:
                # Use st.code to preserve formatting and allow wrapping
                st.code(cfg.prompt_template, language=None)
            else:
                st.markdown("–")

            # Tools summary
            st.markdown(
                f"**Tool calling enabled:** {'Yes' if cfg.tools_enabled else 'No'}"
            )
            st.markdown(
                f"**Tools:** {', '.join(cfg.tools) if cfg.tools else '–'}"
            )

            if "web_search" in cfg.tools:
                if cfg.web_search_allowed_domains:
                    st.markdown("**Allowed domains for `web_search`:**")
                    st.code("\n".join(cfg.web_search_allowed_domains))
                else:
                    st.markdown(
                        "**Allowed domains for `web_search`:** not restricted (any domain)"
                    )

            st.markdown(
                f"**Reasoning effort:** {cfg.reasoning_effort}"
            )
            st.markdown(
                f"**Reasoning summary:** {cfg.reasoning_summary}"
            )

            # Historic runs can expose stored raw JSON configs
            if is_historic:
                if cfg.raw_reasoning_config:
                    with st.expander(
                        "Full reasoning configuration (raw JSON)", expanded=False
                    ):
                        st.json(cfg.raw_reasoning_config)
                if cfg.raw_tools_config:
                    with st.expander(
                        "Low-level tools configuration (raw JSON)", expanded=False
                    ):
                        st.json(cfg.raw_tools_config)
                if cfg.raw_extra_config:
                    with st.expander(
                        "Extra config (raw JSON)", expanded=False
                    ):
                        st.json(cfg.raw_extra_config)

    # Tool usage section (shared)
    st.markdown("---")
    usage_header = "Tool Usage"
    if title_suffix:
        usage_header += f" ({title_suffix})"
    st.subheader(usage_header)

    if not any(tool_usages_by_agent.values()):
        st.info("No tools were used in this run.")
        return

    # Detailed per-agent tool calls
    for agent_name, entries in tool_usages_by_agent.items():
        if not entries:
            continue

        with st.expander(f"{agent_name} — Tool calls", expanded=False):
            for u in entries:
                tool_type = u.get("tool_type") or "unknown"
                # Live result uses "id", historical uses "tool_call_id"
                call_id = (
                    u.get("id")
                    if "id" in u
                    else u.get("tool_call_id") or "n/a"
                )
                st.markdown(
                    f"- **Tool:** `{tool_type}` · **Call ID:** `{call_id}`"
                )

                # Live result: action dict; historical: args_json
                if not is_historic:
                    action = u.get("action") or {}
                    if action:
                        if tool_type == "web_search":
                            allowed_domains = _extract_allowed_domains_from_tool_entry(
                                u, is_historic=False
                            )
                            if allowed_domains:
                                st.markdown("  - Allowed domains for this call:")
                                st.code("\n".join(allowed_domains))
                        st.json(action, expanded=False)
                else:
                    args = parse_json_field(u.get("args_json"), {})
                    if args:
                        if tool_type == "web_search":
                            allowed_domains = _extract_allowed_domains_from_tool_entry(
                                u, is_historic=True
                            )
                            if allowed_domains:
                                st.markdown("  - Allowed domains for this call:")
                                st.code("\n".join(allowed_domains))
                        st.json(args, expanded=False)

    # Compact overview
    st.markdown("---")
    overview_header = "Tool Usage Overview"
    if title_suffix:
        overview_header += f" ({title_suffix})"
    st.subheader(overview_header)

    rows_overview: List[Dict[str, Any]] = []

    for agent_name, entries in tool_usages_by_agent.items():
        if not entries:
            continue

        counts: Dict[str, int] = {}
        has_web_search = False

        for e in entries:
            ttype = e.get("tool_type") or "unknown"
            counts[ttype] = counts.get(ttype, 0) + 1
            if ttype == "web_search":
                has_web_search = True

        row: Dict[str, Any] = {
            "Agent": agent_name,
            "Tools": ", ".join(f"{k}×{v}" for k, v in counts.items()),
        }

        if has_web_search:
            domains: set = set()
            for e in entries:
                if e.get("tool_type") != "web_search":
                    continue
                for d in _extract_allowed_domains_from_tool_entry(
                    e, is_historic=is_historic
                ):
                    domains.add(d)

            if domains:
                # We found allowed domains directly on the tool calls
                row["Allowed Domains (web_search)"] = ", ".join(sorted(domains))
            else:
                # fall back to configured per-agent allowed domains
                cfg_domains = configured_domains_by_agent.get(agent_name) or []
                if cfg_domains:
                    row["Allowed Domains (web_search)"] = ", ".join(cfg_domains)
                else:
                    row["Allowed Domains (web_search)"] = "–"

        rows_overview.append(row)

    if rows_overview:
        st.dataframe(pd.DataFrame(rows_overview), width="stretch")
    else:
        st.info("No tool calls recorded.")


def build_tool_calls_overview(
    tool_usages_by_agent: Dict[str, List[dict]],
    *,
    is_historic: bool,
) -> pd.DataFrame:
    """
    Build a flat per-call overview table for tools.

    Columns:
      - Agent
      - Tool
      - Call ID
      - Args (compact)
    """
    rows: List[Dict[str, Any]] = []

    for agent_name, entries in tool_usages_by_agent.items():
        for e in entries:
            tool_type = e.get("tool_type") or "unknown"

            # Normalize call id: live uses "id"; historic uses "tool_call_id"
            call_id = e.get("id") or e.get("tool_call_id") or "n/a"

            # Build a compact args view
            args: Dict[str, Any] = {}

            if not is_historic:
                # Live run: action dict is already present
                action = e.get("action") or {}
                if isinstance(action, dict):
                    if tool_type == "web_search":
                        action_type = action.get("type")

                        if action_type == "search":
                            if "query" in action:
                                args["query"] = action["query"]
                            filters = action.get("filters") or {}
                            if isinstance(filters, dict) and "allowed_domains" in filters:
                                args["allowed_domains"] = filters["allowed_domains"]

                        elif action_type == "open_page":
                            if "url" in action:
                                args["url"] = action["url"]
                            args["type"] = "open_page"

                        if not args:
                            args = {
                                k: v for k, v in action.items() if k != "sources"
                            }
                    else:
                        args = {
                            k: v for k, v in action.items() if k != "sources"
                        }
            else:
                # Historic: args_json contains the original tool call
                raw_args = parse_json_field(e.get("args_json"), {})
                if isinstance(raw_args, dict):
                    if tool_type == "web_search":
                        action = raw_args.get("action") or {}
                        if isinstance(action, dict):
                            action_type = action.get("type")
                            if action_type == "search":
                                if "query" in action:
                                    args["query"] = action["query"]
                                filters = action.get("filters") or {}
                                if isinstance(filters, dict) and "allowed_domains" in filters:
                                    args["allowed_domains"] = filters["allowed_domains"]
                            elif action_type == "open_page":
                                if "url" in action:
                                    args["url"] = action["url"]
                                args["type"] = "open_page"
                            if not args:
                                args = {
                                    k: v for k, v in action.items() if k != "sources"
                                }
                        else:
                            # fallback: show raw args without sources
                            args = {
                                k: v for k, v in raw_args.items() if k != "sources"
                            }
                    else:
                        args = {
                            k: v for k, v in raw_args.items() if k != "sources"
                        }

            rows.append(
                {
                    "Agent": agent_name,
                    "Tool": tool_type,
                    "Call ID": call_id,
                    "Args": args,
                }
            )

    return pd.DataFrame(rows)
