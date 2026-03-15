# ui/tools_view.py
"""
Tool configuration and usage analysis module for the Multi-Agent Dashboard.

This module provides comprehensive visualization and analysis components for
agent tool usage, including configuration display, per-call details, and
domain filtering analysis. It handles both live execution results and
historical run data with consistent presentation.

Key responsibilities:
- Render detailed agent configuration panels with tool settings
- Display per-agent tool usage with call details and arguments
- Analyze and visualize web search domain filtering
- Provide tool usage overview tables and summaries
- Handle differences between live and historical data formats

Architecture:
- Uses Streamlit expanders for detailed configuration panels
- Integrates with view models from `view_models.py` for configuration data
- Provides specialized handling for web search tools and domain filtering
- Supports both current run results and historical run data

Visualization Components:
    - Agent configuration panels: Model settings, tools, reasoning, prompts
    - Tool usage details: Per-call arguments and domain filtering
    - Tool usage overview: Compact tables with call counts and domains
    - Domain filtering analysis: Configured vs actual usage patterns

Usage:
    # Render agent configuration and tool usage
    >>> render_agent_config_section(config_view, tool_usages_by_agent, 
    ...                             title_suffix="stored", is_historic=True)
    
    # Build tool calls overview DataFrame
    >>> df = build_tool_calls_overview(tool_usages_by_agent, is_historic=False)

Dependencies:
    - `streamlit`: UI components and rendering
    - `pandas`: DataFrame manipulation for tabular display
    - `view_models.py`: Agent configuration view models
    - `utils.py`: JSON parsing utilities for historical data
"""
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
    Supports web_search (filters.allowed_domains) and web_search_ddg (domain_filter).
    """
    domains: List[str] = []

    if not is_historic:
        action = entry.get("action") or {}
        if not isinstance(action, dict):
            return domains
    else:
        args = parse_json_field(entry.get("args_json"), {})
        # For web_search, args contains {"action": {...}}; for web_search_ddg, args is the tool call directly
        action = args.get("action") if isinstance(args, dict) else None
        if not isinstance(action, dict):
            # If no action key, treat args as the action (for function-calling tools)
            action = args if isinstance(args, dict) else {}
        if not isinstance(action, dict):
            return domains

    # Extract from filters.allowed_domains (web_search)
    filters = action.get("filters") or {}
    if isinstance(filters, dict):
        allowed = filters.get("allowed_domains")
        if isinstance(allowed, list):
            domains.extend(str(d) for d in allowed)
        elif allowed:
            domains.append(str(allowed))
    
    # Extract from domain_filter parameter (web_search_ddg)
    domain_filter = action.get("domain_filter")
    if isinstance(domain_filter, list):
        domains.extend(str(d) for d in domain_filter)
    elif domain_filter:
        domains.append(str(domain_filter))
    
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
    def _format_token_limit(value: Optional[int]) -> str:
        """Format token limit for display: None → '–', 0 → 'no limit', else formatted number."""
        if value is None:
            return "–"
        if value == 0:
            return "no limit"
        return f"{value:,}"
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
        title_badges = []
        if cfg.strict_schema_validation:
            title_badges.append("[strict schema]")
        if cfg.schema_validation_failed:
            title_badges.append("[schema failed]")
        badge_suffix = f" {' '.join(title_badges)}" if title_badges else ""

        with st.expander(f"{cfg.agent_name} — Configuration{badge_suffix}", expanded=False):
            # Structured output & validation status
            st.subheader("Structured output & validation status")
            st.markdown(f"Strict schema validation: `{'on' if cfg.strict_schema_validation else 'off'}`")
            st.markdown(f"Structured output: `{'on' if cfg.structured_output_enabled else 'off'}`")

            if cfg.structured_output_enabled:
                if cfg.schema_validation_failed:
                    st.warning("Schema validation failed", icon=":material/warning:")
                else:
                    st.success("Schema validation passed", icon=":material/check:")

                # Show configured schema (and note if malformed)
                schema_display = "Not configured"
                schema_malformed = False
                parsed_schema = None
                if cfg.schema_json:
                    schema_display = cfg.schema_json
                    try:
                        import json
                        parsed_schema = json.loads(cfg.schema_json)
                        if not isinstance(parsed_schema, dict) or len(parsed_schema) == 0:
                            schema_malformed = True
                    except Exception:
                        schema_malformed = True

                st.markdown("**Configured schema:**")
                if parsed_schema and not schema_malformed:
                    st.json(parsed_schema)
                else:
                    st.code(schema_display or "Not configured", language="json")
                    if schema_malformed:
                        st.caption(":warning: Schema appears malformed or empty.")
            st.divider()

            st.subheader("Model and Provider settings")
            # Model & role
            st.markdown(f"**Model:** `{cfg.model}`")
            st.markdown(f"**Role:** {cfg.role}")

            # Provider snapshot (explicit)
            # st.markdown("**Provider snapshot:**")
            st.markdown(f"**Provider:** `{cfg.provider_id or '–'}`")
            st.markdown(f"**Provider model class:** `{cfg.model_class or '–'}`")
            st.markdown(f"**Endpoint:** `{cfg.endpoint or '–'}`")
            st.markdown(f"**Use Responses API:** `{'Yes' if cfg.use_responses_api else 'No'}`")
            # Temperature and output token limits
            temp_display = f"`{cfg.temperature:.2f}`" if cfg.temperature is not None else "–"
            st.markdown(f"**Temperature:** {temp_display}")
            # Max output tokens
            max_output_display = _format_token_limit(cfg.max_output)
            has_effective = cfg.max_output_effective is not None and cfg.max_output_effective != cfg.max_output
            if has_effective:
                suffix = f" (effective: `{_format_token_limit(cfg.max_output_effective)}`)"
            else:
                suffix = ""
            st.markdown(f"**Max output tokens:** `{max_output_display}`{suffix}")
            st.divider()

            st.subheader("Prompt templates")
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
            st.divider()

            st.subheader("Advanced configuration")
            # Tools summary
            st.markdown(
                f"**Tool calling enabled:** `{'Yes' if cfg.tools_enabled else 'No'}`"
            )
            st.markdown(
                f"**Tools:** `{', '.join(cfg.tools) if cfg.tools else '–'}`"
            )

            if "web_search" in cfg.tools:
                if cfg.web_search_allowed_domains:
                    st.markdown("**Allowed domains for `web_search`:**")
                    st.code("\n".join(cfg.web_search_allowed_domains))
                else:
                    st.markdown(
                        "**Allowed domains for `web_search`:** not restricted (any domain)"
                    )

            if "web_search_ddg" in cfg.tools:
                if cfg.web_search_allowed_domains:
                    st.markdown("**Allowed domains for `web_search_ddg`:**")
                    st.code("\n".join(cfg.web_search_allowed_domains))
                else:
                    st.markdown(
                        "**Allowed domains for `web_search_ddg`:** not restricted (any domain)"
                    )


            st.markdown(
                f"**Reasoning effort:** `{cfg.reasoning_effort}`"
            )
            st.markdown(
                f"**Reasoning summary:** `{cfg.reasoning_summary}`"
            )

            # Provider features (derived or explicit)
            if cfg.provider_features:
                # Temporarily disabled: Enable when dynamic capability detection is implemented
                if not True:
                    st.markdown("**Provider features (derived / snapshot):** ")
                    try:
                        st.json(cfg.provider_features)
                    except Exception:
                        st.markdown(str(cfg.provider_features))



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
                                st.caption(
                                    "Historic note: stored tool call args are displayed as-is; provider enforcement may differ."
                                )
                        st.json(args, expanded=False)
                    if args:
                        if tool_type == "web_search_ddg":
                            allowed_domains = _extract_allowed_domains_from_tool_entry(
                                u, is_historic=True
                            )
                            if allowed_domains:
                                st.markdown("  - Allowed domains for this call:")
                                st.code("\n".join(allowed_domains))
                                st.caption(
                                    "Historic note: stored tool call args are displayed as-is; provider enforcement may differ."
                                )
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
            if ttype in ("web_search", "web_search_ddg"):
                has_web_search = True

        row: Dict[str, Any] = {
            "Agent": agent_name,
            "Tools": ", ".join(f"{k}×{v}" for k, v in counts.items()),
        }

        if has_web_search:
            domains: set = set()
            for e in entries:
                if e.get("tool_type") not in ("web_search", "web_search_ddg"):
                    continue
                for d in _extract_allowed_domains_from_tool_entry(
                    e, is_historic=is_historic
                ):
                    domains.add(d)

            if domains:
                # We found allowed domains directly on the tool calls
                row["Allowed Domains (web search)"] = ", ".join(sorted(domains))
            else:
                # fall back to configured per-agent allowed domains
                cfg_domains = configured_domains_by_agent.get(agent_name) or []
                if cfg_domains:
                    row["Allowed Domains (web search)"] = ", ".join(cfg_domains)
                else:
                    row["Allowed Domains (web search)"] = "–"

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
