# ui/app.py
import json
import logging
import time
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

import graphviz
import pandas as pd
import streamlit as st
from openai import OpenAI  # type/factory only

from multi_agent_dashboard.config import (
    DB_FILE_PATH,
    LOG_FILE_PATH,
    OPENAI_API_KEY,
    UI_COLORS,
    configure_logging,
)
from multi_agent_dashboard.db.db import init_db
from multi_agent_dashboard.db.services import AgentService, PipelineService, RunService
from multi_agent_dashboard.engine import EngineResult, MultiAgentEngine
from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard.models import AgentSpec
from multi_agent_dashboard.ui.utils import parse_json_field
from multi_agent_dashboard.ui.view_models import (
    AgentMetricsView,
    AgentConfigView,
    summarize_agent_metrics,
    df_replace_none,
    metrics_view_from_engine_result,
    metrics_view_from_db_rows,
    config_view_from_db_rows,
)

# ======================================================
# STREAMLIT PAGE CONFIG (must be first Streamlit call)
# ======================================================

st.set_page_config(
    page_title="Multi-Agent Dashboard",
    layout="wide",
)

# ======================================================
# CONSTANTS / GLOBALS
# ======================================================

logger = logging.getLogger(__name__)

DB_PATH = DB_FILE_PATH

# Default agent colors
DEFAULT_COLOR = UI_COLORS["default"]["value"]
DEFAULT_SYMBOL = UI_COLORS["default"]["symbol"]

pipeline_svc = PipelineService(DB_PATH)
agent_svc = AgentService(DB_PATH)
run_svc = RunService(DB_PATH)

# Centralized mapping of log levels to colors and symbols (derived from UI_COLORS)
LOG_LEVEL_STYLES: Dict[str, Dict[str, str]] = {
    "DEBUG": {
        "color": UI_COLORS["grey"]["value"],
        "symbol": UI_COLORS["grey"]["symbol"],
    },
    "INFO": {
        "color": UI_COLORS["green"]["value"],
        "symbol": UI_COLORS["green"]["symbol"],
    },
    "WARNING": {
        "color": UI_COLORS["orange"]["value"],
        "symbol": UI_COLORS["orange"]["symbol"],
    },
    "ERROR": {
        "color": UI_COLORS["red"]["value"],
        "symbol": UI_COLORS["red"]["symbol"],
    },
    "CRITICAL": {
        "color": UI_COLORS["purple"]["value"],
        "symbol": UI_COLORS["purple"]["symbol"],
    },
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_TOTAL_SIZE = 20 * 1024 * 1024  # 20 MB
AD_HOC_PIPELINE_LABEL = "<Ad-hoc>"


# =======================
# DEFAULT AGENTS
# =======================
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


# ======================================================
# SMALL HELPERS
# ======================================================

def format_cost(value: Optional[float]) -> str:
    if value is None:
        return "–"
    return f"${value:.5f}"


def format_latency(value: Optional[float]) -> str:
    if value is None:
        return "–"
    return f"{value:.2f}s"


def inject_tag_style(scope: str = "global"):
    """
    Shared CSS injector for tag styling.

    scope = "global"  -> all tags
    scope = "sidebar" -> sidebar tags only
    """
    if scope == "sidebar":
        selector = 'section[data-testid="stSidebar"] span[data-baseweb="tag"]'
    else:
        selector = 'span[data-baseweb="tag"]'
    st.markdown(
        f"""
        <style>
        {selector} {{
            background-color: #55575b !important;
            color: white !important;
        }}
        {selector}:hover {{
            background-color: #41454b !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_tag_style_for_sidebar():
    """Backward-compatible wrapper for sidebar tag styling."""
    inject_tag_style("sidebar")


def inject_global_tag_style():
    """Backward-compatible wrapper for global tag styling."""
    inject_tag_style("global")


# =======================
# METRICS / CONFIG VIEW MODELS & HELPERS
# =======================

# (Imported from multi_agent_dashboard.ui.view_models)
# AgentMetricsView
# AgentConfigView
# summarize_agent_metrics
# df_replace_none
# metrics_view_from_engine_result
# metrics_view_from_db_rows
# config_view_from_db_rows


def render_cost_latency_section(
    metrics_view: List[AgentMetricsView],
    title_suffix: str = "",
):
    """
    Shared renderer for cost & latency, used for both current and historical runs.
    """
    if not metrics_view:
        st.info("No metrics available.")
        return

    # Convert back into the generic metrics mapping expected by summarize_agent_metrics
    metrics_map: Dict[str, dict] = {
        m.agent_name: {
            "model": m.model,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "latency": m.latency,
            "input_cost": m.input_cost,
            "output_cost": m.output_cost,
            "cost": m.total_cost,
        }
        for m in metrics_view
    }

    summary, df = summarize_agent_metrics(metrics_map)

    header = "Cost & Latency"
    if title_suffix:
        header += f" ({title_suffix})"
    st.subheader(header)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Latency", format_latency(summary["total_latency"]))
    with col2:
        st.metric("Total Input Cost", format_cost(summary["total_input_cost"]))
    with col3:
        st.metric("Total Output Cost", format_cost(summary["total_output_cost"]))
    with col4:
        st.metric("Total Cost", format_cost(summary["total_cost"]))

    st.markdown("---")
    st.subheader("Per-Agent Breakdown")
    st.dataframe(df_replace_none(df), width="stretch")


def _extract_allowed_domains_from_tool_entry(
    entry: dict,
    is_historic: bool,
) -> List[str]:
    """
    Normalize extraction of allowed_domains from a tool usage entry.
    Handles both live (action dict) and historic (args_json) forms.
    """
    from multi_agent_dashboard.ui.utils import parse_json_field as _pj

    domains: List[str] = []

    if not is_historic:
        action = entry.get("action") or {}
        if not isinstance(action, dict):
            return domains
    else:
        args = _pj(entry.get("args_json"), {})
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
            st.markdown(f"**Model:** `{cfg.model}`")
            st.markdown(f"**Role:** {cfg.role}")

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


# =======================
# CACHED WRAPPERS
# =======================

@st.cache_data(ttl=60)
def cached_load_agents():
    return agent_svc.list_agents()


@st.cache_data(ttl=60)
def cached_load_pipelines():
    return pipeline_svc.list_pipelines()


@st.cache_data(ttl=30)
def cached_load_runs():
    return run_svc.list_runs()


@st.cache_data(ttl=30)
def cached_load_run_details(run_id: int):
    return run_svc.get_run_details(run_id)


@st.cache_data(ttl=60)
def cached_load_prompt_versions(agent_name: str):
    return agent_svc.load_prompt_versions(agent_name)


# =======================
# LOG HANDLER CLASS
# =======================

class StreamlitLogHandler(logging.Handler):
    """
    Logging handler that stores recent log records in Streamlit session_state.
    """

    def __init__(self, capacity: int = 500):
        super().__init__()
        self.capacity = capacity

    def emit(self, record: logging.LogRecord):
        try:
            # Only touch Streamlit from the main thread to avoid
            # "missing ScriptRunContext" warnings.
            if threading.current_thread() is not threading.main_thread():
                return

            entry = {
                "time": (
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(record.created),
                    )
                    + f",{int(record.msecs):03d}"
                ),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }

            logs = st.session_state.setdefault(
                "_log_buffer",
                deque(maxlen=self.capacity),
            )
            logs.append(entry)
        except Exception:
            # Never let logging break the app
            pass


def load_historic_logs_into_buffer(
    log_path: Path,
    capacity: int = 500,
    session_key: str = "_log_buffer",
):
    """
    Load existing log file lines into the Streamlit log buffer on first app start.
    """
    if not log_path.exists():
        return

    # Avoid reloading if buffer already initialized (e.g., rerun)
    if session_key in st.session_state and st.session_state[session_key]:
        return

    buf = deque(maxlen=capacity)

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue

                # Expected format:
                # "YYYY-MM-DD HH:MM:SS,mmm [LEVEL] logger.name: message"
                time_part = ""
                level = "INFO"
                logger_name = ""
                message = line

                try:
                    ts_and_rest = line.split(" [", 1)
                    time_part = ts_and_rest[0].strip()
                    rest = ts_and_rest[1] if len(ts_and_rest) > 1 else ""

                    level_and_rest = rest.split("]", 1)
                    level = level_and_rest[0].strip("[]") or "INFO"
                    rest2 = level_and_rest[1].strip() if len(level_and_rest) > 1 else ""

                    logger_and_msg = rest2.split(":", 1)
                    logger_name = logger_and_msg[0].strip() if logger_and_msg else ""
                    message = logger_and_msg[1].strip() if len(logger_and_msg) > 1 else rest2
                except Exception:
                    # If parsing fails, just store the raw line as message
                    pass

                entry = {
                    "time": time_part,
                    "level": level,
                    "logger": logger_name,
                    "message": message,
                }
                buf.append(entry)
    except Exception:
        # Never let log loading break the UI
        pass

    st.session_state[session_key] = buf


# =======================
# CACHE INVALIDATION HELPERS
# =======================

def invalidate_caches(*names: str):
    """
    Centralized cache invalidation helper.
    Allowed names:
      - agents
      - prompt_versions
      - pipelines
      - runs
      - run_details
      - all
    """
    if "all" in names:
        cached_load_agents.clear()
        cached_load_prompt_versions.clear()
        cached_load_pipelines.clear()
        cached_load_runs.clear()
        cached_load_run_details.clear()
        return

    for name in names:
        if name == "agents":
            cached_load_agents.clear()
        elif name == "prompt_versions":
            cached_load_prompt_versions.clear()
        elif name == "pipelines":
            cached_load_pipelines.clear()
        elif name == "runs":
            cached_load_runs.clear()
        elif name == "run_details":
            cached_load_run_details.clear()


def invalidate_agents():
    """Invalidate caches related to agents and prompt versions."""
    invalidate_caches("agents", "prompt_versions")


def invalidate_pipelines():
    """Invalidate caches related to pipelines."""
    invalidate_caches("pipelines")


def invalidate_runs():
    """Invalidate caches related to runs and run details."""
    invalidate_caches("runs", "run_details")


# =======================
# OpenAI client factory
# =======================

def create_openai_client(api_key: str):
    """Factory to create an OpenAI client. Allows tests to replace this factory or pass fake client."""
    return OpenAI(api_key=api_key)


# ======================================================
# SHARED HELPERS (USED BY APP_START AND UI)
# ======================================================

def pipeline_requires_files(engine, steps) -> bool:
    for name in steps:
        agent = engine.agents.get(name)
        if not agent:
            continue
        if "files" in agent.spec.input_vars:
            return True
    return False


def render_agent_graph(steps: List[str], agent_metrics: Optional[Dict[str, dict]] = None):
    dot = graphviz.Digraph()
    # Keep defaults for nodes; individual nodes will override `color`
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        color="#6baed6",
        fillcolor="#deebf7",
    )

    agent_metrics = agent_metrics or {}
    engine = st.session_state.engine

    for agent in steps:
        runtime = engine.agents.get(agent)

        # Default label (fallbacks if agent not in engine)
        role = runtime.spec.role if runtime else None
        symbol = getattr(runtime.spec, "symbol", DEFAULT_SYMBOL) if runtime else DEFAULT_SYMBOL
        color = getattr(runtime.spec, "color", DEFAULT_COLOR) if runtime else DEFAULT_COLOR

        # Prefix name with emoji/symbol
        base_label = f"{symbol} {agent}"
        if role:
            label = f"{base_label}\n({role})"
        else:
            label = base_label

        # Optionally annotate node with cost/latency
        m = agent_metrics.get(agent, {})
        extra = []
        if m.get("latency") is not None:
            extra.append(format_latency(m.get("latency")))
        if m.get("cost") is not None:
            extra.append(format_cost(m.get("cost")))
        if extra:
            label = f"{label}\n" + " | ".join(extra)

        # Use agent-specific color as border color
        dot.node(
            agent,
            label=label,
            color=color,        # border color
            fillcolor="#deebf7" # keep shared fill to stay readable
        )

    # Edges: label with downstream agent's metrics
    for i in range(len(steps) - 1):
        src = steps[i]
        dst = steps[i + 1]
        m = agent_metrics.get(dst, {})
        latency = m.get("latency")
        cost = m.get("cost")
        if latency is not None or cost is not None:
            edge_label = f"{format_latency(latency)} | {format_cost(cost)}"
        else:
            edge_label = "passes state →"
        dot.edge(src, dst, label=edge_label)

    return dot


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
            color=a.get("color") or DEFAULT_COLOR,
            symbol=a.get("symbol") or DEFAULT_SYMBOL,
            tools=a.get("tools") or {},
            reasoning_effort=a.get("reasoning_effort"),
            reasoning_summary=a.get("reasoning_summary"),
        )
        engine.add_agent(spec)


def get_agent_symbol_map() -> Dict[str, str]:
    """Return a mapping of agent_name -> symbol (with defaults applied)."""
    symbol_map: Dict[str, str] = {}

    # Prefer engine agents (already loaded with defaults)
    engine = st.session_state.get("engine")
    if engine and getattr(engine, "agents", None):
        for name, runtime in engine.agents.items():
            sym = getattr(runtime.spec, "symbol", None) or DEFAULT_SYMBOL
            symbol_map[name] = sym
        return symbol_map

    # Fallback to DB listing
    try:
        for a in cached_load_agents():
            symbol_map[a["agent_name"]] = a.get("symbol") or DEFAULT_SYMBOL
    except Exception:
        # If something goes wrong, default all to DEFAULT_SYMBOL
        pass

    return symbol_map


def export_pipeline_agents_as_json(pipeline_name: str, steps: List[str]) -> str:
    """
    Export all agents used in a pipeline as a JSON string.
    """
    engine = st.session_state.engine

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


# ======================================================
# INITIALIZE DB AND PREPARE ENGINE
# ======================================================

def bootstrap_default_agents(defaults: Dict[str, dict]):
    existing = cached_load_agents()
    if existing:
        return
    for name, data in defaults.items():
        agent_svc.save_agent(
            name,
            data.get("model", "gpt-4.1-nano"),
            data.get("prompt_template", ""),
            data.get("role", ""),
            data.get("input_vars", []),
            data.get("output_vars", []),
        )
        # also save a versioned prompt snapshot
        agent_svc.save_prompt_version(
            name,
            data.get("prompt_template", ""),
            metadata={"role": data.get("role", "")},
        )


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
    # Attach Streamlit log handler once per session
    if "_log_handler_attached" not in st.session_state:
        handler = StreamlitLogHandler(capacity=500)
        logging.getLogger().addHandler(handler)
        st.session_state["_log_handler_attached"] = True

        # Load historic logs into buffer on first attachment
        load_historic_logs_into_buffer(LOG_FILE_PATH, capacity=500)

    # Initialize DB and apply migrations
    init_db(DB_PATH)

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


configure_logging()

# Only start the app if we don't already have an engine in session state.
# This prevents double initialization during reruns.
if "engine" not in st.session_state:
    app_start()
else:
    # Ensure engine has up-to-date agents after code reloads
    reload_agents_into_engine()

# Initialize Ad-hoc pipeline state (empty on first app start)
if "adhoc_pipeline_steps" not in st.session_state:
    st.session_state.adhoc_pipeline_steps = []

# Also provide the conventional guard so running the script directly will bootstrap.
if __name__ == "__main__":
    if "engine" not in st.session_state:
        app_start()


# ======================================================
# GLOBAL UI STYLES
# ======================================================

st.markdown(
    """
    <style>
    /* Wrap code blocks */
    pre code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-x: auto !important;
    }

    /* Wrap Streamlit code blocks */
    .stCodeBlock pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }

    /* Wrap JSON output */
    .stJson pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }

    /* Wrap text areas */
    textarea {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }

    /* Prevent horizontal scroll everywhere */
    section.main {
        overflow-x: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================
# TOP-LEVEL UI
# ======================================================

st.title("🧠 Multi-Agent Pipeline Dashboard")
st.caption("Design pipelines, run tasks, inspect agent behavior, and manage prompt versions.")

MODE_RUN = "▶️ Run Pipeline"
MODE_AGENTS = "🧠 Manage Agents"
MODE_HISTORY = "📜 History"
MODE_LOGS = "🪵 Logs"

mode = st.radio(
    "Mode",
    [MODE_RUN, MODE_AGENTS, MODE_HISTORY, MODE_LOGS],
    horizontal=True,
)

strict_mode = st.sidebar.checkbox(
    "Strict output validation",
    value=False,
    help="Fail fast on agent output mismatches",
)

st.divider()


# ======================================================
# RUN MODE
# ======================================================

def render_run_sidebar() -> Tuple[
    str,
    List[str],
    str,
    bool,
    Optional[List[Dict[str, Any]]],
    Dict[str, List[str]],
    bool,
]:
    st.sidebar.header("Run Configuration")

    inject_tag_style_for_sidebar()

    # -------------------------
    # Load pipelines
    # -------------------------
    pipelines = cached_load_pipelines()
    pipeline_names = [p["pipeline_name"] for p in pipelines]

    selected_pipeline = st.sidebar.selectbox(
        "Pipeline",
        [AD_HOC_PIPELINE_LABEL] + pipeline_names,
    )

    # -------------------------
    # Task input
    # -------------------------
    task = st.sidebar.text_area(
        "Task",
        placeholder="Describe the task you want the agents to solve…",
        height=120,
    )

    # -------------------------
    # Resolve base steps
    # -------------------------
    engine = st.session_state.engine
    available_agents = list(engine.agents.keys())

    if selected_pipeline != AD_HOC_PIPELINE_LABEL:
        base_steps = next(
            (p["steps"] for p in pipelines if p["pipeline_name"] == selected_pipeline),
            [],
        )
    else:
        # Use stored Ad-hoc pipeline steps from session state
        base_steps = st.session_state.get("adhoc_pipeline_steps", [])

    # -------------------------
    # Agent selection (SOURCE OF TRUTH)
    # -------------------------
    agent_symbols = get_agent_symbol_map()

    def format_agent_label(name: str) -> str:
        symbol = agent_symbols.get(name, DEFAULT_SYMBOL)
        return f"{symbol} {name}"

    selected_steps = st.sidebar.multiselect(
        "Agents (execution order)",
        available_agents,
        default=base_steps,
        format_func=format_agent_label,
    )

    # Persist Ad-hoc steps so they survive pipeline switching & reruns
    if selected_pipeline == AD_HOC_PIPELINE_LABEL:
        st.session_state.adhoc_pipeline_steps = selected_steps

    # -------------------------
    # Detect file requirement (from selected steps!)
    # -------------------------
    requires_files = pipeline_requires_files(
        engine,
        selected_steps,
    )

    # -------------------------
    # File attachment section
    # -------------------------
    files_payload: Optional[List[Dict[str, Any]]] = None
    file_error = False

    if requires_files:
        with st.sidebar.expander("📎 Attach files", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload files",
                accept_multiple_files=True,
                type=["txt", "pdf", "csv", "md", "json", "log", "py", "sql", "patch"],
            )

            files_payload = []
            total_size = 0

            for f in uploaded_files or []:
                if f.size > MAX_FILE_SIZE:
                    st.error(f"{f.name} exceeds 5MB limit")
                    file_error = True
                    continue

                total_size += f.size
                if total_size > MAX_TOTAL_SIZE:
                    st.error("Total file size exceeds 20MB")
                    file_error = True
                    break

                files_payload.append({
                    "filename": f.name,
                    "content": f.read(),
                    "mime_type": f.type,
                })

            if file_error:
                files_payload = None

    # -------------------------
    # Detect whether any selected agent has web_search enabled
    # -------------------------
    def agent_uses_web_search(agent_name: str) -> bool:
        ag = engine.agents.get(agent_name)
        if not ag:
            return False
        tools_cfg = getattr(ag.spec, "tools", {}) or {}
        if not tools_cfg.get("enabled"):
            return False
        tools = tools_cfg.get("tools") or []
        return "web_search" in tools

    any_web_search = any(agent_uses_web_search(a) for a in selected_steps)

    # -------------------------
    # Websearch domain filter section (per agent)
    # -------------------------
    allowed_domains_by_agent: Dict[str, List[str]] = {}

    if any_web_search:
        st.sidebar.markdown("### 🔎 Web Search Domains")
        st.sidebar.caption(
            "Optionally limit web search to specific domains for each agent "
            "(one domain per line). Leave empty to allow any domain."
        )

        for agent_name in selected_steps:
            if not agent_uses_web_search(agent_name):
                continue

            with st.sidebar.expander(
                f"Web Search Domains — {agent_name}", expanded=False
            ):
                domains_text = st.text_area(
                    "Allowed domains (optional)",
                    value="",
                    placeholder="example.com\nanother-example.org",
                    height=80,
                    key=f"allowed_domains_{agent_name}",
                )
                domains = [
                    d.strip()
                    for d in domains_text.splitlines()
                    if d.strip()
                ]
                if domains:
                    allowed_domains_by_agent[agent_name] = domains

    # -------------------------
    # Run button
    # -------------------------
    run_clicked = st.sidebar.button(
        "🚀 Run Pipeline",
        width="stretch",
    )

    # -------------------------
    # Advanced: Pipeline editing
    # -------------------------
    with st.sidebar.expander("Advanced", expanded=False):
        name = st.text_input("Save as Pipeline")

        if st.button("💾 Save Pipeline"):
            pipeline_name = name.strip()
            if not pipeline_name:
                st.error("Pipeline name cannot be empty.")
            elif not selected_steps:
                st.error("Cannot save an empty pipeline. Select at least one agent.")
            else:
                try:
                    pipeline_svc.save_pipeline(pipeline_name, selected_steps)
                except Exception:
                    logger.exception("Failed to persist pipeline")
                    st.error("Failed to save pipeline to database")
                invalidate_pipelines()
                st.success("Pipeline saved")
                st.rerun()

        st.divider()

        if selected_pipeline != AD_HOC_PIPELINE_LABEL and selected_steps:
            agents_json = export_pipeline_agents_as_json(
                selected_pipeline,
                selected_steps,
            )

            st.download_button(
                label="⬇️ Download Pipeline Agents (JSON)",
                data=agents_json,
                file_name=f"{selected_pipeline}_agents.json",
                mime="application/json",
                width="stretch",
            )

    # -------------------------
    # Return values
    # -------------------------
    return (
        selected_pipeline,
        selected_steps,
        task,
        run_clicked,
        files_payload,
        allowed_domains_by_agent,
        file_error,
    )


def render_warnings(result: EngineResult):
    if not result.warnings:
        return

    st.warning("⚠️ Pipeline Warnings")
    for w in result.warnings:
        st.markdown(f"- {w}")


def render_output_block(
    label: str,
    text: str,
    *,
    is_json_hint: Optional[bool] = None,
    key_prefix: str,
):
    """
    Shared viewer: try JSON, else Markdown/Code toggle.
    """
    with st.expander(label, expanded=(label.lower().startswith("final output"))):
        if text is None:
            st.info("No output.")
            return

        if is_json_hint:
            try:
                st.json(json.loads(text))
                return
            except Exception:
                st.warning("⚠️ Marked as JSON but failed to parse; falling back to text view.")

        try:
            parsed = json.loads(text)
            st.json(parsed)
        except Exception:
            view = st.radio(
                "View as",
                ["Markdown", "Code"],
                horizontal=True,
                key=f"{key_prefix}_view",
            )
            if view == "Markdown":
                st.markdown(text)
            else:
                st.code(text)


def render_final_output(
    result: EngineResult,
    current_run_export: Optional[dict] = None,
):
    if not result.final_output:
        st.info("No output yet.")
        return

    render_output_block(
        "Final Output",
        result.final_output,
        key_prefix="run_final",
    )

    # Download button for the current run JSON export (if provided)
    if current_run_export is not None:
        st.download_button(
            "⬇️ Download This Run as JSON",
            data=json.dumps(current_run_export, indent=2),
            file_name="current_run.json",
            mime="application/json",
        )


def render_agent_outputs(result: EngineResult, steps: List[str]):
    for agent in steps:
        out = result.memory.get(agent, "")
        render_output_block(
            f"🔹 {agent}",
            out,
            key_prefix=f"run_agent_{agent}",
        )


def render_graph_tab(result: EngineResult, steps: List[str]):
    if not steps:
        st.info("No agents selected.")
        return
    st.graphviz_chart(render_agent_graph(steps, result.agent_metrics))


def render_compare_tab(result: EngineResult, steps: List[str]):
    if not steps:
        st.info("No agents selected.")
        return

    col1, col2 = st.columns(2)

    with col1:
        a1 = st.selectbox("Agent A", steps, key="cmp_a")

    with col2:
        a2 = st.selectbox("Agent B", steps, key="cmp_b")

    if a1 != a2:
        import difflib

        out1 = str(result.memory.get(a1, ""))
        out2 = str(result.memory.get(a2, ""))

        diff = difflib.unified_diff(
            out1.splitlines(),
            out2.splitlines(),
            fromfile=a1,
            tofile=a2,
            lineterm="",
        )

        st.code("\n".join(diff), language="diff")


def render_cost_latency_tab(result: EngineResult, steps: List[str]):
    metrics_view = metrics_view_from_engine_result(result, steps)
    render_cost_latency_section(metrics_view, title_suffix="This run")

    # Tool usage overview moved to Tools & Advanced tab in the original UI.
    # To preserve behavior, we do not alter that separation here.


def render_tools_advanced_tab(result: EngineResult, steps: List[str]):
    engine = st.session_state.engine
    tool_usages = result.tool_usages or {}

    tool_usages_by_agent: Dict[str, List[dict]] = {
        name: tool_usages.get(name) or [] for name in steps
    }

    config_view = config_view_from_engine_result(result, steps)
    render_agent_config_section(
        config_view,
        tool_usages_by_agent,
        title_suffix="this run",
        is_historic=False,
    )

    # Optional: flat per-call overview table for this run
    st.markdown("---")
    st.subheader("Tool Usage Overview (per call, this run)")
    df_calls = build_tool_calls_overview(
        tool_usages_by_agent,
        is_historic=False,
    )
    if not df_calls.empty:
        st.dataframe(df_calls, width="stretch")
    else:
        st.info("No tool calls recorded.")


def config_view_from_engine_result(
    result: EngineResult,
    steps: List[str],
) -> List[AgentConfigView]:
    engine = st.session_state.engine
    views: List[AgentConfigView] = []

    allowed_domains_by_agent = result.state.get("allowed_domains_by_agent") or {}
    global_domains = result.state.get("allowed_domains")

    for name in steps:
        ag = engine.agents.get(name)
        if not ag:
            continue
        spec = ag.spec
        tools_cfg = spec.tools or {}
        enabled = bool(tools_cfg.get("enabled"))
        enabled_tools = tools_cfg.get("tools") or []

        allowed_domains_agent: Optional[List[str]] = None
        maybe = allowed_domains_by_agent.get(name)
        if isinstance(maybe, list) and maybe:
            allowed_domains_agent = maybe
        elif isinstance(global_domains, list) and global_domains:
            allowed_domains_agent = global_domains

        views.append(
            AgentConfigView(
                agent_name=name,
                model=spec.model,
                role=spec.role or "–",
                tools_enabled=enabled,
                tools=enabled_tools,
                web_search_allowed_domains=allowed_domains_agent,
                reasoning_effort=spec.reasoning_effort or "default",
                reasoning_summary=spec.reasoning_summary or "none",
            )
        )
    return views


def render_run_results(
    result: EngineResult,
    steps: List[str],
    current_run_export: Optional[dict] = None,
):
    tabs = st.tabs(
        [
            "🟢 Final Output",
            "⚠️ Warnings",
            "📁 Agent Outputs",
            "🧩 Graph",
            "🔍 Compare",
            "💲 Cost & Latency",
            "🛠 Tools & Advanced",
        ]
    )

    with tabs[0]:
        render_final_output(result, current_run_export=current_run_export)

    with tabs[1]:
        render_warnings(result)

    with tabs[2]:
        render_agent_outputs(result, steps)

    with tabs[3]:
        render_graph_tab(result, steps)

    with tabs[4]:
        render_compare_tab(result, steps)

    with tabs[5]:
        render_cost_latency_tab(result, steps)

    with tabs[6]:
        render_tools_advanced_tab(result, steps)


def render_run_mode():
    (
        pipeline,
        steps,
        task,
        run_clicked,
        files_payload,
        allowed_domains_by_agent,
        file_error,
    ) = render_run_sidebar()

    if run_clicked:
        if not steps:
            st.error("Select at least one agent before running the pipeline.")
            return

        if file_error:
            st.error("Fix file upload errors before running the pipeline.")
            return

        engine = st.session_state.engine

        # -----------------------
        # Local shared state for this run
        # (no Streamlit calls in the worker thread)
        # -----------------------
        progress_state = {
            "pct": 0,
            "agent": None,
        }
        result_container: Dict[str, Any] = {"result": None, "error": None}
        pipeline_done = False

        # -----------------------
        # Timer + progress UI
        # -----------------------
        st.session_state["pipeline_start_time"] = time.time()

        timer_placeholder = st.empty()
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def _format_elapsed(elapsed_seconds: float) -> str:
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            return f"{minutes:02d}:{seconds:02d}"

        def _update_timer():
            start = st.session_state.get("pipeline_start_time")
            if start is None:
                return
            elapsed = time.time() - start
            timer_placeholder.markdown(
                f"**Elapsed time:** {_format_elapsed(elapsed)}"
            )

        # Progress callback used by the engine:
        # only mutates local Python state; no Streamlit APIs
        def on_progress(pct: int, agent_name: Optional[str] = None):
            progress_state["pct"] = int(pct)
            progress_state["agent"] = agent_name

        engine.on_progress = on_progress

        # Worker function: runs the pipeline in a background thread
        def _run_engine():
            nonlocal pipeline_done
            try:
                result: EngineResult = engine.run_seq(
                    steps=steps,
                    initial_input=task,
                    strict=strict_mode,
                    files=files_payload if files_payload else None,
                    allowed_domains=allowed_domains_by_agent,
                )
                result_container["result"] = result
            except Exception as e:
                logger.exception("Pipeline execution failed")
                result_container["error"] = str(e)
            finally:
                pipeline_done = True

        # Start engine in a background thread
        worker = threading.Thread(target=_run_engine, daemon=True)
        worker.start()

        # Initial timer display at 0s
        _update_timer()

        # -----------------------
        # Main UI loop: while worker is alive, update timer + progress
        # -----------------------
        with st.spinner("Running pipeline…"):
            while not pipeline_done:
                pct = progress_state["pct"]
                agent_name = progress_state["agent"]

                # Progress bar
                progress_bar.progress(int(pct))

                # Progress text (restore original label)
                if agent_name:
                    progress_text.caption(f"Running {agent_name} — {pct}%")
                else:
                    progress_text.caption(f"Pipeline progress: {pct}%")

                # Timer
                _update_timer()

                # Small sleep to refresh ~5 times/sec without hammering UI
                time.sleep(0.2)

        # One final update when done
        progress_bar.progress(100)
        progress_text.caption("Pipeline completed ✅")
        _update_timer()

        # -----------------------
        # Handle error / success
        # -----------------------
        if result_container["error"]:
            st.error(f"Pipeline failed: {result_container['error']}")
            return

        result: EngineResult = result_container["result"]
        st.session_state.last_result = result
        st.session_state.last_steps = steps
        st.session_state.last_task = task

        # Persist run to DB
        agent_models = {
            name: runtime.spec.model
            for name, runtime in engine.agents.items()
        }

        final_model = (
            agent_models.get(result.final_agent)
            if result.final_agent
            else None
        )

        try:
            run_svc.save_run(
                task,
                result.final_output,
                result.memory,
                agent_models=agent_models,
                final_model=final_model,
                agent_configs=result.agent_configs,
                agent_metrics=result.agent_metrics,
                tool_usages=result.tool_usages,
            )
        except Exception:
            logger.exception("Failed to persist run")
            st.error("Run completed but failed to save to database")
        invalidate_runs()

        st.success("Pipeline completed!")

        # Inline warning banner
        if "__warnings__" in st.session_state.engine.memory:
            st.warning(
                f"{len(st.session_state.engine.memory['__warnings__'])} warning(s) occurred during execution."
            )

    # Always render last result (if any)
    if "last_result" in st.session_state:
        last_result: EngineResult = st.session_state.last_result
        last_steps: List[str] = st.session_state.last_steps
        last_task: str = st.session_state.get("last_task", "")

        # Build export for the current (most recent) run
        current_run_export = build_export_from_engine_result(
            last_result,
            last_steps,
            last_task,
        )

        render_run_results(
            last_result,
            last_steps,
            current_run_export=current_run_export,
        )


# ======================================================
# AGENT EDITOR MODE
# ======================================================

def render_agent_editor():
    st.header("🧠 Agent Editor")

    # -------------------------
    # Load agents from DB
    # -------------------------
    agents_raw = cached_load_agents()
    agents = [
        {
            "name": a["agent_name"],
            "model": a["model"],
            "role": a["role"],
            "prompt": a["prompt_template"],
            "input_vars": a["input_vars"],
            "output_vars": a["output_vars"],
            "color": a.get("color") or DEFAULT_COLOR,
            "symbol": a.get("symbol") or DEFAULT_SYMBOL,
            "tools": a.get("tools") or {},
            "reasoning_effort": a.get("reasoning_effort"),
            "reasoning_summary": a.get("reasoning_summary"),
        }
        for a in agents_raw
    ]
    names = [a["name"] for a in agents]

    # Map agent name -> symbol (with defaults)
    agent_symbols = get_agent_symbol_map()

    def format_agent_label(name: str) -> str:
        if name == "<New Agent>":
            return name
        symbol = agent_symbols.get(name, DEFAULT_SYMBOL)
        return f"{symbol} {name}"

    # -------------------------
    # Editor internal state
    # -------------------------
    if "agent_editor_state" not in st.session_state:
        st.session_state.agent_editor_state = {
            "selected_name": "<New Agent>",
            "name": "",
            "model": "gpt-4.1-nano",
            "role": "",
            "prompt": "",
            "input_vars": [],
            "output_vars": [],
            "color": DEFAULT_COLOR,
            "symbol": DEFAULT_SYMBOL,
            "tools": {"enabled": False, "tools": []},
            "reasoning_effort": None,
            "reasoning_summary": None,
        }
    state = st.session_state.agent_editor_state

    # Persistent flag to survive st.rerun
    if "agent_editor_state_changed_this_run" not in st.session_state:
        st.session_state.agent_editor_state_changed_this_run = False

    state_changed_this_run = st.session_state.agent_editor_state_changed_this_run

    # -------------------------
    # Agent selection
    # -------------------------
    options = ["<New Agent>"] + names

    current_selected_name = state.get("selected_name", "<New Agent>")
    if current_selected_name not in options:
        current_selected_name = "<New Agent>"
    current_index = options.index(current_selected_name)

    selected = st.selectbox(
        "Agent",
        options,
        index=current_index,
        key="agent_editor_selected_agent",
        format_func=format_agent_label,
    )

    # If user changed selection via the selectbox, populate editor fields
    if selected != state.get("selected_name"):
        state["selected_name"] = selected
        if selected == "<New Agent>":
            base_agent = {
                "name": "",
                "model": "gpt-4.1-nano",
                "role": "",
                "prompt": "",
                "input_vars": [],
                "output_vars": [],
                "color": DEFAULT_COLOR,
                "symbol": DEFAULT_SYMBOL,
                "tools": {"enabled": False, "tools": []},
                "reasoning_effort": None,
                "reasoning_summary": None,
            }
        else:
            base_agent = next(a for a in agents if a["name"] == selected)

        state.update(
            {
                "name": base_agent["name"],
                "model": base_agent["model"],
                "role": base_agent["role"],
                "prompt": base_agent["prompt"],
                "input_vars": base_agent["input_vars"],
                "output_vars": base_agent["output_vars"],
                "color": base_agent.get("color", DEFAULT_COLOR) or DEFAULT_COLOR,
                "symbol": base_agent.get("symbol", DEFAULT_SYMBOL)
                or DEFAULT_SYMBOL,
                "tools": base_agent.get("tools")
                or {"enabled": False, "tools": []},
                "reasoning_effort": base_agent.get("reasoning_effort"),
                "reasoning_summary": base_agent.get("reasoning_summary"),
            }
        )

        state_changed_this_run = True
        st.session_state.agent_editor_state_changed_this_run = True

    is_new = state.get("selected_name") == "<New Agent>"

    # -------------------------
    # Tabs
    # -------------------------
    tabs = st.tabs(
        [
            "1️⃣ Basics",
            "2️⃣ Prompt",
            "3️⃣ Inputs / Outputs",
            "⚙️ Advanced",
            "📚 Versions",
        ]
    )

    basics_tab, prompt_tab, io_tab, adv_tab, versions_tab = tabs

    # ----- Basics tab -----
    with basics_tab:
        name_val = st.text_input(
            "Name",
            value=state["name"],
        )
        model_val = st.text_input(
            "Model",
            value=state["model"],
        )
        role_val = st.text_input(
            "Role",
            value=state["role"],
        )

        st.markdown("### Appearance")

        # Prepare color options
        color_keys = list(UI_COLORS.keys())
        color_labels = {
            key: f"{UI_COLORS[key]['symbol']} {key.capitalize()}"
            for key in color_keys
        }

        def infer_color_key_from_hex(hex_value: str) -> str:
            for k, v in UI_COLORS.items():
                if v["value"].lower() == (hex_value or "").lower():
                    return k
            return "default"

        current_color_key = infer_color_key_from_hex(
            state.get("color", DEFAULT_COLOR)
        )
        if current_color_key not in color_labels:
            current_color_key = "default"

        prev_palette_key = state.get("_palette_key", current_color_key)

        selected_color_key = st.selectbox(
            "Base Color",
            options=color_keys,
            index=color_keys.index(current_color_key),
            format_func=lambda k: color_labels[k],
            help=(
                "Choose a base color; this sets both the default color and "
                "symbol for the agent. You can still override them below."
            ),
            key="agent_editor_base_color",
        )

        selected_palette = UI_COLORS[selected_color_key]
        palette_color_value = selected_palette["value"]
        palette_symbol_value = selected_palette["symbol"]

        # If user changed the base palette, immediately update color/symbol
        if selected_color_key != prev_palette_key:
            state["color"] = palette_color_value
            state["symbol"] = palette_symbol_value
            state["_palette_key"] = selected_color_key
        else:
            state["_palette_key"] = prev_palette_key

        color_val = st.color_picker(
            "Agent Color",
            value=state.get("color", palette_color_value) or palette_color_value,
            help="Override the color for this agent.",
        )

        symbol_val = st.text_input(
            "Agent Symbol (emoji or short text)",
            value=state.get("symbol", palette_symbol_value) or palette_symbol_value,
            help="Override the symbol for this agent.",
            max_chars=8,
        )

    # ----- Prompt tab -----
    with prompt_tab:
        prompt_val = st.text_area(
            "Prompt Template",
            height=400,
            value=state["prompt"],
        )

    # ----- Inputs / Outputs tab -----
    with io_tab:
        col1, col2 = st.columns(2)
        with col1:
            input_vars_val = st.text_area(
                "Input Variables (one per line)",
                value="\n".join(state["input_vars"]),
            )
        with col2:
            output_vars_val = st.text_area(
                "Output Variables (one per line)",
                value="\n".join(state["output_vars"]),
            )

    # ----- Versions tab -----
    with versions_tab:
        if not is_new:
            versions = cached_load_prompt_versions(state["name"])
            for v in versions:
                with st.expander(
                    f"Version {v['version']} — {v['created_at']}",
                ):
                    st.code(v["prompt"])
                    if v.get("metadata"):
                        st.json(v["metadata"])

    # ----- Advanced tab -----
    with adv_tab:
        st.markdown("### Tools")

        tools_state = state.get("tools") or {}
        tools_enabled_default = bool(tools_state.get("enabled"))
        selected_tools_default = tools_state.get("tools") or []

        tools_enabled = st.checkbox(
            "Enable tool calling",
            value=tools_enabled_default,
            help="Allow this agent to call tools such as web search.",
        )

        available_tools = ["web_search"]
        selected_tools: List[str] = []
        if tools_enabled:
            selected_tools = st.multiselect(
                "Allowed tools",
                options=available_tools,
                default=[
                    t for t in selected_tools_default if t in available_tools
                ],
                help="For now only web_search is available.",
            )

        st.markdown("### Reasoning")

        effort_options = ["none", "low", "medium", "high", "xhigh"]
        current_effort = state.get("reasoning_effort") or "medium"
        reasoning_effort_val = st.selectbox(
            "Reasoning Effort",
            options=effort_options,
            index=effort_options.index(current_effort)
            if current_effort in effort_options
            else effort_options.index("medium"),
            help="Controls depth & cost. 'none' disables special reasoning behavior.",
        )

        summary_options = ["auto", "concise", "detailed", "none"]
        current_summary = state.get("reasoning_summary") or "none"
        reasoning_summary_val = st.selectbox(
            "Reasoning Summary",
            options=summary_options,
            index=summary_options.index(current_summary)
            if current_summary in summary_options
            else summary_options.index("none"),
            help="Configure reasoning summaries returned by reasoning models.",
        )

    st.divider()

    # -------------------------
    # Reflect edits into state
    # -------------------------
    if not state_changed_this_run:
        state.update(
            {
                "name": name_val.strip(),
                "model": model_val.strip() or "gpt-4.1-nano",
                "role": role_val.strip(),
                "prompt": prompt_val,
                "input_vars": [
                    v.strip() for v in input_vars_val.splitlines() if v.strip()
                ],
                "output_vars": [
                    v.strip() for v in output_vars_val.splitlines() if v.strip()
                ],
                "color": color_val or DEFAULT_COLOR,
                "symbol": symbol_val or UI_COLORS[selected_color_key]["symbol"],
                "tools": {
                    "enabled": tools_enabled,
                    "tools": selected_tools if tools_enabled else [],
                },
                "reasoning_effort": reasoning_effort_val,
                "reasoning_summary": reasoning_summary_val,
            }
        )

    # Reset the persistent flag at the end of the render
    st.session_state.agent_editor_state_changed_this_run = False

    col_a, col_b, col_c, col_d = st.columns(4)

    # -------------------------
    # Save button
    # -------------------------
    with col_a:
        if st.button("💾 Save"):
            old_name = state["selected_name"] if not is_new else ""
            new_name = state["name"].strip()

            if not new_name:
                st.error("Name cannot be empty.")
                return

            # Rename agent if needed
            if not is_new and old_name != new_name:
                try:
                    agent_svc.rename_agent_atomic(old_name, new_name)
                except Exception:
                    logger.exception("Failed to rename agent")
                    st.error("Rename completed but failed to save to database")
                invalidate_agents()

            previous_prompt = ""
            if not is_new:
                previous_agent = next(a for a in agents if a["name"] == old_name)
                previous_prompt = previous_agent["prompt"]

            agent_svc.save_agent_atomic(
                new_name,
                state["model"],
                state["prompt"],
                state["role"],
                state["input_vars"],
                state["output_vars"],
                color=state.get("color") or DEFAULT_COLOR,
                symbol=state.get("symbol") or DEFAULT_SYMBOL,
                save_prompt_version=(state["prompt"] != previous_prompt),
                tools=state.get("tools")
                or {"enabled": False, "tools": []},
                reasoning_effort=state.get("reasoning_effort"),
                reasoning_summary=state.get("reasoning_summary"),
            )

            invalidate_agents()
            reload_agents_into_engine()
            st.success("Agent saved")
            st.rerun()

    # -------------------------
    # Duplicate button
    # -------------------------
    with col_b:
        if not is_new and st.button("📄 Duplicate"):
            try:
                agent_svc.save_agent(
                    f"{state['name']}_copy",
                    state["model"],
                    state["prompt"],
                    state["role"],
                    state["input_vars"],
                    state["output_vars"],
                    color=state.get("color") or DEFAULT_COLOR,
                    symbol=state.get("symbol") or DEFAULT_SYMBOL,
                    tools=state.get("tools")
                    or {"enabled": False, "tools": []},
                    reasoning_effort=state.get("reasoning_effort"),
                    reasoning_summary=state.get("reasoning_summary"),
                )
            except Exception:
                logger.exception("Failed to duplicate agent")
                st.error("Agent duplicated but failed to save to database")
            invalidate_agents()
            reload_agents_into_engine()
            st.success("Duplicated")
            st.rerun()

    # -------------------------
    # Delete button
    # -------------------------
    with col_c:
        if not is_new and st.button("🗑 Delete"):
            try:
                agent_svc.delete_agent_atomic(state["name"])
            except Exception:
                logger.exception("Failed to delete agent")
                st.error("Failed to delete agent from database")
            invalidate_agents()
            reload_agents_into_engine()
            st.warning("Deleted")
            st.rerun()

    # -------------------------
    # Load from Template (no DB persistence)
    # -------------------------
    if is_new:
        with col_d:
            st.markdown("**Load from Template**")
            uploaded = st.file_uploader(
                "Upload agent template JSON",
                type=["json"],
                key="agent_template_file",
                help="Upload a JSON file containing one or more agents",
            )

            template_agents: List[dict] = []
            if uploaded is not None:
                try:
                    tpl_data = json.loads(uploaded.read().decode("utf-8"))
                    st.session_state["agent_template_data"] = tpl_data
                except Exception as e:
                    st.error(f"Failed to parse JSON: {e}")
                    st.session_state["agent_template_data"] = None

            tpl_data = st.session_state.get("agent_template_data")
            if tpl_data:
                if isinstance(tpl_data, dict) and "agents" in tpl_data:
                    template_agents = tpl_data["agents"]
                elif isinstance(tpl_data, dict):
                    template_agents = [tpl_data]
                elif isinstance(tpl_data, list):
                    template_agents = tpl_data

                template_agents = [
                    a
                    for a in template_agents
                    if isinstance(a, dict)
                    and all(
                        k
                        in a
                        for k in (
                            "name",
                            "model",
                            "prompt_template",
                            "input_vars",
                            "output_vars",
                        )
                    )
                ]

            chosen_agent = None
            if template_agents:
                if len(template_agents) == 1:
                    chosen_agent = template_agents[0]
                else:
                    names_tpl = [a["name"] for a in template_agents]
                    chosen_name = st.selectbox(
                        "Choose agent from template",
                        names_tpl,
                        key="agent_template_choose_name",
                    )
                    chosen_agent = next(
                        a for a in template_agents if a["name"] == chosen_name
                    )

            if chosen_agent and st.button(
                "Apply Template", key="agent_template_apply"
            ):
                original_name = str(chosen_agent.get("name", "")).strip() or "imported_agent"
                suffix = datetime.now().strftime("_%y%m%d-%H%M_imported")
                imported_name = f"{original_name}{suffix}"

                state["selected_name"] = "<New Agent>"

                state.update(
                    {
                        "name": imported_name,
                        "model": chosen_agent.get("model", "gpt-4.1-nano"),
                        "role": chosen_agent.get("role", ""),
                        "prompt": chosen_agent.get("prompt_template", ""),
                        "input_vars": chosen_agent.get("input_vars", []),
                        "output_vars": chosen_agent.get("output_vars", []),
                        "color": DEFAULT_COLOR,
                        "symbol": DEFAULT_SYMBOL,
                    }
                )

                st.session_state.agent_editor_state_changed_this_run = True

                st.success(
                    "Template loaded into editor. Review and click Save to store the agent."
                )
                st.rerun()


# ======================================================
# HISTORY MODE
# ======================================================

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

    # Shared cost & latency rendering for stored metrics
    metrics_view = metrics_view_from_db_rows(metrics)
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

        agent_config = {
            "model": cfg.get("model") or a.get("model") or "unknown",
            "role": cfg.get("role") or "–",
            "tools": {
                "enabled": bool(tools_json.get("enabled")),
                "tools": tools_json.get("tools") or [],
            },
            "reasoning": {
                "effort": cfg.get("reasoning_effort") or "default",
                "summary": cfg.get("reasoning_summary") or "none",
            },
            "raw": {
                "tools_json": tools_json or None,
                "tools_config_json": tools_cfg_json or None,
                "reasoning_config_json": reasoning_cfg_json or None,
                "extra_config_json": extra_cfg_json or None,
            },
        }

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
        data=json.dumps(export, indent=2),
        file_name=f"run_{run_id}.json",
        mime="application/json",
    )


# ======================================================
# LOGS MODE
# ======================================================

def render_logs_mode():
    st.header("🪵 Application Logs")

    logs = st.session_state.get("_log_buffer", [])

    if not logs:
        st.info("No logs yet.")
        return

    inject_global_tag_style()

    col1, col2 = st.columns([3, 1])

    with col2:
        LEVEL_LABELS = {
            level: f"{style['symbol']} {level}"
            for level, style in LOG_LEVEL_STYLES.items()
        }

        level_filter = st.multiselect(
            "Levels",
            list(LEVEL_LABELS.keys()),
            default=["INFO", "WARNING", "ERROR", "CRITICAL"],
            format_func=lambda v: LEVEL_LABELS[v],
        )

        def build_log_lines(
            logs_list: List[Dict[str, Any]],
            level_filter_list: List[str],
            search_str: str,
        ):
            lines: List[str] = []
            for entry in logs_list:
                if entry["level"] not in level_filter_list:
                    continue
                if search_str and search_str.lower() not in entry[
                    "message"
                ].lower():
                    continue

                lines.append(
                    f"{entry['time']} "
                    f"[{entry['level']}] "
                    f"{entry['logger']}: "
                    f"{entry['message']}"
                )
            return "\n".join(lines)

        search = st.text_input("Search")

        if st.button("🧹 Clear logs"):
            logs.clear()
            st.rerun()

        export_text = build_log_lines(
            logs_list=list(logs),
            level_filter_list=level_filter,
            search_str=search,
        )

        st.download_button(
            label="⬇️ Download logs",
            data=export_text,
            file_name="application.log",
            mime="text/plain",
            disabled=not bool(export_text),
        )

    with col1:
        for entry in reversed(logs):
            if entry["level"] not in level_filter:
                continue
            if search and search.lower() not in entry["message"].lower():
                continue

            level = entry["level"]
            style = LOG_LEVEL_STYLES.get(
                level, {"color": "#000000", "symbol": ""}
            )
            color = style["color"]

            prefix = f"{entry['time']} "
            level_token = f"[{level}]"
            suffix = f" {entry['logger']}: {entry['message']}"

            st.markdown(
                f"""
            <pre style="
                margin: 0;
                padding: 6px 10px;
                background-color: #f8f9fa;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
            ">
{prefix}<span style="color:{color}; font-weight:bold;">{level_token}</span>{suffix}
            </pre>
            """,
                unsafe_allow_html=True,
            )


# ======================================================
# MODE ROUTER
# ======================================================

if mode == MODE_RUN:
    render_run_mode()
elif mode == MODE_AGENTS:
    render_agent_editor()
elif mode == MODE_HISTORY:
    render_history_mode()
elif mode == MODE_LOGS:
    render_logs_mode()
