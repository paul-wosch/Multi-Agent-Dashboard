# ui/app.py
import json
import logging
import time
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from multi_agent_dashboard.config import (
    DB_FILE_PATH,
    UI_COLORS,
    configure_logging,
)
from multi_agent_dashboard.engine import EngineResult
from multi_agent_dashboard.ui.utils import parse_json_field
from multi_agent_dashboard.ui.view_models import (
    AgentConfigView,
    metrics_view_from_engine_result,
    metrics_view_from_db_rows,
    config_view_from_db_rows,
)
from multi_agent_dashboard.ui.metrics_view import (
    render_cost_latency_section,
)
from multi_agent_dashboard.ui.tools_view import (
    render_agent_config_section,
    build_tool_calls_overview,
)
from multi_agent_dashboard.ui.graph_view import render_agent_graph
from multi_agent_dashboard.ui.exports import (
    export_pipeline_agents_as_json,
    build_export_from_engine_result,
)

# Imports moved to dedicated modules (Phase 3)
from multi_agent_dashboard.ui.cache import (
    cached_load_agents,
    cached_load_pipelines,
    cached_load_runs,
    cached_load_run_details,
    cached_load_prompt_versions,
    invalidate_agents,
    invalidate_pipelines,
    invalidate_runs,
    get_agent_service,
    get_pipeline_service,
    get_run_service,
)
from multi_agent_dashboard.ui.bootstrap import app_start, reload_agents_into_engine
from multi_agent_dashboard.ui.logging_ui import LOG_LEVEL_STYLES

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

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_TOTAL_SIZE = 20 * 1024 * 1024  # 20 MB
AD_HOC_PIPELINE_LABEL = "<Ad-hoc>"


# ======================================================
# SMALL HELPERS
# ======================================================


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


# ======================================================
# INITIALIZE DB AND PREPARE ENGINE
# ======================================================

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

def pipeline_requires_files(engine, steps) -> bool:
    for name in steps:
        agent = engine.agents.get(name)
        if not agent:
            continue
        if "files" in agent.spec.input_vars:
            return True
    return False


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
                    # use getter to obtain pipeline service lazily
                    get_pipeline_service().save_pipeline(pipeline_name, selected_steps)
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
    # render_agent_graph reads engine from session_state internally
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
            # use getter to obtain run service lazily
            get_run_service().save_run(
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
                    get_agent_service().rename_agent_atomic(old_name, new_name)
                except Exception:
                    logger.exception("Failed to rename agent")
                    st.error("Rename completed but failed to save to database")
                invalidate_agents()

            previous_prompt = ""
            if not is_new:
                previous_agent = next(a for a in agents if a["name"] == old_name)
                previous_prompt = previous_agent["prompt"]

            get_agent_service().save_agent_atomic(
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
                get_agent_service().save_agent(
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
                get_agent_service().delete_agent_atomic(state["name"])
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
