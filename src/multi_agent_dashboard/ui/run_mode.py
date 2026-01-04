# ui/run_mode.py
from __future__ import annotations

import json
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from multi_agent_dashboard.config import UI_COLORS, ATTACHMENT_FILE_TYPES, ATTACHMENT_FILE_TYPES_RESTRICTED
from multi_agent_dashboard.engine import EngineResult
from multi_agent_dashboard.ui.cache import (
    cached_load_pipelines,
    get_pipeline_service,
    get_run_service,
    invalidate_runs,
    invalidate_pipelines,
)
from multi_agent_dashboard.ui.exports import (
    export_pipeline_agents_as_json,
    build_export_from_engine_result,
)
from multi_agent_dashboard.ui.metrics_view import (
    render_cost_latency_section,
    format_cost,
    format_latency,
)
from multi_agent_dashboard.ui.tools_view import (
    render_agent_config_section,
    build_tool_calls_overview,
)
from multi_agent_dashboard.ui.graph_view import render_agent_graph
from multi_agent_dashboard.ui.styles import inject_tag_style_for_sidebar

logger = logging.getLogger(__name__)

DEFAULT_COLOR = UI_COLORS["default"]["value"]
DEFAULT_SYMBOL = UI_COLORS["default"]["symbol"]

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_TOTAL_SIZE = 20 * 1024 * 1024  # 20 MB
AD_HOC_PIPELINE_LABEL = "<Ad-hoc>"


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
        from multi_agent_dashboard.ui.cache import cached_load_agents
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
            # Determine the 'type' parameter for Streamlit's file_uploader:
            # - If ATTACHMENT_FILE_TYPES_RESTRICTED is True, pass the allowed extensions list.
            # - If False, pass None to allow any file extension.
            file_types_param = ATTACHMENT_FILE_TYPES if ATTACHMENT_FILE_TYPES_RESTRICTED else None

            uploaded_files = st.file_uploader(
                "Upload files",
                accept_multiple_files=True,
                type=file_types_param,
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
                    get_pipeline_service().save_pipeline(pipeline_name, selected_steps)
                except Exception:
                    logger.exception("Failed to persist pipeline")
                    st.error("Failed to save pipeline to database")
                # Invalidate pipelines cache so saved pipelines show up immediately
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
    text: Any,
    *,
    is_json_hint: Optional[bool] = None,
    key_prefix: str,
):
    """
    Shared viewer: try JSON, else Markdown/Code toggle.

    Robust handling: 'text' may be:
      - a Python dict/list (already parsed JSON) -> display via st.json
      - a JSON string -> try to json.loads and display as JSON
      - any other string -> show Markdown/Code toggle
      - None -> show "No output."
    """
    with st.expander(label, expanded=(label.lower().startswith("final output"))):
        if text is None:
            st.info("No output.")
            return

        # If the output is already a Python object (dict/list), show it as JSON directly
        if isinstance(text, (dict, list)):
            st.json(text)
            return

        # If a JSON hint was provided, attempt to parse; otherwise still attempt parsing but be lenient
        if is_json_hint:
            try:
                parsed = json.loads(text) if isinstance(text, str) else None
                if parsed is not None:
                    st.json(parsed)
                    return
            except Exception:
                st.warning("⚠️ Marked as JSON but failed to parse; falling back to text view.")

        # Try a best-effort parse: if the stored text looks like JSON, render it
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                st.json(parsed)
                return
            except Exception:
                # Not JSON — fall back to Markdown/Code view
                pass

        # Fallback to Markdown/Code toggle for any other text
        view = st.radio(
            "View as",
            ["Markdown", "Code"],
            horizontal=True,
            key=f"{key_prefix}_view",
        )
        if view == "Markdown":
            st.markdown(str(text))
        else:
            st.code(str(text))


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
            width="stretch",
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
    st.graphviz_chart(render_agent_graph(steps, result.agent_metrics, engine=st.session_state.engine))


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
    metrics_view = []
    # Keep simple: construct metrics view using engine result mapping
    from multi_agent_dashboard.ui.view_models import metrics_view_from_engine_result
    metrics_view = metrics_view_from_engine_result(result, steps)
    render_cost_latency_section(metrics_view, title_suffix="This run")


def render_tools_advanced_tab(result: EngineResult, steps: List[str]):
    engine = st.session_state.engine
    tool_usages = result.tool_usages or {}

    tool_usages_by_agent: Dict[str, List[dict]] = {
        name: tool_usages.get(name) or [] for name in steps
    }

    # Build per-agent config view from live result
    config_view = config_view_from_engine_result(result, steps)
    render_agent_config_section(
        config_view,
        tool_usages_by_agent,
        title_suffix="this run",
        is_historic=False,
    )

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
) -> List[Any]:
    engine = st.session_state.engine
    views: List[Any] = []

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

        from multi_agent_dashboard.ui.view_models import AgentConfigView

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
                # Expose both prompt templates from the live AgentSpec
                prompt_template=getattr(spec, "prompt_template", None),
                system_prompt_template=getattr(spec, "system_prompt_template", None),
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


def render_run_mode(strict_mode: bool = False):
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
            engine=st.session_state.engine,
        )

        render_run_results(
            last_result,
            last_steps,
            current_run_export=current_run_export,
        )
